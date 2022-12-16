################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
from nltk.tokenize import word_tokenize
import caption_utils

ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

from PIL import Image

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__batch_size = config_data['dataset']['batch_size']
        
        self.__temperature = config_data["generation"]["temperature"]

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__best_model = deepcopy(self.__model.state_dict())

        # criterion
        self.__criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        self.__learning_rate = config_data["experiment"]["learning_rate"]
        if config_data["experiment"]["optimizer"] == "Adam":
            self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__learning_rate)

        # LR Scheduler
        if config_data["experiment"]["lr_scheduler"] == "steplr":
            self.__lr_scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, 
                                                                  step_size=self.__epochs / 3)


        # for find_examples()
        images_root_dir = config_data['dataset']['images_root_dir']
        self.__test_dir = os.path.join(images_root_dir, 'test')
        
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()


    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train()
            print('Validating...')
            print('-------------')
            val_loss = self.__val()

            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = deepcopy(self.__model.state_dict())

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()
            
        self.__model.load_state_dict(self.__best_model)

    def __compute_loss(self, images, captions):
        """
        Computes the loss after a forward pass through the model
        """
        raise NotImplementedError()

    def __train(self):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        
        total_loss = 0.0
        # iterate through data in train loader, each iteration is a minibatch
        for i,(images, captions, _) in tqdm(enumerate(self.__train_loader), total=len(self.__train_loader)):
            images, captions = images.cuda(), captions.cuda()
            
            # zero the parameter gradients
            self.__optimizer.zero_grad()
            
            # forward + backward
            outputs, _ = self.__model(images, captions, teacher_forcing=True)   # forward()
            outputs.cuda()
            outputs_flat = outputs.flatten(0, 1).cuda()
            captions_flat = captions.flatten(0, 1).cuda()
            loss = self.__criterion(outputs_flat, captions_flat)    # calculate loss
            loss.backward()            # calculating gradients
            self.__optimizer.step()    # updating weights
            
            # calculate loss of this minibatch
            total_loss += loss.item()
        
        return total_loss / len(self.__train_loader)

    def __sampled_to_string(self, sampled_caption):
        captions = []
        for i in range(sampled_caption.shape[0]): #number of images
            caption = []
            for j in range(sampled_caption.shape[1]): #caption length
                id = sampled_caption[i, j].item()
                if id >= 4:
                    caption.append(self.__vocab.idx2word[id])
            captions.append(caption)
        return captions

    def __generate_captions(self, i, img_id, outputs, testing):
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            outputs: output from the forward pass for this img_id
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        
        ann_ids = []
        for j in range(len(self.__coco_test.imgToAnns[img_id])):
            ann_ids.append(self.__coco_test.imgToAnns[img_id][j]['id'])
        
        # get the tokenized target captions
        target_captions = [self.__coco_test.anns[ann_id]['caption'] for ann_id in ann_ids]
        target_captions = [word_tokenize(str(c).lower()) for c in target_captions]
        output_caption = outputs[i]
        
        return target_captions, output_caption
    

    def __str_captions(self, img_id, original_captions, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nActual: {},\nPredicted: {}\n".format(
            img_id, original_captions, predicted_caption)
        return result_str

    def __val(self):
        """
        Validate the model for one epoch using teacher forcing
        """
        
        total_loss = 0.0
        with torch.no_grad():   # prevent pytorch from calculating gradients
            for i, (images, captions, _) in tqdm(enumerate(self.__val_loader), total=len(self.__val_loader)):
                images, captions = images.cuda(), captions.cuda()
                outputs, _ = self.__model(images, captions, teacher_forcing=True)
                outputs.cuda()
                outputs_flat = outputs.flatten(0, 1).cuda()
                captions_flat = captions.flatten(0, 1).cuda()
                loss = self.__criterion(outputs_flat, captions_flat)
                total_loss += loss.item()
        
        return total_loss / len(self.__val_loader)

    def test(self):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        
        self.__model.eval()
        
        # variables for loss and BLEU calculation
        total_loss = 0.0
        bleu1 = 0.0
        bleu4 = 0.0
        bleu_count = 0
        
        with torch.no_grad():
            for _, (images, captions, img_ids) in tqdm(enumerate(self.__test_loader), total=len(self.__test_loader)):
                images, captions = images.cuda(), captions.cuda()
                
                # two forward passes, one for calculating loss, one for calculating BLEU scores
                outputs, _ = self.__model(images, captions, teacher_forcing=True)
                _, sampled_captions = self.__model(images, captions, teacher_forcing=False)
                
                # ----- calculate loss -------
                outputs.cuda()
                outputs_flat = outputs.flatten(0, 1).cuda()
                captions_flat = captions.flatten(0, 1).cuda()
                loss = self.__criterion(outputs_flat, captions_flat)
                total_loss += loss.item()
                
                # load the output captions in string
                output_captions = self.__sampled_to_string(sampled_captions)
                
                # ----- calculate BLEU scores -------
                for i in range(0, len(img_ids)):
                    # get the tokenized target captions and predict caption
                    target_captions, output_caption = self.__generate_captions(i, img_ids[i], output_captions, testing=True)
                    
                    bleu1 += caption_utils.bleu1(target_captions, output_caption)
                    bleu4 += caption_utils.bleu4(target_captions, output_caption)
                    bleu_count += 1
                
        bleu1 /= bleu_count
        bleu4 /= bleu_count
        test_loss = total_loss / len(self.__test_loader)
        
        # ----- print the result ------
        print("testing is done!")
        result = "Test Loss: {}, Bleu1: {}, Bleu4: {}, Temperature: {}".format(test_loss, bleu1, bleu4, self.__temperature)
        self.__log(result)
        
        
        
    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()

    def find_examples(self):
        self.__load_experiment()
        self.__model.eval()
        
        print("---------- looking for examples ---------")
        tuple_list = []     # list of tuple(BLEU1, image_id, output_caption)
        # do the forward pass on test dataset to get the tuple list
        with torch.no_grad():
            for _, (images, captions, img_ids) in tqdm(enumerate(self.__test_loader), total=len(self.__test_loader)):
                images, captions = images.cuda(), captions.cuda()
                # get default captions
                _, output_ids = self.__model(images, captions, teacher_forcing=False)
                # load the output captions in string
                output_captions = self.__sampled_to_string(output_ids)
                # ----- calculate BLEU scores -------
                for i in range(0, len(img_ids)):
                    img_id = img_ids[i]
                    tokened_target_captions, output_caption = self.__generate_captions(i, img_id, output_captions, testing=True)
                    score = caption_utils.bleu1(tokened_target_captions, output_caption)
                    # push the desired values into tuple_list
                    tuple_list.append((score, img_ids[i], output_caption))

        # ------ find the good and the bad imgs with captions ------
        tuple_list.sort()
        
        print("---------- Five Worst Images with Captions:")
        
        count = 0
        i = 0
        previous_id = -1
        while count < 5:
            _, img_id, output_caption = tuple_list[i]
            # pass if meet duplicate
            if (img_id == previous_id):
                i += 1
                continue
            
            self.show_example(img_id, output_caption)
            
            # update variables
            previous_id = img_id
            i += 1
            count += 1
        
        print("---------- Five Best Images with Captions:")
        
        count = 0
        i = len(tuple_list) - 1
        previous_id = -1
        while count < 5:
            _, img_id, output_caption = tuple_list[i]
            # pass if meet duplicate
            if (img_id == previous_id):
                i -= 1
                continue
            
            self.show_example(img_id, output_caption)
            
            # update variables
            previous_id = img_id
            i -= 1
            count += 1
            

    # -------------- helper funcitons -------------------
    
    def show_img(self, img_name):
        dir = os.path.join(self.__test_dir, img_name)
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, dir)
        img = Image.open(abs_file_path)
        img.show()
    
    
    def get_untokenized_str(self, output_caption):
        result = ""
        for word in output_caption:
            result += word + " "
        result += "\n"
        return result
    
    def show_example(self, img_id, output_caption):
        # show image
        ann_ids = []
        for j in range(len(self.__coco_test.imgToAnns[img_id])):
            ann_ids.append(self.__coco_test.imgToAnns[img_id][j]['id'])
        img_name = self.__coco_test.loadImgs([img_id])[0]['file_name']
        self.show_img(img_name)
        # print target captions
        target_captions = [self.__coco_test.anns[ann_id]['caption'] for ann_id in ann_ids]
        print("---------- target captions for image", img_id, ":")
        for cap in target_captions:
            print(cap)
        # print output caption
        print("---------- output caption for image with default temperature:")
        print(self.get_untokenized_str(output_caption))
        
    
    # prints the captions of the previous examples
    def print_captions(self):
        # TODO: hard code the following set to find the corresponding captions
        set = {377706, 173500, 519838, 580720, 579635, 578324}
        
        print("---------- looking for examples ---------")
        
        self.__load_experiment()
        self.__model.eval()
        
        tuple_list = []     # list of tuple(imgid, output_caption)
        with torch.no_grad():
            for _, (images, captions, img_ids) in tqdm(enumerate(self.__test_loader), total=len(self.__test_loader)):
                images, captions = images.cuda(), captions.cuda()
                
                # get default captions
                _, output_ids = self.__model(images, captions, teacher_forcing=False)
                
                # load the output captions in string
                output_captions = self.__sampled_to_string(output_ids)
                
                # ----- calculate BLEU scores -------
                for img in range(0, len(img_ids)):
                    id = img_ids[img]
                    # if id == 138589 or id == 211069 or id == 99218 or id == 393056 or id == 513054 or id == 579635:
                    if id in set:
                        output_caption = output_captions[img]
                        # tuple(imgid, output_caption)
                        tuple_list.append((id, output_caption))
                        
        tuple_list.sort()
        for (id, output_caption) in tuple_list:
            print("----- output caption for", id, "is: ")
            print(self.get_untokenized_str(output_caption))
