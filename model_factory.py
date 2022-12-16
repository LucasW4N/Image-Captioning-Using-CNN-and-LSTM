################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)
        self.batchnorm64 = nn.BatchNorm2d(num_features=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.batchnorm128 = nn.BatchNorm2d(num_features=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm256 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # batchnorm256
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        # batchnorm256
        # maxpool2
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=outputs)

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''

        # first conv layer
        x = self.conv1(x)
        x = F.relu(self.batchnorm64(x))
        x = self.maxpool1(x)
        # second conv layer
        x = self.conv2(x)
        x = F.relu(self.batchnorm128(x))
        x = self.maxpool2(x)
        # conv layer 3 - 5
        x = self.conv3(x)
        x = F.relu(self.batchnorm256(x))
        x = self.conv4(x)
        x = F.relu(self.batchnorm256(x))
        x = self.conv5(x)
        x = F.relu(self.batchnorm128(x))
        x = self.maxpool2(x)
        x = self.adaptive_avgpool(x)
        
        # flatten the result so that fc can work
        x = torch.flatten(x, 1)
        
        # fullly connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']
        self.layer = config_data['model']['layer']
        self.customCNN = CustomCNN(self.embedding_size)
        self.embedding_layer = nn.Embedding(len(self.vocab), self.embedding_size)
        self.linear = nn.Linear(self.hidden_size, len(self.vocab))
        

        self.resnet_50 = resnet50(pretrained = True, progress = True)
        for param in self.resnet_50.parameters():
            param.requires_grad = False
        penultimate_hidden_layer_dim = self.resnet_50.fc.in_features
        self.resnet_50.fc = nn.Linear(penultimate_hidden_layer_dim, self.embedding_size)

        self.lstm = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
            num_layers = self.layer,
            batch_first = True
        )


    def forward(self, images, captions, teacher_forcing=True):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        
        #assert captions.shape[1] == self.max_length
        batch_size = images.shape[0]
        caption_length = captions.shape[1]
        vocab_size = len(self.vocab)

        outs = torch.zeros(size = (batch_size, caption_length, vocab_size)).cuda() #Size: batch_size * caption_length * vocab_size
        sampled_captions = torch.zeros(size = (batch_size, caption_length)).int().cuda() #Size: batch_size * caption_length
        
        embedded_images = self.customCNN(images) if self.model_type == "Custom" else self.resnet_50(images) #Size: batch_size * embedding_size
        embedded_images = embedded_images[:, None, :] #Size: batch_size * 1 * embedding_size, adding the "L" (input sequence length) dimension
        out, hidden_states = self.lstm(embedded_images) #Size_out: batch_size * 1 * hidden_size #LSTM output, and the hidden states for the next time step
        out = self.linear(out) #Size: batch_size * 1 * vocab_size (Linear layer transform hidden_size to vocab_size)
        out = torch.squeeze(out) #Size: batch_size * vocab_size
        outs[:, 0, :] = out #Store the out results for all images, words number 0

        if not teacher_forcing: #If testing, sample the word from the out result for all images
            sampled_captions[:, 0] = sample_words_from_distribution(out, self.temp, self.deterministic)
        
        input_words = None
        for i in range(1, caption_length): #For later time steps
            input_words = captions[:, i - 1] if teacher_forcing else sampled_captions[:, i - 1] #If training, use ground truth caption. Else use the last sampled caption
        
            embedded_words = self.embedding_layer(input_words) #Size: batch_size * embedding_size
            embedded_words = embedded_words[:, None, :] #Size: batch_size * 1 * embedding_size, adding the "L" (input sequence length) dimension
            out, hidden_states = self.lstm(embedded_words, hidden_states) #Size_out: batch_size * 1 * hidden_size
            out = self.linear(out) #Size: batch_size * 1 * vocab_size
            out = torch.squeeze(out) #Size: batch_size * vocab_size
            outs[:, i, :] = out #Store the out results for all images, words number i

            if not teacher_forcing: #If testing, sample the word from the out result for all images
                sampled_captions[:, i] = sample_words_from_distribution(out, self.temp, self.deterministic)
        
        return outs, sampled_captions    

def sample_words_from_distribution(out, temp, deterministic):
    sampled_words = None
    if deterministic:
        sampled_words = torch.argmax(out, axis = 1) #Just pick the largest one
    else:
        m = nn.Softmax(dim = 1)
        distribution = m(out / temp)
        sampled_words = torch.squeeze(torch.multinomial(distribution, num_samples = 1)) #Sample one result
    return sampled_words

def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)
