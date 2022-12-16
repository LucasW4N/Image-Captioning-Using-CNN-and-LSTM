# PA 3: Image Captioning


## Contributors
- Jared Zhang
- Lucas Wan


## Task
- In Task 1, we assembled a Convolution Neural Network and an LSTM using PyTorch modules. We tried to optimize the CNN + LSTM model by adjusting different hyperparameters.
  * We halved the size of hidden units in each layer. We got similar BLEU scores.
  * We added a hidden layer in our LSTM structure. We obtained slightly better results.

- In Task 2, we replaced the custom CNN built in the previous Task with a ResNet-50 pre-trained on ImageNet to transfer its knowledge to the dataset we are using here. Then we made 2 changes to the hyperparameters and discussed the results in each case.
  * We switch to SGD optimizer instead of the default Adam optimizer. We got an obviously worse result by using SGD
  * We noticed some overfitting during the default config, so we changed the learning rate from 5e-4 to 2e-4. The BLEU scores improved.


## How to run
- If you want to train, test, and see examples (5 good, and 5 bad in our case) according to your defined config:
  - Run `python3 main.py your_config`
- Otherwise, comment out the part you don't want in main.py and run.
  - For instance, comment out `exp.find_examples()` in main.py if you don't want to see examples.

## Usage

* Define the configuration for your experiment. See `task-1-default-config.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training or evaluate performance.

## Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace
