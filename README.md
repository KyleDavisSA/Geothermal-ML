# Instructions for training and post-processing for the shallow geothermal temperature prediction

## Download

This folder contains all python files to gather the temperature data along the streamline of the thermal plume that runs throught the heat pump.  

## Data

Please see the data repository here: https://doi.org/10.18419/darus-3184. 
All data necessary for training the networks are provided in "data.tar.gz". This is divided into 32 batches of 25 samples each. Unzip the file to access the folder `data`.

## Setup

Add the folder `data` form above into this projects home directory. Next, set the source code path and data path in "train.yml". Various other parameters are adjusted in the train.yml file:

Note: Some of the parameters below are for training a convolutional neural network and are not required for data sampling. 
data_augmentation: if set to True, then the images are randomly rotated to generate more training samples.
n_epochs: total number of training epochs.
lr: sets the learning rate.
lra_alpha: learning rate alpha.
channelExponent: The number of initial features is adjusted by varying the "channelExponent" value. The number of initial features is calculated by 2**channelexponent. Thererfore, channelExponent=2 equals 4 initial features, channelExponent=3 equals 8 initial features etc.
batch_size: batch size number (https://pytorch.org/docs/stable/data.html?highlight=batch%20size).
write_freq: how many epochs the training and testing results are written to storage.
physical_loss: false - no physics informed loss function is used.
imsize: number of pixels in each direction of the input and outputs images.
total_batch_groups: the total number of sample groups used for training can be set in the train.yml file by varying the "total_batch_groups" setting, which specifies the number of batch folders to use. 
base_path: path to source code
data_path: path to data where the batch folders are stored


## Run

Extracting the data along the thermal plume is done by running:
```
python3 train_model_fc.py
```
Note that if more than 5 batches are set in the `train.yaml` file, then the folders `5, 10, 15, 20, 25, 30` will be seperated into a testing list. Extracting the testing data has not been implemented yet. 


## Software Dependency

The following software versions were used in running the experiments:
1. Python - 3.8.10
2. matplotlib - 3.4.2
3. numpy - 1.20.3
4. pytorch - 1.8.0+cu111
