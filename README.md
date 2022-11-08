# Instructions for training and post-processing for the shallow geothermal temperature prediction

## Download

This folder contains all python files and data to train a convolutional neural network to predict the temperature profile due to the presence of a groundwater heat pump. The idea for the temperature plume prediction is provided here: https://arxiv.org/pdf/2203.14961.pdf. 

## Data

Please see the data repository here: https://doi.org/10.18419/darus-3184.
All data necessary for training the networks are provided in "data.tar.gz". This is divided into 32 batches of 25 samples each. Unzip the file to access the data.

## Setup

The first step in reproducing the results is to set the source code path and data path in "train.yml". Various other parameters are adjusted in the train.yml file:

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

Each convolutional neural network type can be trained using the three scripts provided: train_model_TurbNetGeo.py, train_model_TurbNetGeo_Light.py and train_model_TurbNetGeo_NoSkip_Light.py. Simply run one of the python scripts via
```
python3 train_model_TurbNetGeo.py
```

This will load the training and testing data and begin training the network according to the specifications in `train.yml`. The test samples that are excluded from the training data are batchs: 5, 10, 15, 20, 25 and 30. All other batches were used for training. This is hard-coded in the provided model files. 

## Results 

Results for three network architectures trained over 50,000 epochs are provided in "Results.tar.gz". After unzipping the file, within are the folder "Results/TNG", "Results/TNG-L" and "Results/TNG-NS-L". Each are further divided based on the channelExponent value. Each training results folder contains the `train.yml` file used for training. 
The models can be loaded into tensorboard (https://www.tensorflow.org/tensorboard/), where the visualisation output for all testing samples can be viewed under the "images" heading. 

## Software Dependency

The following software versions were used in running the experiments:
1. Python - 3.8.10
2. matplotlib - 3.4.2
3. numpy - 1.20.3
4. pytorch - 1.8.0+cu111
