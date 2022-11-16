import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import RandomRotation
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import CenterCrop

from data_fc import MultiFolderDataset
from unet import TurbNetG_Linear_Plume
from tqdm import tqdm
from datetime import datetime
from plot import plot_multi_comparison
import numpy as np
import random
import yaml
import math
import time
import matplotlib.pyplot as plt
from utils import *

with open("train.yml", "r") as stream:
    config = yaml.safe_load(stream)
    
# PARAMETERS
n_epochs = config["n_epochs"]
lr = config["lr"]
res_loss_weight = config["res_loss_weight"]
lra_alpha = config["lra_alpha"]
batch_size = config["batch_size"]
write_freq = config["write_freq"]
physical_loss = config["physical_loss"]
imsize = config["imsize"]
total_batch_groups = config["total_batch_groups"]
width_plume = config["width_plume"]
length_plume = config["length_plume"]
length_scaling = config["length_scaling"]
width_scaling = config["width_scaling"]
base_path = config["base_path"]
data_path = config["data_path"]
view_image = config["view_image"]

base_dir = str(base_path)	#Set this path to the current folder with the model and dataset
cache_dir_local = base_dir + "cache/"
use_cache = False
distributed_training = False


def augment_data(input, target):
    seed = np.random.randint(2147483647)  # make a seed with numpy generator
    random.seed(seed)  # apply this seed to img tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    # input = trans(input)

    random.seed(seed)  # apply this seed to target tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    # target = trans(target)

    return input, target

# Automatically select the testing data folders to reproduce results
test_folders = [5,10,15,20,25,30]

folder_list = [os.path.join(str(data_path), f"batch{i+1}") for i in range(total_batch_groups)]
print("Train dataset creation ")
mf_dataset = MultiFolderDataset(folder_list, test_folders, imsize, normalize=True, test=False )
print("Test dataset creation ")
mf_dataset_test = MultiFolderDataset(folder_list, test_folders, imsize, normalize=True, test=True)

# Create the training and testing data arrays
train_dataset = mf_dataset
test_dataset = mf_dataset_test
train_size = int(len(mf_dataset))
test_size = int(len(mf_dataset_test))
assert(train_size > 0), "No training data provided. Ensure that the data exists and that the paths are correct"

#view_image = False      # If True, outputs images of the temperature input data and plume domain

temperature_train, Vmax_train, qx_train, qy_train, loc_off_plume_x_train, loc_off_plume_y_train = train_dataset.extract_plume_data(view_image, length_plume, width_plume, length_scaling, width_scaling)

if (view_image):
    for i in range(train_size):
        x = np.linspace(1, 25, 25)
        y = np.linspace(1, 25, 25)
        X, Y = np.meshgrid(x, y)
        Temp = temperature_train[i,:,:]
        cp = plt.contourf(X, Y, Temp, levels=[11,12,13,14,15],cmap='viridis')
        plt.colorbar(cp)
        #plt.imshow(H, interpolation='none',cmap='jet')
        plt.show()

if (test_size > 0):
    temperature_test, Vmax_test, qx_test, qy_test, loc_off_plume_x_test, loc_off_plume_y_test = test_dataset.extract_plume_data(view_image)