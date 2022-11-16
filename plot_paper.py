import yaml
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data import MultiFolderDataset
from matplotlib import cm, colors

from plot import plot_comparison, plot_comparison_plume, plot_comparison_small, plot_multi_comparison, plot_velocity_temperature_ax, plot_velocity_temperature_ax_inner

from unet import TurbNetG, TurbNetG_noSkip_Light, TurbNetG_Light

cm = 1 / 2.54  # centimeters in inches
width_cm = 17
height_cm = 17 * 0.4
fig, ax = plt.subplots(1, 1, figsize=[width_cm, height_cm])

os.system("mkdir image/good")
os.system("mkdir image/medium")
os.system("mkdir image/bad")

# Path to results of the model.pt file
model_name = "TNG/Channel_4"
model_path = f"Results/{model_name}/"
with open(model_path + "train.yml", "r") as stream:
    config = yaml.safe_load(stream)

model_dict = torch.load(model_path + "model.pt", map_location=torch.device('cpu'))
lra = config["lra"]
data_augmentation = config["data_augmentation"]
n_epochs = config["n_epochs"]
lr = config["lr"]
res_loss_weight = config["res_loss_weight"]
lra_alpha = config["lra_alpha"]
channelExponent = config["channelExponent"]
batch_size = config["batch_size"]
write_freq = config["write_freq"]
physical_loss = config["physical_loss"]
imsize = config["imsize"]
total_batch_groups = config["total_batch_groups"]

data_path = "data"

# Testing data folders
test_folders = [5,10,15,20,25,30]
folder_list = [os.path.join(str(data_path), f"batch{i+1}") for i in range(total_batch_groups)]
dataset = MultiFolderDataset(folder_list, test_folders, imsize, normalize=True, data_augmentation=False, Inner=False, test=True)

# PARAMETERS
channelExponent = config["channelExponent"]

model = TurbNetG(channelExponent=channelExponent)
model.load_state_dict(model_dict["model_state_dict"])

model.eval()

max_err = 0.0
rel_err = 0.0
count = [0,0,0]
errors_reduced = []
errors_reduced_outer = []
errors_reduced_inner = []

for i in range(dataset.dataset_size):
    model_input = dataset[i][0]
    target = dataset[i][1]

    pred = model(model_input.unsqueeze(0))
    pred = dataset.get_temp_unnormalized(pred)
    target = dataset.get_temp_unnormalized(target)

    fig, axes = plt.subplots(1, 3, figsize=(18, 18))
    errors = (target[0, :, :] - pred[0, 0, :, :]).squeeze()
    
    errors_min = errors.min()
    errors_max = errors.max()  
    
    # Calculate a new error based on the background temps
    for k in range(imsize):
        for j in range(imsize):
            if (abs(target[0,k, j]) > 10.2):
                errors_reduced.append(abs(errors[k,j]))
                if (k > 26 and k < 40 and j < 40 and j > 26):
                    errors_reduced_inner.append(abs(errors[k,j]))
                else:
                    errors_reduced_outer.append(abs(errors[k,j]))
    
    vmin = target.min()
    vmax = target.max()
    
    imsize = 65
    ax_pred = plot_velocity_temperature_ax(
        axes[0], model_input[:, :, :], pred[0, :, :].squeeze(), vmin, vmax, imsize
    )
    ax_target = plot_velocity_temperature_ax(
        axes[1], model_input[:, :, :], target[0, :, :].squeeze(), vmin, vmax, imsize
    )
    errors = (target[0, :, :] - pred[0, 0, :, :]).squeeze()
    errors_min = -errors.abs().max()
    errors_max = errors.abs().max()
    ax_error = plot_velocity_temperature_ax(
        axes[2], model_input[:, :, :], errors, errors_min, errors_max, imsize
    )
    
    tempstring = "Temperature [" + "\u00B0" + "C]"
    errorstring = "Error [" + "\u00B0" + "C]"
    axes[0].set_title("Prediction")
    axes[1].set_title("Target")
    cbar = fig.colorbar(ax_pred[1], ax=axes.ravel().tolist(), shrink=0.22,location="left",label=tempstring, extend = 'both')
    cbar.minorticks_on()
    axes[2].set_title("Error")
    cbar = fig.colorbar(ax_error[1], ax=axes.ravel().tolist(), shrink=0.22,location="right",label=errorstring, extend = 'both')

    axes[0].set_xticks([0,33,65])
    axes[0].set_yticks([0,33,65])
    axes[1].set_xticks([0,33,65])
    axes[1].set_yticks([])
    axes[2].set_xticks([0,33,65])
    axes[2].set_yticks([])
    
    if (abs(errors_min) < 0.5 and abs(errors_max) < 0.5):
        model_dir = "image/good"
        count[0] += 1
    elif (abs(errors_min) < 1.0 and abs(errors_max) < 1.0):
        model_dir = "image/medium"
        count[1] += 1
    else:
        model_dir = "image/bad"
        count[2] += 1
    
    print(f"Good: {count[0]} - Medium: {count[1]} - Bad: {count[2]}")
    plt.savefig(model_dir + "/output_" + str(i) + "_Epoch_" + str(1) + ".png", bbox_inches='tight',pad_inches = 0.2, dpi = 200)
    
    
