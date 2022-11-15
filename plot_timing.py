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
import time

cm = 1 / 2.54  # centimeters in inches
width_cm = 17
height_cm = 17 * 0.4
fig, ax = plt.subplots(1, 1, figsize=[width_cm, height_cm])

os.system("mkdir /home/daviske/Software/git/geothermal-ml/image/good")
os.system("mkdir /home/daviske/Software/git/geothermal-ml/image/medium")
os.system("mkdir /home/daviske/Software/git/geothermal-ml/image/bad")

model_name = "TNG/Channel_5"
model_path = f"/home/daviske/Software/git/geothermal-ml/Results/{model_name}/"
   
print(model_path)
    
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

data_path = "/home/daviske/Research/Darus/geothermal-ml/data"
test_folders = [5,10,15,20,25,30]
folder_list = [os.path.join(str(data_path), f"batch{i+1}") for i in range(total_batch_groups)]
dataset = MultiFolderDataset(folder_list, test_folders, imsize, normalize=True, data_augmentation=False, Inner=False, test=True)

# PARAMETERS
channelExponent = config["channelExponent"]
# imsize = 65 #config["imsize"]


model = TurbNetG(channelExponent=channelExponent)
model.load_state_dict(model_dict["model_state_dict"])

model.eval()

# test_idx = 1
# model_input = dataset[test_idx][0]
# target = dataset[test_idx][1]

# pred = model(model_input.unsqueeze(0))
# pred = dataset.get_temp_unnormalized(pred)
# target = dataset.get_temp_unnormalized(target)


max_err = 0.0
rel_err = 0.0
count = [0,0,0]
errors_reduced = []
errors_reduced_outer = []
errors_reduced_inner = []
for i in range(dataset.dataset_size):
    #for i in range(0,10):
    # print(i)
    model_input = dataset[i][0]
    target = dataset[i][1]

    start = time.time()
    pred = model(model_input.unsqueeze(0))
    end = time.time()
    print(f"Model evaluation time: {end-start}")
    exit()
    
    pred = dataset.get_temp_unnormalized(pred)
    target = dataset.get_temp_unnormalized(target)
    #print("size model_input: ", model_input.size())
    #print("size pred: ", pred.size())
    #print("size target: ", target.size())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 18))
    errors_min_check = []
    errors_max_check = []
    errors = (target[0, :, :] - pred[0, 0, :, :]).squeeze()
    errors_min_check.append(errors.min())
    errors_max_check.append(errors.max())
    
    errors_min = min(errors_min_check)    
    errors_max = max(errors_max_check)  
    
    # Calculate a new error based on the background temps
    for k in range(imsize):
        for j in range(imsize):
            if (abs(errors[k, j]) > 0.2):
                errors_reduced.append(abs(errors[k,j]))
                if (k > 26 and k < 40 and j < 40 and j > 26):
                    errors_reduced_inner.append(abs(errors[k,j]))
                else:
                    errors_reduced_outer.append(abs(errors[k,j]))
    
    vmin = target.min()
    vmax = target.max()
    

model_dir = "/home/daviske/Software/git/geothermal-ml/image"

#fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig, axes = plt.subplots(figsize=(3, 5))
data = []

boxplot_error = np.zeros(len(errors_reduced))
for j in range(len(errors_reduced)):
    boxplot_error[j] = errors_reduced[j]
#fig = plt.figure(figsize =(10, 7))
data.append(boxplot_error)
#axes[0].boxplot(boxplot_error, showfliers=False, notch=True)
#plt.savefig(model_dir + "/box_output_1.png", bbox_inches='tight',pad_inches = 0.2, dpi = 200)

ymax_2 = np.amax(boxplot_error) + 0.1
ymax_1 = 1.5

boxplot_error = np.zeros(len(errors_reduced_inner))
for j in range(len(errors_reduced_inner)):
    boxplot_error[j] = errors_reduced_inner[j]
#fig = plt.figure(figsize =(10, 7))
data.append(boxplot_error)
#axes[1].boxplot(boxplot_error, showfliers=False, notch=True)
#plt.savefig(model_dir + "/box_output_2.png", bbox_inches='tight',pad_inches = 0.2, dpi = 200)

boxplot_error = np.zeros(len(errors_reduced_outer))
for j in range(len(errors_reduced_outer)):
    boxplot_error[j] = errors_reduced_outer[j]
#fig = plt.figure(figsize =(10, 7))
data.append(boxplot_error)
#axes[2].boxplot(boxplot_error, showfliers=False, notch=True)

#norm = colors.Normalize(0.2, ymax_2)
#fig.colorbar(cm.ScalarMappable(norm=norm,cmap='YlGnBu'), axes=axes)


axes.set_ylim(0,ymax_1)
#axes[1].set_ylim(0,ymax)
#axes[2].set_ylim(0,ymax)
#barWidth = 0.18
#axes.xticks([r + barWidth for r in range(len(data))], ['Whole', 'Inner', 'Outer'],rotation = 70,fontsize=16)
#axes.set_title("Whole")
#axes.set_title("Inner")
#axes.set_title("Outer")
axes.set_xticklabels(['All', 'Inner', 'Outer'])
                    
bp = axes.boxplot(data, patch_artist = True, showfliers=False, notch=True,widths=0.5)
for median in bp['medians']:
    median.set_color('black')
#bp = axes.violin(data)

palette_tab20 = sns.color_palette("tab20", 20)
colors = [palette_tab20[0], palette_tab20[2], palette_tab20[4]]
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig(model_dir + "/box_output_3.png", bbox_inches='tight',pad_inches = 0.2, dpi = 200)


