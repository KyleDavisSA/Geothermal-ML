import yaml
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from data import (
    get_dataset_4_ex_cached,
    get_dataset_all_dir_cached,
    get_dataset_complete_cached,
    get_dataset_all_dir_test_cached,
    get_mid_perm_test_cached,
)
from plot import plot_comparison, plot_comparison_plume
#from plot_plume import get_contour_data
from unet import TurbNetG


cm = 1 / 2.54  # centimeters in inches
width_cm = 17
height_cm = 17 * 0.4
fig, ax = plt.subplots(1, 1, figsize=[width_cm, height_cm])

# dataset = get_dataset_complete_cached()
# dataset = get_dataset_4_ex_cached()
# dataset = get_dataset_all_dir_cached(data_augmentation=True)
# dataset = get_dataset_all_dir_test_cached(data_augmentation=False)
dataset = get_mid_perm_test_cached(data_augmentation=False)


## load trained model
# model_name = "geoml_turbnet_rc1"
# model_name = "geoml_turbnet_rc2_all_dir"
model_name = "geomlmid_perm_test20220309-154655"
model_path = f"/data/scratch/leiterrl/{model_name}/"
with open(model_path + "train.yml", "r") as stream:
    config = yaml.safe_load(stream)

model_dict = torch.load(model_path + "model.pt")

# PARAMETERS
channelExponent = config["channelExponent"]
imsize = config["imsize"]
# imsize = 64

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
for i in range(dataset.dataset_size):
    # print(i)
    model_input = dataset[i][0]
    target = dataset[i][1]

    pred = model(model_input.unsqueeze(0))
    pred = dataset.get_temp_unnormalized(pred)
    target = dataset.get_temp_unnormalized(target)

    # model_input = dataset.get_velocities_unnormalized(model_input)
    # vel = model_input[:, 31, 31]
    # print(f"vel[{i}] = [{vel[0]}, {vel[1]}]")
    # vel_mag = vel.norm()
    # vel_dir = vel / vel_mag
    # print(np.rad2deg(math.atan2(vel_dir[1], vel_dir[0])))
    # direction = math.atan2(vel_dir[1], vel_dir[0])
    # direction = np.deg2rad(45.0)

    # x_grid, y_grid, plume_data, levels = get_contour_data(vel, direction)
    # levels = np.arange(10, 15.0, 1.0)

    # plt.tight_layout()
    # fig = plot_comparison_plume(
    #     model_input, pred, target, x_grid - 0.5, y_grid - 0.5, plume_data, levels, imsize
    # )
    # # fig = plot_comparison_plume(model_input, pred, target)
    # fig.savefig(f"plots/dataset_all_dir_test/{i}_comparison_test4.pdf")
    # plt.close()

    error = target - pred
    max_err = max(error.abs().max(), max_err)
    # rel_err = (error).abs().sum() / target.abs().sum()
    rel_err = torch.linalg.norm(error) / torch.linalg.norm(target)

print(max_err)
print(rel_err)
