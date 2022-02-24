import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_dataset_4_ex_cached, get_dataset_complete_cached
from plot import plot_comparison
from unet import TurbNetG


cm = 1 / 2.54  # centimeters in inches
width_cm = 17
height_cm = 17 * 0.4
fig, ax = plt.subplots(1, 1, figsize=[width_cm, height_cm])

# dataset = get_dataset_complete_cached()
dataset = get_dataset_4_ex_cached()


## load trained model
model_path = "/data/scratch/leiterrl/geoml_turbnet_rc1/"
with open(model_path + "train.yml", "r") as stream:
    config = yaml.safe_load(stream)

model_dict = torch.load(model_path + "model.pt")

# PARAMETERS
channelExponent = config["channelExponent"]
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
for i in range(dataset.dataset_size):
    # print(i)
    model_input = dataset[i][0]
    target = dataset[i][1]

    pred = model(model_input.unsqueeze(0))
    pred = dataset.get_temp_unnormalized(pred)
    target = dataset.get_temp_unnormalized(target)

    # plt.tight_layout()
    # fig = plot_comparison(model_input, pred, target)
    # fig.savefig(f"plots/dataset_4_ex/{i}_comparison.png")

    model_input = dataset.get_velocities_unnormalized(model_input)
    print(f"vel[{i}] = [{model_input[0,31,31]}, {model_input[1,31,31]}]")

    error = target - pred
    max_err = max(error.abs().max(), max_err)

print(max_err)
