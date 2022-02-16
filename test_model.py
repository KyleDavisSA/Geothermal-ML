from unet import TurbNetG
from data import get_single_example
import torch
import time
import numpy as np

from collections import OrderedDict

base_dir = "/data/scratch/leiterrl/geoml"
cache_dir = "/data/scratch/leiterrl/"
rank = "cuda:0"

model_path = cache_dir + "geoml_turbnet_rot_lr_phys20220215-092700"

model_dict = torch.load(model_path + "/model.pt")
new_state_dict = OrderedDict()
for k, v in model_dict["model_state_dict"].items():
    name = k[7:]  # remove 'module.' of dataparallel
    new_state_dict[name] = v

example_in, example_out = get_single_example()
example_in = example_in.unsqueeze(0)
# example_in = example_in.to(rank)
# example_out.to(rank)


# PARAMETERS
channelExponent = 5

model = TurbNetG(channelExponent=channelExponent)
# model.to(rank)

# print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

model.load_state_dict(new_state_dict)
model.eval()

start = time.process_time()
pred = model.forward(example_in)
# torch.cuda.synchronize()
print(time.process_time() - start)


# model.apply(weights_init)
