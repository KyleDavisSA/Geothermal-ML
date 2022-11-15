import os

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import CenterCrop
from data import MultiFolderDataset
from unet import TurbNetG, TurbNetG_Light, weights_init, TurbNetG_noSkip_Light
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, L1Loss
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from plot import plot_comparison, plot_multi_comparison
from torchvision.transforms import RandomCrop, Resize, Compose, RandomRotation
import torch
import numpy as np
import random
import yaml
import math
import time

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *

with open("train.yml", "r") as stream:
    config = yaml.safe_load(stream)
    
# PARAMETERS
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
base_path = config["base_path"]
data_path = config["data_path"]

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
mf_dataset = MultiFolderDataset(folder_list, test_folders, imsize, normalize=True, data_augmentation=False, Inner=False, test=False )
print("Test dataset creation ")
mf_dataset_test = MultiFolderDataset(folder_list, test_folders, imsize, normalize=True, data_augmentation=False, Inner=False, test=True)

# Create the training and testing data arrays
train_dataset = mf_dataset
test_dataset = mf_dataset_test
train_size = int(len(mf_dataset))
test_size = int(len(mf_dataset_test))

def run_epoch(rank, world_size):
    if distributed_training:
        print(f"Running DDP example on rank {rank}.")
        setup(rank, world_size)

    rand_rot_trans = RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
    crop_trans = CenterCrop(45)

    print(f"Total Length: {len(mf_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    if distributed_training:
        sampler_train = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

        sampler_test = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    else:
        sampler_train = None
        sampler_test = None

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=distributed_training is None,
        pin_memory=True,
        sampler=sampler_train,
        # drop_last=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=distributed_training is None,
        pin_memory=True,
        sampler=sampler_test,
    )


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = base_dir + "results/" + "TurbNetG_noSkip_Light" + timestamp + "_batch_size_" + str(batch_size) + "_total_batch_groups_" + str(total_batch_groups) + "_channelExponent_" + str(channelExponent)
    writer = SummaryWriter(model_dir, flush_secs=10)
    with open(os.path.join(model_dir, "train.yml"), "w") as yml_file:
        yaml.dump(config, yml_file)

    model = TurbNetG_noSkip_Light(channelExponent=channelExponent)
    model.to(rank)

    if distributed_training:
        model = DDP(model, device_ids=[rank])
    #print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized TurbNetG_noSkip_Light with {} trainable params ".format(params))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.apply(weights_init)

    loss_fn = MSELoss()

    postfix_dict = {
        "loss": "",
        "t_loss": "",
        "rel_err": "",
        # "lr": "NaN",
        "pde": "NaN",
        "dir": "NaN",
        # "neu": "NaN",
    }

    if rank == 0 or not distributed_training:
        progress_bar = tqdm(desc="Epoch: ", total=n_epochs, postfix=postfix_dict, delay=0.5)
    else:
        progress_bar = None

    net = TurbNetG_Light(channelExponent=channelExponent)
    dataiter = iter(train_data_loader)
    images, labels = dataiter.next()
    writer.add_graph(net, images)

    loss = 0
    iteration = 0

    res_loss_weight = config["res_loss_weight"]

    for epoch in range(n_epochs + 1):
        if distributed_training:
            sampler_train.set_epoch(epoch)
            sampler_test.set_epoch(epoch)
        for batch_idx, sample in enumerate(train_data_loader):
            # sample = sample.to(device)
            input = sample[0].to(rank)
            target = sample[1].to(rank)
            if data_augmentation:
                input, target = augment_data(input, target)

            input.requires_grad = True

            # Learning Rate Annealing
            if lra and iteration % 10 == 0 and iteration > 1:
                optimizer.zero_grad()
                output = model(input)
                mse_loss = loss_fn(output, target)

                optimizer.zero_grad()
                mse_loss_grad = compute_loss_grads(model, mse_loss)

                first_loss_max_grad = torch.max(torch.abs(mse_loss_grad))

            model.zero_grad()
            optimizer.zero_grad()

            # with autocast("cuda"):
            output = model(input)
            mse_loss = loss_fn(output, target)
            loss = mse_loss

            loss.backward()
            optimizer.step()
            iteration += 1

        if (epoch % write_freq == 0) or (epoch == (n_epochs - 1)) :
            startplot = time.time()
            model.eval()
            writeNimages = 0
            for test_batch_idx, test_sample in enumerate(test_data_loader):
                test_input = test_sample[0].to(rank)
                test_target = test_sample[1].to(rank)
                if data_augmentation:
                    test_input, test_target = augment_data(test_input, test_target)

                test_output = model(test_input)

                # TODO: use dataset helper functions
                # denormalization
                test_output = test_output * mf_dataset.norm_temp[1] + mf_dataset.norm_temp[0]
                test_target = test_target * mf_dataset.norm_temp[1] + mf_dataset.norm_temp[0]
                test_loss = loss_fn(test_output, test_target)
                test_rel_error = torch.linalg.norm(test_output - test_target) / torch.linalg.norm(
                    test_target
                )
                nPlots = math.floor(len(test_dataset)) - 1
                
                saveImage = False
                if (epoch != 0):
                    
                    if (epoch == (n_epochs - 1)):
                        saveImage = True
                        os.system("mkdir " + model_dir + "/images")
                        
                    for i in range(nPlots):
                        name = "comp-" + str(i)
                        writer.add_figure(
                            name,
                            plot_multi_comparison(test_input, test_output, test_target, imsize, i, epoch, saveImage, model_dir),
                            epoch,
                        )
            endplot = time.time()
            print(f"Plot time at {epoch}: {endplot-startplot}")
                
                
            if rank == 0 or not distributed_training:
                postfix_dict["t_loss"] = f"{test_loss:.5f}"
                writer.add_scalar("test_loss", test_loss, epoch)

            model.train()

            if rank == 0 or not distributed_training:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict()
                        if distributed_training
                        else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "test_loss": test_loss,
                    },
                    model_dir + "/model.pt",
                )

        if (rank == 0 or not distributed_training) and progress_bar:
            writer.add_scalar("loss", loss, epoch)
            postfix_dict["loss"] = f"{loss:.5f}"
            progress_bar.set_postfix(postfix_dict)
            progress_bar.update(1)
            
    writer.close()
    print(f"Saving results to: {model_dir}")
    
    if distributed_training:
        cleanup()


if __name__ == "__main__":
    if distributed_training:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_parallel(run_epoch, world_size)
    else:
        run_epoch("cuda:0", 1)
