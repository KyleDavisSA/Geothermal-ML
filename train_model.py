import os

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import CenterCrop
from data import (
    CacheDataset,
    MultiFolderDataset,
    get_dataset_all_dir_cached,
    get_dataset_complete_cached,
    get_mid_perm_training_cached,
)
from physics import SobelFilter, constitutive_constraint
from unet import TurbNetG, UNet, weights_init
from models import DenseED
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

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *

# model_dir = "runs/run"
# scratch_dir = "/scratch/sc/"
scratch_dir = "/data/scratch/"
base_dir = scratch_dir + "leiterrl/geoml"
cache_dir = scratch_dir + "leiterrl/"
use_cache = True
distributed_training = False

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


def augment_data(input, target):
    seed = np.random.randint(2147483647)  # make a seed with numpy generator
    random.seed(seed)  # apply this seed to img tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    # input = trans(input)

    random.seed(seed)  # apply this seed to target tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    # target = trans(target)

    return input, target


# data_path = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/generated/SingleDirection"
# data_path = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/noFlow_withFlow"
# data_path_cache = "/data/scratch/leiterrl/data_complete.pt"


# if not use_cache:
#     folder_list = [os.path.join(data_path, f"batch{i+1}") for i in range(2)]
#     mf_dataset = MultiFolderDataset(folder_list, data_augmentation=True)
#     torch.save(mf_dataset, cache_dir + "cache.pt")
# else:
#     mf_dataset = torch.load(cache_dir + "cache.pt")

# mf_dataset = get_dataset_complete_cached(data_augmentation=True)
# mf_dataset = get_dataset_all_dir_cached(data_augmentation=True)
mf_dataset = get_mid_perm_training_cached()


# mf_dataset.dataset_tensor = mf_dataset.dataset_tensor.to("cuda")

train_size = int(0.8 * len(mf_dataset))
test_size = len(mf_dataset) - train_size
train_dataset, test_dataset = random_split(mf_dataset, [train_size, test_size])


def run_epoch(rank, world_size):
    if distributed_training:
        print(f"Running DDP example on rank {rank}.")
        setup(rank, world_size)

    rand_rot_trans = RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
    crop_trans = CenterCrop(45)
    # resize_trans = Resize((imsize, imsize))
    # trans = Compose([rand_rot_trans, crop_trans, resize_trans])

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

    # def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     pin_memory=pin_memory,
    #     num_workers=num_workers,
    #     drop_last=False,
    #     shuffle=False,
    #     sampler=sampler,
    # )

    # return dataloader

    # device = "cuda:0"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = base_dir + "mid_perm_test" + timestamp
    writer = SummaryWriter(model_dir, flush_secs=10)
    with open(os.path.join(model_dir, "train.yml"), "w") as yml_file:
        yaml.dump(config, yml_file)

    # model = DenseED(
    #     in_channels=2,
    #     out_channels=1,
    #     imsize=64,
    #     blocks=[3, 4, 3],
    #     growth_rate=24,
    #     init_features=32,
    #     drop_rate=0.2,
    #     out_activation=None,
    #     upsample="nearest",
    # )

    # model = UNet(in_channels=2, out_channels=1)

    model = TurbNetG(channelExponent=channelExponent)
    model.to(rank)

    if distributed_training:
        model = DDP(model, device_ids=[rank])
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized TurbNet with {} trainable params ".format(params))

    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=0.01, steps_per_epoch=len(train_data_loader), epochs=n_epochs
    # )

    loss_fn = MSELoss()
    # loss_fn = L1Loss()
    sobel_filter = SobelFilter(imsize, correct=True, device=rank)

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

    # writer.add_graph(model)

    # scaler = GradScaler()

    # CUDA Graph warmup
    # static_sample = next(iter(train_data_loader))
    # static_input = static_sample[0].to(rank)
    # static_target = static_sample[1].to(rank)

    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for i in range(20):
    #         optimizer.zero_grad(set_to_none=True)
    #         output = model(static_input)
    #         loss = loss_fn(output, static_target)
    #         loss.backward()
    #         optimizer.step()
    # torch.cuda.current_stream().wait_stream(s)

    # # CUDA Graph capture
    # g = torch.cuda.CUDAGraph()
    # # Sets grads to None before capture, so backward() will create
    # # .grad attributes with allocations from the graph's private pool
    # optimizer.zero_grad(set_to_none=True)
    # with torch.cuda.graph(g):
    #     static_y_pred = model(static_input)
    #     loss = loss_fn(static_y_pred, static_target)
    #     loss.backward()
    #     optimizer.step()

    loss = 0
    iteration = 0

    res_loss_weight = config["res_loss_weight"]

    for epoch in range(n_epochs):
        if distributed_training:
            sampler_train.set_epoch(epoch)
            sampler_test.set_epoch(epoch)
        for batch_idx, sample in enumerate(train_data_loader):
            # sample = sample.to(device)
            input = sample[0].to(rank)
            target = sample[1].to(rank)
            if data_augmentation:
                input, target = augment_data(input, target)

            # static_input.copy_(input)
            # static_target.copy_(target)
            # g.replay()

            input.requires_grad = True

            # Learning Rate Annealing
            if lra and iteration % 10 == 0 and iteration > 1:
                optimizer.zero_grad()
                output = model(input)
                mse_loss = loss_fn(output, target)
                res_loss = constitutive_constraint(input, output, sobel_filter)

                optimizer.zero_grad()
                mse_loss_grad = compute_loss_grads(model, mse_loss)
                optimizer.zero_grad()
                res_loss_grad = compute_loss_grads(model, res_loss)

                # TODO: this is bad to assume i guess...
                first_loss_max_grad = torch.max(torch.abs(mse_loss_grad))
                update = first_loss_max_grad / torch.mean(torch.abs(res_loss_grad))
                res_loss_weight = (1.0 - lra_alpha) * res_loss_weight + lra_alpha * update
                writer.add_scalar("res_weight", res_loss_weight, epoch)

            model.zero_grad()
            optimizer.zero_grad()

            # with autocast("cuda"):
            output = model(input)
            mse_loss = loss_fn(output, target)
            loss = mse_loss
            if physical_loss:
                res_loss = constitutive_constraint(input, output, sobel_filter, mf_dataset)
                loss = mse_loss + res_loss_weight * res_loss

            loss.backward()
            # scaler.scale(loss).backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()
            iteration += 1

        if epoch % write_freq == 0:
            model.eval()
            for test_batch_idx, test_sample in enumerate(test_data_loader):
                test_input = test_sample[0].to(rank)
                test_target = test_sample[1].to(rank)
                if data_augmentation:
                    test_input, test_target = augment_data(test_input, test_target)

                test_output = model(test_input)
                # test_output = model.module.forward_simple(test_input)

                # TODO: use dataset helper functions
                # denormalization
                test_output = test_output * mf_dataset.norm_temp[1] + mf_dataset.norm_temp[0]
                test_target = test_target * mf_dataset.norm_temp[1] + mf_dataset.norm_temp[0]
                test_loss = loss_fn(test_output, test_target)
                test_rel_error = torch.linalg.norm(test_output - test_target) / torch.linalg.norm(
                    test_target
                )
            if rank == 0 or not distributed_training:
                postfix_dict["t_loss"] = f"{test_loss:.5f}"
                postfix_dict["rel_err"] = f"{test_rel_error:.5f}"
                writer.add_scalar("test_loss", test_loss, epoch)
                writer.add_scalar("rel_err", test_rel_error, epoch)
                writer.add_figure(
                    "comp",
                    plot_multi_comparison(test_input, test_output, test_target, imsize),
                    epoch,
                )

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
            # postfix_dict["dir"] = f"{mse_loss:.5f}"
            # postfix_dict["pde"] = f"{res_loss:.5f}"
            postfix_dict["loss"] = f"{loss:.5f}"
            # postfix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.5f}"
            progress_bar.set_postfix(postfix_dict)
            progress_bar.update(1)

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
