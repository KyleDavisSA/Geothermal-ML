import os
from datetime import datetime

import numpy as np
import torch
import yaml
from torch import optim
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data import SVDDataset
from models import GWHPSVDEncodeDecode, GWHPSVDEncodeDecodeLinear, GWHPSVDModel
from plot import plot_comparison, plot_multi_input_comparison
from unet import weights_init
from utils import cleanup, run_parallel, setup

import hydra
from omegaconf import DictConfig, OmegaConf

# model_dir = "runs/run"
# scratch_dir = "/scratch/sc/"

# scratch_dir = "/data/scratch/"
# base_dir = scratch_dir + "leiterrl/geoml"
# cache_dir = scratch_dir + "leiterrl/"
# use_cache = True
# distributed_training = False

# with open("train_multi.yml", "r") as stream:
#     config = yaml.safe_load(stream)

# PARAMETERS
# n_epochs = config["n_epochs"]
# lr = config["lr"]
# batch_size = config["batch_size"]
# write_freq = config["write_freq"]
# imsize = config["imsize"]
# num_modes = config["num_modes"]


def run_training(rank, world_size, cfg):
    is_distributed = cfg.distributed_training

    base_dir = os.path.join(cfg.io.scratch_dir, "leiterrl/geoml")
    # cache_dir: scratch_dir + "leiterrl/"

    cache_file = "/data/scratch/leiterrl/data_all_all_direction.pt"
    data_tensor = torch.load(cache_file)
    dataset = SVDDataset(
        data_tensor,
        cfg.params.imsize,
        normalize=cfg.params.normalize,
        augment=cfg.params.augment,
        remove_rotation=cfg.params.remove_rotation,
    )

    train_size = int(cfg.params.train_split_percentage * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    U, S, V = dataset.get_svd_data()

    if is_distributed:
        print(f"Running DDP example on rank {rank}.")
        setup(rank, world_size)

    print(f"Total Length: {len(dataset)}")
    print(f"Test size: {len(test_dataset)}")

    if is_distributed:
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
        batch_size=cfg.params.batch_size,
        shuffle=not is_distributed,
        pin_memory=True,
        sampler=sampler_train,
        # drop_last=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=not is_distributed,
        pin_memory=True,
        sampler=sampler_test,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = base_dir + "multi_input" + timestamp
    writer = SummaryWriter(model_dir, flush_secs=10)
    # with open(os.path.join(model_dir, "train_multi.yml"), "w") as yml_file:
    #     yaml.dump(config, yml_file)
    writer.add_text("config", OmegaConf.to_yaml(cfg))

    # model = GWHPSVDModel(U.to(rank), cfg.params.num_modes)
    model = GWHPSVDEncodeDecode(
        U.to(rank), cfg.params.num_modes, cfg.params.n_hidden, cfg.params.n_latent_size
    )
    # model = GWHPSVDEncodeDecodeLinear(U.to(rank), cfg.params.num_modes)
    model.to(rank)

    if is_distributed:
        model = DDP(model, device_ids=[rank])
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = np.sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized custom SVD Model with {} trainable params ".format(params))

    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=cfg.params.lr)
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=0.01, steps_per_epoch=len(train_data_loader), epochs=n_epochs
    # )

    loss_fn = MSELoss()
    # loss_fn = L1Loss()

    postfix_dict = {
        "loss": "",
        "t_loss": "",
        "rel_err": "",
        # "lr": "NaN"
    }

    if rank == 0 or not is_distributed:
        progress_bar = tqdm(
            desc="Epoch: ", total=cfg.params.n_epochs, postfix=postfix_dict, delay=0.5
        )
    else:
        progress_bar = None

    loss = 0
    iteration = 0

    for epoch in range(cfg.params.n_epochs):
        if is_distributed:
            assert sampler_test
            assert sampler_train
            sampler_train.set_epoch(epoch)
            sampler_test.set_epoch(epoch)
        for batch_idx, sample in enumerate(train_data_loader):
            # sample = sample.to(device)
            input = sample[0].to(rank)
            target = sample[1].to(rank)

            input.requires_grad = True

            model.zero_grad()
            optimizer.zero_grad()

            # with autocast("cuda"):
            output = model(input)
            mse_loss = loss_fn(output, target)
            loss = mse_loss

            loss.backward()
            # scaler.scale(loss).backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()
            iteration += 1

        if epoch % cfg.io.write_freq == 0:
            model.eval()

            # fix unbound var warning
            # test_input = None
            # test_target = None
            # test_loss = None
            # test_rel_error = None
            # test_output = None

            test_loss = 0.0
            for test_batch_idx, test_sample in enumerate(test_data_loader):
                test_input = test_sample[0].to(rank)
                test_target = test_sample[1].to(rank)
                test_indices = test_sample[2].to(rank)

                test_output = model(test_input)
                # test_output = model.module.forward_simple(test_input)
                if cfg.params.normalize:
                    test_output = dataset.un_normalize_temp(test_output)
                    test_target = dataset.un_normalize_temp(test_target)
                    test_input = dataset.un_normalize(test_input)

                # TODO: use dataset helper functions
                # denormalization
                # test_output = test_output * mf_dataset.norm_temp[1] + mf_dataset.norm_temp[0]
                # test_target = test_target * mf_dataset.norm_temp[1] + mf_dataset.norm_temp[0]
                test_loss = loss_fn(test_output, test_target)
                test_rel_error = torch.linalg.norm(test_output - test_target) / torch.linalg.norm(
                    test_target
                )

                if rank == 0 or not is_distributed:
                    # rotate back for display
                    if cfg.params.remove_rotation:
                        for idx, test_idx in enumerate(test_indices):
                            test_output[idx, ...] = dataset.un_rotate_temp(
                                test_output[idx, ...], test_idx
                            )
                            test_target[idx, ...] = dataset.un_rotate_temp(
                                test_target[idx, ...], test_idx
                            )
                            test_input[idx, ...] = dataset.un_rotate(test_input[idx, ...], test_idx)

                    postfix_dict["t_loss"] = f"{test_loss:.5f}"
                    postfix_dict["rel_err"] = f"{test_rel_error:.5f}"
                    writer.add_scalar("test_loss", test_loss, epoch)
                    writer.add_scalar("rel_err", test_rel_error, epoch)
                    writer.add_figure(
                        "comp",
                        plot_multi_input_comparison(
                            test_input,
                            test_output,
                            test_target,
                            cfg.params.imsize,
                        ),
                        epoch,
                    )

            model.train()

            if rank == 0 or not is_distributed:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict()
                        if is_distributed and isinstance(model, DDP)
                        else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "test_loss": test_loss,
                    },
                    model_dir + "/model.pt",
                )

        if (rank == 0 or not is_distributed) and progress_bar:
            writer.add_scalar("loss", loss, epoch)
            # postfix_dict["dir"] = f"{mse_loss:.5f}"
            # postfix_dict["pde"] = f"{res_loss:.5f}"
            postfix_dict["loss"] = f"{loss:.5f}"
            # postfix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.5f}"
            progress_bar.set_postfix(postfix_dict)
            progress_bar.update(1)

    if is_distributed:
        cleanup()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.distributed_training:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_parallel(run_training, world_size, cfg)
    else:
        run_training("cuda:0", 1, cfg)


if __name__ == "__main__":
    hydra_main()
