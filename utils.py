
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_parallel(train_fn, world_size):
    mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)

def compute_loss_grads(network: torch.nn.Module, loss: torch.Tensor):
    loss.backward(retain_graph=True)
    grads = []
    for param in network.parameters():
        if param.grad is not None:
            grads.append(torch.flatten(param.grad))
    return torch.cat(grads).clone()