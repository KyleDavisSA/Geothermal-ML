import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_parallel(train_fn, world_size, cfg):
    mp.spawn(
        train_fn,
        args=(
            world_size,
            cfg,
        ),
        nprocs=world_size,
        join=True,
    )


def compute_loss_grads(network: torch.nn.Module, loss: torch.Tensor):
    loss.backward(retain_graph=True)
    grads = []
    for param in network.parameters():
        if param.grad is not None:
            grads.append(torch.flatten(param.grad))
    return torch.cat(grads).clone()


def svd_fields(data, idx):
    return torch.linalg.svd(data[:, idx, :].T)


def get_main_angle(model_input: torch.Tensor):
    """
    return angle by which to rotate flow to point to the right
    param: model_input Tensor with size [2 x x_dir x y_dir]
    """
    assert len(model_input.shape) == 3
    assert model_input.shape[0] == 2

    # print(model_input)
    mean_dir = model_input.mean(dim=[1, 2])
    # print(mean_dir)
    # mean_dir = torch.Tensor([1.0, -1.0])
    mean_dir = mean_dir / mean_dir.norm()
    # mean_dir = torch.nn.functional.normalize(mean_dir)
    target_dir = torch.Tensor([1.0, 0.0])
    target_dir = target_dir / mean_dir.norm()

    # print(mean_dir.cross(target_dir))
    direction = torch.sign(mean_dir[0] * target_dir[1] - mean_dir[1] * target_dir[0])

    # magnitudes = target_dir.norm() * mean_dir.norm()
    angle = torch.rad2deg(torch.acos(mean_dir.dot(target_dir))) * direction

    print(angle)
    # print(direction)
    return angle.item()


def rotate_full_sample(sample: torch.Tensor, angle: float):
    assert len(sample.shape) == 3
    assert sample.shape[1] == 5
