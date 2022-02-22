import os
import numpy as np
import matplotlib.pyplot as plt
from data import MultiFolderDataset
from random import randint


def plot_temperature_ax(ax, temp):
    return ax.imshow(temp, origin="lower", vmin=-1.0, vmax=1.0)


def plot_velocity_temperature_ax(ax, vel, temp, imsize: int = 64):
    U = vel[0, :, :].squeeze().detach().cpu()
    V = vel[1, :, :].squeeze().detach().cpu()
    plt_stream = ax.streamplot(
        np.linspace(0, imsize, imsize),
        np.linspace(0, imsize, imsize),
        U,
        V,
        density=0.5,
    )
    plt_temp = plot_temperature_ax(ax, temp.detach().cpu())

    return plt_stream, plt_temp


def plot_velocity_temperature(vel, temp, imsize: int = 64):

    # temp = temp[ :, :].squeeze()

    fig, ax = plt.subplots(figsize=(12, 12))

    _, _ = plot_velocity_temperature_ax(ax, vel, temp, imsize)
    # q = ax.quiver(U, V)

    # plt.savefig("tmp.png")
    # plt.show()
    return fig


def plot_comparison(input, pred, target):
    fig, axes = plt.subplots(1, 3, figsize=(12, 12))
    ax_pred = plot_velocity_temperature_ax(axes[0], input, pred.squeeze())
    ax_target = plot_velocity_temperature_ax(axes[1], input, target.squeeze())
    ax_error = plot_velocity_temperature_ax(axes[2], input, (target - pred).abs().squeeze())
    axes[0].set_title("Pred")
    axes[1].set_title("GT")
    axes[2].set_title("Err")
    cbar = fig.colorbar(ax_pred[1], ax=axes.ravel().tolist(), shrink=0.33)

    return fig


def plot_multi_comparison(input, pred, target):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        ax_pred = plot_velocity_temperature_ax(
            axes[i, 0], input[i, :, :, :], pred[i, :, :].squeeze()
        )
        ax_target = plot_velocity_temperature_ax(
            axes[i, 1], input[i, :, :, :], target[i, :, :].squeeze()
        )
        ax_error = plot_velocity_temperature_ax(
            axes[i, 2],
            input[i, :, :, :],
            (target[i, :, :] - pred[i, :, :]).abs().squeeze(),
        )

    axes[0, 0].set_title("Pred")
    axes[0, 1].set_title("GT")
    axes[0, 2].set_title("Err")
    cbar = fig.colorbar(ax_pred[1], ax=axes.ravel().tolist(), shrink=0.33)

    return fig


if __name__ == "__main__":
    data_path = (
        "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/generated/SingleDirection"
    )
    folder_list = [os.path.join(data_path, f"batch{i+1}") for i in range(2, 3)]
    mf_dataset = MultiFolderDataset(folder_list)
    idx = randint(0, len(mf_dataset) - 1)
    input, target = mf_dataset[idx]

    fig = plot_velocity_temperature(input, target.squeeze())
    fig.savefig("plots/field-output" + str(idx) + ".png", dpi=300)
