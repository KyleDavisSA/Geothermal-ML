import os
import numpy as np
import matplotlib.pyplot as plt
from data import MultiFolderDataset
from random import randint
import seaborn as sns
from cycler import cycler
import matplotlib as mpl


# sns.set_theme(style="white", palette="mako")
# sns.color_palette("mako", as_cmap=True)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Times New Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
)

# general
blue = "#6c8199ff"
Blue = "#a2c1e5ff"
red = "#ff6161ff"
Red = "#ffb0b0ff"
green = "#7b996cff"
Green = "#b7e5a2ff"
yellow = "#d6bf71ff"
Yellow = "#ffe286ff"
gray = "#999999ff"
Gray = "#e5e5e5ff"
purple = "#8c6c99ff"
Purple = "#cca2e5ff"

colors = [blue, green, yellow, red, purple, gray, Blue, Green, Yellow, Red, Purple, Gray]

mpl.rcParams["backend"] = "cairo"
mpl.rcParams["hatch.linewidth"] = 1.5  # previous pdf hatch linewidth
mpl.rcParams["image.cmap"] = "RdBu"  # 'seismic'  # 'Blues'  # 'viridis' 'RdBu'
mpl.rcParams["savefig.format"] = "pdf"
# mpl.rcParams["figure.autolayout"] = True  # sets 'tight_layout' automatically
# mpl.rcParams["figure.frameon"] = False
# mpl.rcParams["savefig.bbox"] = "tight"
# mpl.rcParams["savefig.pad_inches"] = 0.05  # do not cut off text
# mpl.rcParams["axes.prop_cycle"] = cycler(color=[blue, green, yellow, red, purple, gray]) + cycler(
#     linestyle=["-", "--", "-.", "--", ":", "-."]
# )
# mpl.rcParams["legend.markerscale"] = 1.0


def plot_temperature_ax(ax, temp, vmin, vmax, cmap=None):
    # return ax.imshow(temp, origin="lower", vmin=-1.0, vmax=1.0)
    # return ax.imshow(temp, origin="lower", vmin=10.0, vmax=15.0)
    return ax.imshow(temp, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)


def plot_velocity_temperature_ax(ax, vel, temp, vmin, vmax, imsize: int = 64, cmap=None):
    U = vel[0, :, :].squeeze().detach().cpu()
    V = vel[1, :, :].squeeze().detach().cpu()
    plt_stream = ax.streamplot(
        np.linspace(0, imsize, imsize),
        np.linspace(0, imsize, imsize),
        U,
        V,
        density=0.5,
    )
    plt_temp = plot_temperature_ax(ax, temp.detach().cpu(), vmin, vmax, cmap=cmap)

    return plt_stream, plt_temp


def plot_velocity_temperature(vel, temp, vmin, vmax, imsize: int = 64, cmap=None):

    # temp = temp[ :, :].squeeze()

    fig, ax = plt.subplots(figsize=(12, 12))

    _, _ = plot_velocity_temperature_ax(ax, vel, temp, vmin, vmax, imsize, cmap=cmap)
    # q = ax.quiver(U, V)

    # plt.savefig("tmp.png")
    # plt.show()
    return fig


def plot_comparison(input, pred, target):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # vmin = 10.0
    # vmax = 14.0
    # vmin = target.min() - target.max()
    value_range = target.max() - target.min()
    # vmin = target.min() - value_range
    vmin = target.min()
    vmax = target.max()
    ax_pred = plot_velocity_temperature_ax(axes[0], input, pred.squeeze(), vmin, vmax, cmap="Reds")
    ax_target = plot_velocity_temperature_ax(
        axes[1], input, target.squeeze(), vmin, vmax, cmap="Reds"
    )
    errors = (target - pred).squeeze()
    errors_range = errors.max() - errors.min()
    errors_min = -errors.abs().max()
    errors_max = errors.abs().max()
    ax_error = plot_velocity_temperature_ax(axes[2], input, errors, errors_min, errors_max)
    axes[0].set_title("Prediction")
    axes[1].set_title("Target")
    axes[2].set_title("Error")
    cbar1 = fig.colorbar(ax_pred[1], ax=axes.ravel().tolist()[0:2], shrink=0.73)
    cbar2 = fig.colorbar(ax_error[1], ax=axes[2], shrink=0.73)

    return fig


def plot_multi_comparison(input, pred, target):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        vmin = target.min()
        vmax = target.max()
        ax_pred = plot_velocity_temperature_ax(
            axes[i, 0], input[i, :, :, :], pred[i, :, :].squeeze(), vmin, vmax
        )
        ax_target = plot_velocity_temperature_ax(
            axes[i, 1], input[i, :, :, :], target[i, :, :].squeeze(), vmin, vmax
        )
        errors = (target[i, :, :] - pred[i, :, :]).squeeze()
        errors_min = -errors.abs().max()
        errors_max = errors.abs().max()
        ax_error = plot_velocity_temperature_ax(
            axes[i, 2], input[i, :, :, :], errors, errors_min, errors_max
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

    fig = plot_velocity_temperature(input, target.squeeze(), 10.0, 15.0)
    fig.savefig("plots/field-output" + str(idx) + ".png", dpi=300)
