"""
Load args and model from a directory
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
import h5py
import json
import meshio
import numpy as np

dataset_path = (
    "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/Centered/64x64/"
)


def load_args(run_dir):
    with open(run_dir + "/args.txt") as args_file:
        args = Namespace(**json.load(args_file))
    # pprint(args)
    return args


def load_data(hdf5_file, ndata, batch_size, only_input=True, return_stats=False):
    with h5py.File(hdf5_file, "r") as f:
        x_data = f["input"][:ndata]
        print(f"x_data: {x_data.shape}")
        # print(f'x_data: {x_data}')
        if not only_input:
            y_data = f["output"][:ndata]
            print(f"y_data: {y_data.shape}")
            # print(f'y_data: {y_data}')

    stats = {}
    if return_stats:
        y_variation = ((y_data - y_data.mean(0, keepdims=True)) ** 2).sum(
            axis=(0, 2, 3)
        )
        stats["y_variation"] = y_variation

    data_tuple = (
        (torch.FloatTensor(x_data),)
        if only_input
        else (torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    )
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )
    print(f"Loaded dataset: {hdf5_file}")
    return data_loader, stats


def load_data_vtk_train(
    hdf5_file,
    imsize,
    input_channels,
    ndata,
    batch_size,
    only_input=True,
    return_stats=False,
):
    """
    with h5py.File(hdf5_file, 'r') as f:
        x_data = f['input'][:ndata]
        print(f'x_data: {x_data.shape}')
        #print(f'x_data: {x_data}')
        if not only_input:
            y_data = f['output'][:ndata]
            print(f'y_data: {y_data.shape}')
            #print(f'y_data: {y_data}')
    """
    if input_channels == 2:
        x_data = np.zeros((ndata, 2, imsize, imsize))
    if input_channels == 3:
        x_data = np.zeros((ndata, 3, imsize, imsize))
    y_data = np.zeros((ndata, 1, imsize, imsize))
    for i in range(0, ndata):
        mesh = meshio.read(dataset_path + "training/pflotran-vel-" + str(i) + ".vtk")
        data = meshio.read(dataset_path + "training/pflotran-" + str(i) + ".vtk")
        for k in range(0, imsize):
            for j in range(0, imsize):
                x_data[i, 0, k, j] = mesh.cell_data["Vlx"][0][j + k * imsize]
                x_data[i, 1, k, j] = mesh.cell_data["Vly"][0][j + k * imsize]
                if input_channels == 3:
                    x_data[i, 2, k, j] = (
                        data.cell_data["Temperature"][0][j + k * imsize] - 10
                    )
                y_data[i, 0, k, j] = (
                    data.cell_data["Temperature"][0][j + k * imsize] - 10
                )
        # x_data[i,2,32,20] = 5

    # remove_list = [2, 3]
    # x_data = np.delete(x_data, remove_list, axis=0)
    # y_data = np.delete(y_data, remove_list, axis=0)

    stats = {}
    if return_stats:
        y_variation = ((y_data - y_data.mean(0, keepdims=True)) ** 2).sum(
            axis=(0, 2, 3)
        )
        stats["y_variation"] = y_variation

    data_tuple = (torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=False
    )
    print(f"Loaded dataset: {hdf5_file}")

    return data_loader, stats, x_data, y_data


def load_data_vtk_test(
    hdf5_file,
    imsize,
    input_channels,
    ndata,
    batch_size,
    only_input=True,
    return_stats=False,
):
    """
    with h5py.File(hdf5_file, 'r') as f:
        x_data = f['input'][:ndata]
        print(f'x_data: {x_data.shape}')
        #print(f'x_data: {x_data}')
        if not only_input:
            y_data = f['output'][:ndata]
            print(f'y_data: {y_data.shape}')
            #print(f'y_data: {y_data}')
    """
    if input_channels == 2:
        x_data = np.zeros((ndata, 2, imsize, imsize))
    if input_channels == 3:
        x_data = np.zeros((ndata, 3, imsize, imsize))
    y_data = np.zeros((ndata, 1, imsize, imsize))
    for i in range(0, ndata):
        mesh = meshio.read(dataset_path + "testing/pflotran-vel-" + str(i) + ".vtk")
        data = meshio.read(dataset_path + "testing/pflotran-" + str(i) + ".vtk")
        for k in range(0, imsize):
            for j in range(0, imsize):
                x_data[i, 0, k, j] = mesh.cell_data["Vlx"][0][j + k * imsize]
                x_data[i, 1, k, j] = mesh.cell_data["Vly"][0][j + k * imsize]
                if input_channels == 3:
                    x_data[i, 2, k, j] = (
                        data.cell_data["Temperature"][0][j + k * imsize] - 10
                    )
                y_data[i, 0, k, j] = (
                    data.cell_data["Temperature"][0][j + k * imsize] - 10
                )
        # x_data[i,2,32,20] = 5

    stats = {}
    if return_stats:
        y_variation = ((y_data - y_data.mean(0, keepdims=True)) ** 2).sum(
            axis=(0, 2, 3)
        )
        stats["y_variation"] = y_variation

    data_tuple = (
        (torch.FloatTensor(x_data),)
        if only_input
        else (torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    )
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=False
    )
    print(f"Loaded dataset: {hdf5_file}")
    return data_loader, stats, x_data, y_data
