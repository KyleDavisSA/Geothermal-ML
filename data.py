import os
import meshio
import torch
import math
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import RandomCrop, Resize, Compose, RandomRotation
from torchvision.transforms.functional import InterpolationMode, rotate
from torchvision.transforms.transforms import CenterCrop


temp_offset = 10

def get_eligible_vtk_files(folder: str) -> "list[str]":
    eligible_files: "list[str]" = []
    for data_file in sorted(os.listdir(folder)):
        if "pflotran-noFlow-new-vel-" in data_file:
            eligible_files.append(os.path.join(folder, data_file))

    return eligible_files


def rotate_vector_field(vector_field: torch.Tensor, angle: float):

    u = vector_field[0, :, :]
    v = vector_field[1, :, :]

    new_u = torch.empty_like(u)
    new_v = torch.empty_like(v)
    new_u = math.cos(angle) * u + math.sin(angle) * v
    new_v = -math.sin(angle) * u + math.cos(angle) * v

    new_u.unsqueeze_(0)
    new_v.unsqueeze_(0)

    return torch.cat([new_u, new_v], 0)



        
class CacheDataset(Dataset):
    def __init__(
        self,
        cache_file: str,
        imsize: int,
        normalize: bool = True,
        data_augmentation: bool = False,
    ) -> None:
        data_augmentation_samples = 720

        self.normalize = normalize
        self.imsize: int = imsize
        self.dataset_size: int = 0
        
        self.dataset_tensor = torch.load(cache_file)
        #self.dataset_size = self.dataset_tensor.shape[0]
#
        if data_augmentation:
            self.dataset_size += data_augmentation_samples
            self.dataset_tensor = torch.cat(
                [
                    self.dataset_tensor,
                    torch.empty([data_augmentation_samples, 3, self.imsize, self.imsize]),
                ],
                0,
            )

        self.norm_v_x = [0.0, 1.0]
        self.norm_v_y = [0.0, 1.0]
        self.norm_temp = [temp_offset, 1.0]

        # normalize data by max over whole dataset
        if self.normalize:
            # remove temperature offset
            self.dataset_tensor[:, 2, :, :] -= temp_offset
            # calculate scaling factors
            v_x_max_inv = 1.0 / self.dataset_tensor[:, 0, :, :].abs().max()
            v_y_max_inv = 1.0 / self.dataset_tensor[:, 1, :, :].abs().max()
            temp_max_inv = 1.0 / self.dataset_tensor[:, 2, :, :].abs().max()
            self.dataset_tensor[:, 0, :, :] = self.dataset_tensor[:, 0, :, :] * v_x_max_inv
            self.dataset_tensor[:, 1, :, :] = self.dataset_tensor[:, 1, :, :] * v_y_max_inv
            self.dataset_tensor[:, 2, :, :] = self.dataset_tensor[:, 2, :, :] * temp_max_inv

            self.norm_v_x[1] = 1.0 / v_x_max_inv
            self.norm_v_y[1] = 1.0 / v_y_max_inv
            self.norm_temp[1] = 1.0 / temp_max_inv

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (
            self.dataset_tensor[index, 0:2, :, :],
            self.dataset_tensor[index, 2, :, :].unsqueeze(0),
        )

    def get_temp_unnormalized(self, temp):
        return temp * self.norm_temp[1] + self.norm_temp[0]

    def get_velocities_unnormalized(self, vel):
        vel[0, :, :] = vel[0, :, :] * self.norm_v_x[1]
        vel[1, :, :] = vel[1, :, :] * self.norm_v_y[1]

        return vel


class MultiFolderDataset(Dataset):
    def __init__(
        self,
        folder_list: "list[str]",
        test_folders: [int],
        imsize: int,
        normalize: bool = True,
        data_augmentation: bool = False,
        Inner: bool = False,
        test: bool = True,
    ) -> None:
    
        data_augmentation_samples = 400
        Inner = False

        self.normalize = normalize
        self.imsize: int = imsize
        self.dataset_size: int = 0
        eligible_files: "list[str]" = []
        if (test):
            i = 1
            for folder in folder_list:
                if i in test_folders:
                    print("Folder name testing data selected: ", folder )
                    eligible_files += get_eligible_vtk_files(folder)
                i += 1
        else:
            i = 1
            for folder in folder_list:
                if i not in test_folders:
                    print("Folder name training data selected: ", folder )
                    eligible_files += get_eligible_vtk_files(folder)
                i += 1


        self.dataset_size = len(eligible_files)

        if data_augmentation:
            self.dataset_size += data_augmentation_samples

        self.dataset_tensor = torch.empty([self.dataset_size, 3, self.imsize, self.imsize])
        for idx, file_path in enumerate(eligible_files):
            if (Inner):
                self.dataset_tensor[idx, :, :, :] = load_vtk_file_Inner(file_path, self.imsize)
            else:
                self.dataset_tensor[idx, :, :, :] = load_vtk_file(file_path, self.imsize)

        self.norm_v_x = [0.0, 1.0]
        self.norm_v_y = [0.0, 1.0]
        self.norm_temp = [temp_offset, 1.0]
        
        # TODO: unify with CacheDataset
        # normalize data by max over whole dataset
        if self.normalize:
            v_x_max_inv = 1.0 / self.dataset_tensor[:, 0, :, :].abs().max()
            v_y_max_inv = 1.0 / self.dataset_tensor[:, 1, :, :].abs().max()
            temp_max_inv = 1.0 / self.dataset_tensor[:, 2, :, :].abs().max()
            self.dataset_tensor[:, 0, :, :] = self.dataset_tensor[:, 0, :, :] * v_x_max_inv
            self.dataset_tensor[:, 1, :, :] = self.dataset_tensor[:, 1, :, :] * v_y_max_inv
            self.dataset_tensor[:, 2, :, :] -= temp_offset
            self.dataset_tensor[:, 2, :, :] = self.dataset_tensor[:, 2, :, :] * temp_max_inv
            
            self.norm_v_x[1] = 1.0 / v_x_max_inv
            self.norm_v_y[1] = 1.0 / v_y_max_inv
            self.norm_temp[1] = 1.0 / temp_max_inv


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (
            self.dataset_tensor[index, 0:2, :, :],
            self.dataset_tensor[index, 2, :, :].unsqueeze(0),
        )
    
    def get_temp_unnormalized(self, temp):
        return temp * self.norm_temp[1] + self.norm_temp[0]

    def get_velocities_unnormalized(self, vel):
        vel[0, :, :] = vel[0, :, :] * self.norm_v_x[1]
        vel[1, :, :] = vel[1, :, :] * self.norm_v_y[1]

        return vel


def load_vtk_file(file_path_vel: str, imsize: int):
    """loads mesh and tmeperature data from vtk file
    expects file path including "vel" part
    """
    # file_path = file_path_vel.replace("pflotran-new-vel-", "pflotran-new-")
    file_path = file_path_vel.replace("pflotran-noFlow-new-vel", "pflotran-withFlow-new")
    mesh = meshio.read(file_path_vel)
    data = meshio.read(file_path)

    ret_data = torch.empty([3, imsize, imsize], dtype=torch.float32)
    v_x = mesh.cell_data["Vlx"][0].reshape([imsize, imsize])
    v_y = mesh.cell_data["Vly"][0].reshape([imsize, imsize])
    temp = (data.cell_data["Temperature"][0]).reshape([imsize, imsize])

    ret_data[0, :, :] = torch.as_tensor(v_x, dtype=torch.float32)
    ret_data[1, :, :] = torch.as_tensor(v_y, dtype=torch.float32)
    ret_data[2, :, :] = torch.as_tensor(temp, dtype=torch.float32)

    return ret_data
    
def load_vtk_file_Inner(file_path_vel: str, imsize: int):
    """loads mesh and tmeperature data from vtk file
    expects file path including "vel" part
    """
    # file_path = file_path_vel.replace("pflotran-new-vel-", "pflotran-new-")
    file_path = file_path_vel.replace("pflotran-noFlow-new-vel", "pflotran-withFlow-new")
    mesh = meshio.read(file_path_vel)
    data = meshio.read(file_path)

    ret_data = torch.empty([3, imsize, imsize], dtype=torch.float32)
    out_data = torch.empty([3, 15, 15], dtype=torch.float32)
    v_x = mesh.cell_data["Vlx"][0].reshape([imsize, imsize])
    v_y = mesh.cell_data["Vly"][0].reshape([imsize, imsize])
    temp = (data.cell_data["Temperature"][0]).reshape([imsize, imsize])

    ret_data[0, :, :] = torch.as_tensor(v_x, dtype=torch.float32)
    ret_data[1, :, :] = torch.as_tensor(v_y, dtype=torch.float32)
    ret_data[2, :, :] = torch.as_tensor(temp, dtype=torch.float32)
    out_data[0, :, :] = ret_data[0, 26:41, 26:41]
    out_data[1, :, :] = ret_data[1, 26:41, 26:41]
    out_data[2, :, :] = ret_data[2, 26:41, 26:41]

    return out_data
