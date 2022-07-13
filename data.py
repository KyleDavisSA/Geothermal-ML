import os
import meshio
import torch
import math
import random
import numpy as np
from torch.utils.data import Dataset

from torchvision.transforms import RandomCrop, Resize, Compose, RandomRotation
from torchvision.transforms.functional import InterpolationMode, rotate
from torchvision.transforms.transforms import CenterCrop
from utils import svd_fields


temp_offset = 12.5


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

        self.dataset_size = self.dataset_tensor.shape[0]
        print(self.dataset_size)

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

        if data_augmentation:
            # rand_rot_trans = RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
            crop_trans = CenterCrop(45)
            resize_trans = Resize((self.imsize, self.imsize))
            trans = Compose([crop_trans, resize_trans])

            for i in range(data_augmentation_samples):
                # get dataset sample
                idx = random.randint(0, self.dataset_size - data_augmentation_samples)
                angle = random.uniform(0.0, 360.0)

                input = self.dataset_tensor[idx, 0:2, :, :].detach().clone()
                target = self.dataset_tensor[idx, 2, :, :].unsqueeze(0).detach().clone()

                # seed = np.random.randint(2147483647)  # make a seed with numpy generator
                # random.seed(seed)  # apply this seed to img tranfsorms
                # torch.manual_seed(seed)  # needed for torchvision 0.7
                # input = trans(input)

                # random.seed(seed)  # apply this seed to target tranfsorms
                # torch.manual_seed(seed)  # needed for torchvision 0.7
                # sample_aug = trans(sample)
                input = rotate_vector_field(input, np.deg2rad(angle))
                target = trans(rotate(target, angle))

                new_idx = self.dataset_size - data_augmentation_samples + i
                self.dataset_tensor[new_idx, 0:2, :, :] = input
                self.dataset_tensor[new_idx, 2, :, :] = target
                # self.dataset_tensor[new_idx, :, :, :] = sample_aug

        # normalize data by max over whole dataset
        if self.normalize:
            # remove temperature offset
            self.dataset_tensor[:, 2, :, :] -= temp_offset
            # calculate scaling factors
            v_x_max_inv = 1.0 / self.dataset_tensor[:, 0, :, :].abs().max().item()
            v_y_max_inv = 1.0 / self.dataset_tensor[:, 1, :, :].abs().max().item()
            temp_max_inv = 1.0 / self.dataset_tensor[:, 2, :, :].abs().max().item()
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


# class MultiFolderDataset(Dataset):
#     def __init__(
#         self,
#         folder_list: "list[str]",
#         imsize: int,
#         normalize: bool = True,
#         data_augmentation: bool = False,
#     ) -> None:
#         data_augmentation_samples = 2000

#         self.normalize = normalize
#         self.imsize: int = imsize
#         self.dataset_size: int = 0
#         eligible_files: "list[str]" = []
#         for folder in folder_list:
#             eligible_files += get_eligible_vtk_files(folder)

#         self.dataset_size = len(eligible_files)
#         print(self.dataset_size)

#         if data_augmentation:
#             self.dataset_size += data_augmentation_samples

#         self.dataset_tensor = torch.empty([self.dataset_size, 3, self.imsize, self.imsize])
#         for idx, file_path in enumerate(eligible_files):
#             print(idx)
#             self.dataset_tensor[idx, :, :, :] = load_vtk_file(file_path, self.imsize)

#         # TODO: unify with CacheDataset
#         # normalize data by max over whole dataset
#         if self.normalize:
#             v_x_max_inv = 1.0 / self.dataset_tensor[:, 0, :, :].abs().max()
#             v_y_max_inv = 1.0 / self.dataset_tensor[:, 1, :, :].abs().max()
#             temp_max_inv = 1.0 / self.dataset_tensor[:, 2, :, :].abs().max()
#             self.dataset_tensor[:, 0, :, :] = self.dataset_tensor[:, 0, :, :] * v_x_max_inv
#             self.dataset_tensor[:, 1, :, :] = self.dataset_tensor[:, 1, :, :] * v_y_max_inv
#             self.dataset_tensor[:, 2, :, :] -= temp_offset
#             self.dataset_tensor[:, 2, :, :] = self.dataset_tensor[:, 2, :, :] * temp_max_inv

#         if data_augmentation:
#             rand_rot_trans = RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
#             crop_trans = CenterCrop(45)
#             resize_trans = Resize((self.imsize, self.imsize))
#             trans = Compose([rand_rot_trans, crop_trans, resize_trans])

#             for i in range(data_augmentation_samples):
#                 # get dataset sample
#                 idx = random.randint(0, self.dataset_size - data_augmentation_samples)

#                 input = self.dataset_tensor[idx, 0:2, :, :].detach().clone()
#                 target = self.dataset_tensor[idx, 2, :, :].unsqueeze(0).detach().clone()

#                 seed = np.random.randint(2147483647)  # make a seed with numpy generator
#                 random.seed(seed)  # apply this seed to img tranfsorms
#                 torch.manual_seed(seed)  # needed for torchvision 0.7
#                 input = trans(input)

#                 random.seed(seed)  # apply this seed to target tranfsorms
#                 torch.manual_seed(seed)  # needed for torchvision 0.7
#                 target = trans(target)

#                 new_idx = self.dataset_size - data_augmentation_samples + i
#                 self.dataset_tensor[new_idx, 0:2, :, :] = input
#                 self.dataset_tensor[new_idx, 2, :, :] = target

#     def __len__(self):
#         return self.dataset_size

#     def __getitem__(self, index):
#         return (
#             self.dataset_tensor[index, 0:2, :, :],
#             self.dataset_tensor[index, 2, :, :].unsqueeze(0),
#         )


class SVDDataset(Dataset):
    def __init__(
        self, data_tensor: torch.Tensor, imsize: int, normalize: bool = True, augment: bool = True
    ):
        self.imsize = imsize
        self.data = data_tensor
        # center temp data
        # self.data[:, 2, :] = self.data[:, 2, :] - 12.5

        data_augmentation_samples = 720

        if augment:
            # extend dataset by num augmentation samples
            self.data = torch.cat(
                [
                    self.data,
                    torch.empty([data_augmentation_samples, 5, self.imsize**2]),
                ],
                0,
            )

            extended_size = self.data.shape[0]

            # rand_rot_trans = RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
            crop_trans = CenterCrop(45)
            resize_trans = Resize((self.imsize, self.imsize))
            trans = Compose([crop_trans, resize_trans])

            for i in range(data_augmentation_samples):
                # get dataset sample
                idx = random.randint(0, extended_size - data_augmentation_samples)
                angle = random.uniform(0.0, 360.0)

                # input = self.data[idx, 0:2, :, :].detach().clone()
                # target = self.data[idx, 2, :, :].unsqueeze(0).detach().clone()
                sample = self.data[idx, :, :].detach().clone()
                sample[0:2, :] = rotate_vector_field(
                    sample[0:2, :].reshape((-1, self.imsize, self.imsize)), np.deg2rad(angle)
                ).reshape(-1, self.imsize**2)
                sample[2:, :] = trans(
                    rotate(sample[2:, :].reshape((-1, self.imsize, self.imsize)), angle)
                ).reshape(-1, self.imsize**2)

                # seed = np.random.randint(2147483647)  # make a seed with numpy generator
                # random.seed(seed)  # apply this seed to img tranfsorms
                # torch.manual_seed(seed)  # needed for torchvision 0.7
                # input = trans(input)

                # random.seed(seed)  # apply this seed to target tranfsorms
                # torch.manual_seed(seed)  # needed for torchvision 0.7
                # sample_aug = trans(sample)

                new_idx = extended_size - data_augmentation_samples + i
                self.data[new_idx, :, :] = sample

        if normalize:
            # center temp data
            self.data[:, 2, :] = self.data[:, 2, :] - temp_offset

            self.mean_values = self.data.mean(dim=[0, 2])
            for i in range(5):
                self.data[:, i, :] = self.data[:, i, :] - self.mean_values[i]

            self.scaling_factors_inv = 1.0 / (self.data.abs().max(dim=0).values.max(dim=1).values)
            for i in range(5):
                self.data[:, i, :] = self.data[:, i, :] * self.scaling_factors_inv[i]

            # calculate scaling factors
            # v_x_max_inv = 1.0 / self.dataset_tensor[:, 0, :, :].abs().max().item()
            # v_y_max_inv = 1.0 / self.dataset_tensor[:, 1, :, :].abs().max().item()
            # temp_max_inv = 1.0 / self.dataset_tensor[:, 2, :, :].abs().max().item()
            # self.dataset_tensor[:, 0, :, :] = self.dataset_tensor[:, 0, :, :] * v_x_max_inv
            # self.dataset_tensor[:, 1, :, :] = self.dataset_tensor[:, 1, :, :] * v_y_max_inv
            # self.dataset_tensor[:, 2, :, :] = self.dataset_tensor[:, 2, :, :] * temp_max_inv

            # self.norm_v_x[1] = 1.0 / v_x_max_inv
            # self.norm_v_y[1] = 1.0 / v_y_max_inv
            # self.norm_temp[1] = 1.0 / temp_max_inv

        data_size = self.data.shape[0]
        self.U_tensor = torch.empty([5, self.imsize**2, self.imsize**2])
        self.S_tensor = torch.empty([5, data_size])
        self.V_tensor = torch.empty([5, data_size, data_size])

        for idx in range(5):
            self.U_tensor[idx, :, :], self.S_tensor[idx, :], self.V_tensor = svd_fields(
                self.data, idx
            )

    def get_svd_data(self):
        return self.U_tensor, self.S_tensor, self.V_tensor

    def un_normalize_temp(self, datatensor):
        datatensor /= self.scaling_factors_inv[2]
        datatensor += self.mean_values[2]
        datatensor += temp_offset

        return datatensor

    def un_normalize(self, datatensor):
        for idx in range(5):
            datatensor[:, idx, :] = datatensor[:, idx, :] / self.scaling_factors_inv[idx]
            datatensor[:, idx, :] = datatensor[:, idx, :] + self.mean_values[idx]

        datatensor[:, 2, :] += temp_offset

        return datatensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        "return input and label (index 2 = temp)"
        # todo check target shape
        return self.data[idx, :, :], self.data[idx, 2, :]


def load_vtk_file_all(file_path_vel: str, imsize: int):
    """loads mesh and tmeperature data from vtk file
    expects file path including "vel" part
    """
    # file_path = file_path_vel.replace("pflotran-new-vel-", "pflotran-new-")
    file_path = file_path_vel.replace("pflotran-noFlow-new-vel", "pflotran-withFlow-new")
    mesh = meshio.read(file_path_vel)
    data = meshio.read(file_path)

    ret_data = torch.empty([5, imsize * imsize], dtype=torch.float32)
    v_x = mesh.cell_data["Vlx"][0]
    v_y = mesh.cell_data["Vly"][0]
    temp = data.cell_data["Temperature"][0]
    perm = data.cell_data["Permeability_X"][0]
    pressure = data.cell_data["Liquid_Pressure"][0]

    ret_data[0, :] = torch.as_tensor(v_x, dtype=torch.float32).T
    ret_data[1, :] = torch.as_tensor(v_y, dtype=torch.float32).T
    ret_data[2, :] = torch.as_tensor(temp, dtype=torch.float32).T
    ret_data[3, :] = torch.as_tensor(perm, dtype=torch.float32).T
    ret_data[4, :] = torch.as_tensor(pressure, dtype=torch.float32).T

    return ret_data


def load_vtk_file(file_path_vel: str, imsize: int):
    """loads mesh and tmeperature data from vtk file
    expects file path including "vel" part
    """
    # file_path = file_path_vel.replace("pflotran-new-vel-", "pflotran-new-")
    file_path = file_path_vel.replace("pflotran-noFlow-new-vel", "pflotran-withFlow-new")
    mesh = meshio.read(file_path_vel)
    data = meshio.read(file_path)

    ret_data = torch.empty([5, imsize, imsize], dtype=torch.float32)
    v_x = mesh.cell_data["Vlx"][0].reshape([imsize, imsize])
    v_y = mesh.cell_data["Vly"][0].reshape([imsize, imsize])
    temp = data.cell_data["Temperature"][0].reshape([imsize, imsize])

    ret_data[0, :, :] = torch.as_tensor(v_x, dtype=torch.float32)
    ret_data[1, :, :] = torch.as_tensor(v_y, dtype=torch.float32)
    ret_data[2, :, :] = torch.as_tensor(temp, dtype=torch.float32)

    return ret_data


# def get_single_example():
#     data_path = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/noFlow_withFlow"
#     mf_dataset = MultiFolderDataset([data_path + "/batch1"], data_augmentation=False, imsize=64)

#     return mf_dataset[0]


def get_dataset_complete_cached(data_augmentation=False):
    data_path_cache = "/data/scratch/leiterrl/data_complete.pt"
    return CacheDataset(
        data_path_cache, imsize=64, normalize=True, data_augmentation=data_augmentation
    )


def get_dataset_all_dir_cached(data_augmentation=False):
    data_path_cache = "/data/scratch/leiterrl/data_all_dir.pt"
    return CacheDataset(
        data_path_cache, imsize=64, normalize=True, data_augmentation=data_augmentation
    )


def get_mid_perm_test_cached(data_augmentation=False):
    data_path_cache = "/data/scratch/leiterrl/data_mid_perm_test.pt"
    return CacheDataset(
        data_path_cache, imsize=65, normalize=True, data_augmentation=data_augmentation
    )


def get_mid_perm_training_cached(data_augmentation=False):
    data_path_cache = "/data/scratch/leiterrl/data_mid_perm_training.pt"
    return CacheDataset(
        data_path_cache, imsize=65, normalize=True, data_augmentation=data_augmentation
    )


def get_dataset_all_dir_test_cached(data_augmentation=False):
    data_path_cache = "/data/scratch/leiterrl/data_all_dir_test.pt"
    return CacheDataset(
        data_path_cache, imsize=64, normalize=True, data_augmentation=data_augmentation
    )


def get_dataset_4_ex_cached(data_augmentation=False):
    data_path_cache = "/data/scratch/leiterrl/data_4_ex.pt"
    return CacheDataset(
        data_path_cache, imsize=64, normalize=True, data_augmentation=data_augmentation
    )
