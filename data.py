import os
import meshio
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import RandomCrop, Resize, Compose, RandomRotation
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import CenterCrop


class MultiFolderDataset(Dataset):
    def __init__(
        self,
        folder_list: "list[str]",
        imsize: int = 64,
        normalize: bool = True,
        data_augmentation: bool = False,
    ) -> None:
        data_augmentation_samples = 2000

        self.normalize = normalize
        self.imsize: int = imsize
        self.dataset_size: int = 0
        eligible_files: "list[str]" = []
        for folder in folder_list:
            for data_file in sorted(os.listdir(folder)):
                # if "pflotran-new-vel-" in data_file:
                if "pflotran-noFlow-new-vel-" in data_file:
                    eligible_files.append(os.path.join(folder, data_file))
                    self.dataset_size += 1
                    print(self.dataset_size)

        if data_augmentation:
            self.dataset_size += data_augmentation_samples

        self.dataset_tensor = torch.empty(
            [self.dataset_size, 3, self.imsize, self.imsize]
        )
        for idx, file_path in enumerate(eligible_files):
            print(idx)
            self.dataset_tensor[idx, :, :, :] = self.load_vtk_file(file_path)

        if data_augmentation:
            rand_rot_trans = RandomRotation(
                180, interpolation=InterpolationMode.BILINEAR
            )
            crop_trans = CenterCrop(45)
            resize_trans = Resize((64, 64))
            trans = Compose([rand_rot_trans, crop_trans, resize_trans])

            for i in range(data_augmentation_samples):
                # get dataset sample
                idx = random.randint(0, self.dataset_size - data_augmentation_samples)

                input = self.dataset_tensor[idx, 0:2, :, :].detach().clone()
                target = self.dataset_tensor[idx, 2, :, :].unsqueeze(0).detach().clone()

                seed = np.random.randint(2147483647)  # make a seed with numpy generator
                random.seed(seed)  # apply this seed to img tranfsorms
                torch.manual_seed(seed)  # needed for torchvision 0.7
                input = trans(input)

                random.seed(seed)  # apply this seed to target tranfsorms
                torch.manual_seed(seed)  # needed for torchvision 0.7
                target = trans(target)

                new_idx = self.dataset_size - data_augmentation_samples + i
                self.dataset_tensor[new_idx, 0:2, :, :] = input
                self.dataset_tensor[new_idx, 2, :, :] = target

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (
            self.dataset_tensor[index, 0:2, :, :],
            self.dataset_tensor[index, 2, :, :].unsqueeze(0),
        )

    def load_vtk_file(self, file_path_vel: str):
        """loads mesh and tmeperature data from vtk file
        expects file path including "vel" part
        """
        # file_path = file_path_vel.replace("pflotran-new-vel-", "pflotran-new-")
        file_path = file_path_vel.replace(
            "pflotran-noFlow-new-vel", "pflotran-withFlow-new"
        )
        mesh = meshio.read(file_path_vel)
        data = meshio.read(file_path)

        ret_data = torch.empty([3, self.imsize, self.imsize], dtype=torch.float32)
        v_x = mesh.cell_data["Vlx"][0].reshape([self.imsize, self.imsize])
        v_y = mesh.cell_data["Vly"][0].reshape([self.imsize, self.imsize])
        temp = (data.cell_data["Temperature"][0] - 10).reshape(
            [self.imsize, self.imsize]
        )

        if self.normalize:
            v_x = (v_x / v_x.max()) * 2.0 - 1.0
            v_y = (v_y / v_y.max()) * 2.0 - 1.0
            temp = (temp / temp.max()) * 2.0 - 1.0

        ret_data[0, :, :] = torch.as_tensor(v_x, dtype=torch.float32)
        ret_data[1, :, :] = torch.as_tensor(v_y, dtype=torch.float32)
        ret_data[2, :, :] = torch.as_tensor(temp, dtype=torch.float32)

        return ret_data


def get_single_example():
    data_path = (
        "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/noFlow_withFlow"
    )
    mf_dataset = MultiFolderDataset([data_path + "/batch1"], data_augmentation=False)

    return mf_dataset[0]
