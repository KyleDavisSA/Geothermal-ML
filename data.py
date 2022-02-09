import os
import meshio
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MultiFolderDataset(Dataset):
    def __init__(self, folder_list: "list[str]", imsize: int = 64) -> None:
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

        self.dataset_tensor = torch.empty(
            [self.dataset_size, 3, self.imsize, self.imsize]
        )
        for idx, file_path in enumerate(eligible_files):
            print(idx)
            self.dataset_tensor[idx, :, :, :] = self.load_vtk_file(file_path)

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

        ret_data = torch.empty([3, self.imsize, self.imsize])
        v_x = mesh.cell_data["Vlx"][0].reshape([self.imsize, self.imsize])
        v_y = mesh.cell_data["Vly"][0].reshape([self.imsize, self.imsize])
        temp = (data.cell_data["Temperature"][0] - 10).reshape(
            [self.imsize, self.imsize]
        )
        ret_data[0, :, :] = torch.as_tensor(v_x)
        ret_data[1, :, :] = torch.as_tensor(v_y)
        ret_data[2, :, :] = torch.as_tensor(temp)

        return ret_data
