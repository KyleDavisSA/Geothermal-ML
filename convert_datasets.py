import torch
from data import load_vtk_file, get_eligible_vtk_files, load_vtk_file_all
import os


scratch_dir = "/data/scratch/"
base_dir = scratch_dir + "leiterrl/geoml"
cache_dir = scratch_dir + "leiterrl/"

imsize = 64
imsize_mid_perm = 65


def convert_dataset_folder(name: str, res: int):
    folder_str = os.path.join(
        "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data", name
    )
    files = get_eligible_vtk_files(folder_str)
    data_size = len(files)
    data_tensor = torch.empty([data_size, 3, res, res])
    for idx, file_path in enumerate(files):
        print(idx)
        data_tensor[idx, :, :, :] = load_vtk_file(file_path, res)

    torch.save(data_tensor, cache_dir + "data_" + name + ".pt")


def convert_dataset_folder_all(name: str, res: int):
    """
    convert dataset from vtk files to pt file including all fields (pres, perm, temp etc.)
    """

    folder_str = os.path.join(
        "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data", name
    )
    files = get_eligible_vtk_files(folder_str)

    data_size = len(files)
    data_tensor = torch.empty([data_size, 5, res * res])
    for idx, file_path in enumerate(files):
        print(idx)
        data_tensor[idx, :, :] = load_vtk_file_all(file_path, res)

    torch.save(data_tensor, cache_dir + "data_all_" + name + ".pt")

# convert_dataset_folder("mid_perm_training", 65)
convert_dataset_folder_all("all_direction", 64)
print("done")
