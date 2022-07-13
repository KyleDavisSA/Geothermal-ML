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


# folder_4 = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/rbf_n_4"
# folder_4_ex = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/rbf_n_4_extra"
# folder_8 = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/rbf_n_8"
# folder_testing = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/testing"
# folder_all_dir = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/all_direction"
# folder_mid_perm_test = (
#     "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/mid_perm_testing"
# )
# folder_all_dir_test = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/all_dir_test"

# range_4 = range(1, 120)
# range_4_ex = range(240, 360)
# range_8 = range(120, 240)
# range_testing = range(1, 40)

# files_4 = get_eligible_vtk_files(folder_4)
# files_all_dir = get_eligible_vtk_files(folder_all_dir)
# files_all_dir_test = get_eligible_vtk_files(folder_all_dir_test)
# files_4_ex = get_eligible_vtk_files(folder_4_ex)
# files_8 = get_eligible_vtk_files(folder_8)
# files_testing = get_eligible_vtk_files(folder_testing)
# files_mid_perm_test = get_eligible_vtk_files(folder_mid_perm_test)


# size_4 = len(files_4)
# size_4_ex = len(files_4_ex)
# size_8 = len(files_8)
# size_testing = len(files_testing)
# size_all_dir = len(files_all_dir)
# size_all_dir_test = len(files_all_dir_test)
# size_mid_perm_test = len(files_mid_perm_test)
# print(f"4: {size_4}, 4_ex: {size_4_ex} 8: {size_8}, testing: {size_testing}")

# tensor_4 = torch.empty([size_4, 3, imsize, imsize])
# tensor_4_ex = torch.empty([size_4_ex, 3, imsize, imsize])
# tensor_8 = torch.empty([size_8, 3, imsize, imsize])
# tensor_testing = torch.empty([size_testing, 3, imsize, imsize])
# tensor_all_dir = torch.empty([size_all_dir, 3, imsize, imsize])
# tensor_all_dir_test = torch.empty([size_all_dir_test, 3, imsize, imsize])
# tensor_mid_perm_test = torch.empty([size_mid_perm_test, 3, imsize_mid_perm, imsize_mid_perm])

# for idx, file_path in enumerate(files_4):
#     print(idx)
#     tensor_4[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_4_ex):
#     print(idx)
#     tensor_4_ex[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_all_dir):
#     print(idx)
#     tensor_all_dir[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_all_dir_test):
#     print(idx)
#     tensor_all_dir_test[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_mid_perm_test):
#     print(idx)
#     tensor_mid_perm_test[idx, :, :, :] = load_vtk_file(file_path, imsize_mid_perm)

# for idx, file_path in enumerate(files_8):
#     print(idx)
#     tensor_8[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_testing):
#     print(idx)
#     tensor_testing[idx, :, :, :] = load_vtk_file(file_path, imsize)

# torch.save(tensor_4, cache_dir + "data_4.pt")
# torch.save(tensor_all_dir_test, cache_dir + "data_all_dir_test.pt")
# torch.save(tensor_mid_perm_test, cache_dir + "data_mid_perm_test.pt")
# torch.save(tensor_8, cache_dir + "data_8.pt")
# torch.save(tensor_testing, cache_dir + "data_testing.pt")
# torch.save(torch.cat((tensor_4, tensor_8, tensor_testing), dim=0), cache_dir + "data_complete.pt")

# convert_dataset_folder("mid_perm_training", 65)
convert_dataset_folder_all("all_direction", 64)
print("done")
