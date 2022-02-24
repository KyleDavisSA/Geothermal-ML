import torch
from data import load_vtk_file, get_eligible_vtk_files


scratch_dir = "/data/scratch/"
base_dir = scratch_dir + "leiterrl/geoml"
cache_dir = scratch_dir + "leiterrl/"

imsize = 64

folder_4 = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/rbf_n_4"
folder_4_ex = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/rbf_n_4_extra"
folder_8 = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/rbf_n_8"
folder_testing = "/import/sgs.local/scratch/leiterrl/Geothermal-ML/PFLOTRAN-Data/testing"

# range_4 = range(1, 120)
# range_4_ex = range(240, 360)
# range_8 = range(120, 240)
# range_testing = range(1, 40)

files_4 = get_eligible_vtk_files(folder_4)
files_4_ex = get_eligible_vtk_files(folder_4_ex)
files_8 = get_eligible_vtk_files(folder_8)
files_testing = get_eligible_vtk_files(folder_testing)

size_4 = len(files_4)
size_4_ex = len(files_4_ex)
size_8 = len(files_8)
size_testing = len(files_testing)
print(f"4: {size_4}, 4_ex: {size_4_ex} 8: {size_8}, testing: {size_testing}")

tensor_4 = torch.empty([size_4, 3, imsize, imsize])
tensor_4_ex = torch.empty([size_4_ex, 3, imsize, imsize])
tensor_8 = torch.empty([size_8, 3, imsize, imsize])
tensor_testing = torch.empty([size_testing, 3, imsize, imsize])

# for idx, file_path in enumerate(files_4):
#     print(idx)
#     tensor_4[idx, :, :, :] = load_vtk_file(file_path, imsize)

for idx, file_path in enumerate(files_4_ex):
    print(idx)
    tensor_4_ex[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_8):
#     print(idx)
#     tensor_8[idx, :, :, :] = load_vtk_file(file_path, imsize)

# for idx, file_path in enumerate(files_testing):
#     print(idx)
#     tensor_testing[idx, :, :, :] = load_vtk_file(file_path, imsize)

# torch.save(tensor_4, cache_dir + "data_4.pt")
torch.save(tensor_4_ex, cache_dir + "data_4_ex.pt")
# torch.save(tensor_8, cache_dir + "data_8.pt")
# torch.save(tensor_testing, cache_dir + "data_testing.pt")
# torch.save(torch.cat((tensor_4, tensor_8, tensor_testing), dim=0), cache_dir + "data_complete.pt")
print("done")
