import os
import meshio
import torch
import math
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy import interpolate

from torchvision.transforms import RandomCrop, Resize, Compose, RandomRotation
from torchvision.transforms.functional import InterpolationMode, rotate
from torchvision.transforms.transforms import CenterCrop
import matplotlib.pyplot as plt


temp_offset = 10

def get_eligible_vtk_files(folder: str) -> "list[str]":
    eligible_files: "list[str]" = []
    for data_file in sorted(os.listdir(folder)):
        if "pflotran-withFlow-new-vel-" in data_file:
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
        '''
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
        '''

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

    def extract_plume_data(self):

        H = self.dataset_tensor[1,2,:,:]

        plt.imshow(H, interpolation='none')
        plt.show()

        total_samples = 10
        total_steps = 100
        width = 31
        mid_width = int((width-1)/2)
        scaling_length_l = 0.05
        scaling_length_w = 0.1

        temperature = np.zeros((total_samples,total_steps,width))
        Vmax = np.zeros((total_samples,total_steps,width))
        Vx = np.zeros((total_samples,total_steps,width))
        Vy = np.zeros((total_samples,total_steps,width))

        plume_data = np.zeros((total_samples,total_steps,3))
        locations_plume = np.zeros((total_samples,total_steps,2))
        loc_off_plume_x = np.zeros((total_samples,total_steps,width))
        loc_off_plume_y = np.zeros((total_samples,total_steps,width))
        
        cell_width = 2 # width of FV grids 

        for k in range(total_samples):
            mid_x = 65.001 # heat pump x distance
            mid_y = 65.001 # heat pump y distance
            q_x_mid = self.dataset_tensor[k,0,33,33]
            q_y_mid = self.dataset_tensor[k,1,33,33]
            plume_data[k,0,0] = q_x_mid
            plume_data[k,0,1] = q_y_mid
            plume_data[k,0,2] = self.dataset_tensor[k,2,33,33]
            locations_plume[k,0,0] = 65.001
            locations_plume[k,0,1] = 65.001

            # Length of velocity vector
            vec_length = math.sqrt(q_x_mid**2 + q_y_mid**2 ) 
            # Obtain values every 2 metres, therefore find scaling factor
            scaling_vec = scaling_length_l/vec_length
            scaling_vec_w = scaling_length_w/vec_length

            q_x_scaled = q_x_mid*scaling_vec
            q_y_scaled = q_y_mid*scaling_vec
            q_x_scaled_w = q_x_mid*scaling_vec_w
            q_y_scaled_w = q_y_mid*scaling_vec_w
            
            for w in range(width):
                loc_off_plume_x[k,0,w] = locations_plume[k,0,0] - ((mid_width-w)*q_y_scaled_w)
                loc_off_plume_y[k,0,w] = locations_plume[k,0,1] + ((mid_width-w)*q_x_scaled_w)

            #print(locations_plume[k,0,0] - loc_off_plume_x[k,0,2])
            #print(locations_plume[k,0,1] - loc_off_plume_y[k,0,2])
        
            for i in range(1,total_steps):
                mid_x += q_x_scaled
                mid_y += q_y_scaled
                locations_plume[k,i,0] = mid_x
                locations_plume[k,i,1] = mid_y

                vA = self.get_data_at_location(k,locations_plume[k,i,0],locations_plume[k,i,1], cell_width)
                plume_data[k,i,0] = vA[0]
                plume_data[k,i,1] = vA[1]
                plume_data[k,i,2] = vA[2]

                vec_length = math.sqrt(plume_data[k,i,0]**2 + plume_data[k,i,1]**2 ) 
                scaling_vec = scaling_length_l/vec_length
                scaling_vec_w = scaling_length_w/vec_length
                q_x_scaled = plume_data[k,i,0]*scaling_vec
                q_y_scaled = plume_data[k,i,1]*scaling_vec
                q_x_scaled_w = plume_data[k,i,0]*scaling_vec_w
                q_y_scaled_w = plume_data[k,i,1]*scaling_vec_w

                for w in range(width):
                    loc_off_plume_x[k,i,w] = locations_plume[k,i,0] - ((mid_width-w)*q_y_scaled_w)
                    loc_off_plume_y[k,i,w] = locations_plume[k,i,1] + ((mid_width-w)*q_x_scaled_w)


        print("Getting all values")
        for i in range(total_samples):
            for j in range(total_steps):
                for k in range(width):
                    x = loc_off_plume_x[i,j,k] 
                    y = loc_off_plume_y[i,j,k] 
                    vA = self.get_data_at_location(i,x,y, cell_width)
                    temperature[i,j,k] = vA[2]
                    Vmax[i,j,k] = math.sqrt(vA[0]**2 + vA[1]**2 ) 
                    Vx[i,j,k] = vA[0]
                    Vy[i,j,k] = vA[1]


        return temperature, Vmax, loc_off_plume_x, loc_off_plume_y

    def get_data_at_location(self,k,mid_x,mid_y,cell_width):
        vA = np.zeros(3)
        # https://realpython.com/python-rounding/
        multiplier = 10 ** 0
        lower_x_bound = int(math.floor((mid_x/2)*multiplier + 0.5) / multiplier)
        lower_y_bound = int(math.floor((mid_y/2)*multiplier + 0.5) / multiplier)
        upper_x_bound = int(lower_x_bound + 1)
        upper_y_bound = int(lower_y_bound + 1)
        #upper_x_bound = int(math.floor(math.ceil(mid_x)/cell_width))
        #upper_y_bound = int(math.floor(math.ceil(mid_y)/cell_width))
        #lower_x_bound = int(upper_x_bound - 1)
        #lower_y_bound = int(upper_y_bound - 1)
        #print(mid_x,mid_y)
        #print(lower_x_bound,upper_x_bound)
        #print(lower_y_bound,upper_y_bound)
        xGrid = np.arange(1, 131, 2)
        yGrid = np.arange(1, 131, 2)
        for j in range(3):
            result = self.dataset_tensor[k,j,:,:].flatten()
            f = interpolate.interp2d(xGrid, yGrid, result, kind='linear')
            vA[j] = f(mid_x,mid_y)
            #v1 = self.dataset_tensor[k,j,lower_x_bound,lower_y_bound]
            #v2 = self.dataset_tensor[k,j,upper_x_bound,lower_y_bound]
            #v3 = self.dataset_tensor[k,j,upper_x_bound,upper_y_bound]
            #v4 = self.dataset_tensor[k,j,lower_x_bound,upper_y_bound]
            
            #print("Ratio: ", (upper_x_bound*cell_width - 1) - mid_x, " - and: ",  mid_x - (lower_x_bound*cell_width - 1))
            #v12 = v1*(((upper_x_bound*cell_width - 1) - mid_x)/(cell_width)) + v2*((mid_x - (lower_x_bound*cell_width - 1))/(cell_width))

            #v34 = v3*((upper_x_bound*cell_width - 1) - mid_x)/((cell_width)) + v4*((mid_x - (lower_x_bound*cell_width - 1))/(cell_width))

            #vA[j] = v12*(((upper_y_bound*cell_width - 1) - mid_y)/(cell_width)) + v34*((mid_y - (lower_y_bound*cell_width - 1))/(cell_width)) 
            #print(v1, v2, v3, v4, vA[j])

            #print(vA[j])

        return vA
        
    


def load_vtk_file(file_path_vel: str, imsize: int):
    """loads mesh and tmeperature data from vtk file
    expects file path including "vel" part
    """
    # file_path = file_path_vel.replace("pflotran-new-vel-", "pflotran-new-")
    file_path = file_path_vel.replace("pflotran-withFlow-new-vel", "pflotran-withFlow-new")
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
    file_path = file_path_vel.replace("pflotran-withFlow-new-vel", "pflotran-withFlow-new")
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
    

