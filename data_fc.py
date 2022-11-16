import os
import meshio
import torch
import math
import numpy as np
from torch.utils.data import Dataset
from scipy import interpolate
import matplotlib.pyplot as plt


temp_offset = 10

def get_eligible_vtk_files(folder: str) -> "list[str]":
    eligible_files: "list[str]" = []
    for data_file in sorted(os.listdir(folder)):
        if "pflotran-withFlow-new-vel-" in data_file:
            eligible_files.append(os.path.join(folder, data_file))

    return eligible_files

        
class MultiFolderDataset(Dataset):
    def __init__(
        self,
        folder_list: "list[str]",
        test_folders: "list[int]",
        imsize: int,
        normalize: bool = False,
        test: bool = True,
    ) -> None:


        self.normalize = normalize
        self.imsize: int = imsize
        self.dataset_size: int = 0
        # Find all files within the "data" to extract plume data from
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
        # The torch tensor stores [q_x, q_y, temperature] from the data set of pictures of imsize*imsize
        self.dataset_tensor = torch.empty([self.dataset_size, 3, self.imsize, self.imsize])
        for idx, file_path in enumerate(eligible_files):
            self.dataset_tensor[idx, :, :, :] = load_vtk_file(file_path, self.imsize)

        self.norm_v_x = [0.0, 1.0]
        self.norm_v_y = [0.0, 1.0]
        self.norm_temp = [temp_offset, 1.0]
        
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

    def extract_plume_data(self, view_image, length_plume, width_plume, length_scaling, width_scaling):

        total_samples = self.dataset_size          # Total number of samples to extract data from
        total_steps = length_plume            # Total number of steps along plume streamline
        scaling_length_l = length_scaling      # Distance each step takes along plume streamline
        width = width_plume                  # Total number of steps perpendicular to plume streamline
        assert(width % 2 != 0), "Width must be an odd number to have equal distance either side of plume streamline"
        mid_width = int((width-1)/2)    # location of plume in width
        scaling_length_w = width_scaling      # Distance each width step takes perpendicular plume streamline

        temperature = np.zeros((total_samples,total_steps,width))
        Vmax = np.zeros((total_samples,total_steps,width))      # Absolute velocity
        qx = np.zeros((total_samples,total_steps,width))
        qy = np.zeros((total_samples,total_steps,width))

        # -----------------------------------------------------------------------------------------
        if (view_image):
            # To view the temperature field and velocity streamline through the heat pump location. 
            for i in range(5):
                x = np.linspace(1, 65, 65)
                y = np.linspace(1, 65, 65)
                X, Y = np.meshgrid(x, y)
                Temp = self.dataset_tensor[i,2,:,:]
                u = self.dataset_tensor[i,0,:,:]        # Velocity in X direction
                v = self.dataset_tensor[i,1,:,:]        # Velocity in y direction
                density = 50                            # Density of streamlines
                starting_point = np.array([[32,32]])    # Heat pump locations
                plt.streamplot(X, Y, u, v, density=density, start_points=starting_point)

                cp = plt.contourf(X, Y, Temp, levels=[11,12,13,14,15],cmap='viridis')
                # Obtaining polgon coordinates of plumes 
                # https://www.tutorialspoint.com/how-to-get-coordinates-from-the-contour-in-matplotlib
                for item in cp.collections:
                    for i in item.get_paths():
                        v = i.vertices
                        x = v[:, 0]
                        y = v[:, 1]
                        print(x, y)
                plt.colorbar(cp)
                plt.show()
        # -----------------------------------------------------------------------------------------

        plume_data = np.zeros((total_samples,total_steps,3))
        locations_plume = np.zeros((total_samples,total_steps,2))       # [x,y] coordinate of plume streamline at each step
        loc_off_plume_x = np.zeros((total_samples,total_steps,width))   # [x] coordinate of each measurement point in domain
        loc_off_plume_y = np.zeros((total_samples,total_steps,width))   # [y] coordinate of each measurement point in domain

        for k in range(total_samples):
            mid_x = 64.00                                           # heat pump x distance
            mid_y = 64.00                                           # heat pump y distance
            q_x_mid = self.dataset_tensor[k,0,32,32]                # q_x at heat pump
            q_y_mid = self.dataset_tensor[k,1,32,32]                # q_y at heat pump
            plume_data[k,0,0] = q_x_mid
            plume_data[k,0,1] = q_y_mid
            plume_data[k,0,2] = self.dataset_tensor[k,2,32,32]
            locations_plume[k,0,0] = 64.00
            locations_plume[k,0,1] = 64.00

            vec_length = math.sqrt(q_x_mid**2 + q_y_mid**2 )        # Length of velocity vector
            scaling_vec = scaling_length_l/vec_length               # Obtain values every <scaling_length_l> metres
            scaling_vec_w = scaling_length_w/vec_length             # Obtain values every <scaling_length_w> metres

            q_x_scaled = q_x_mid*abs(scaling_vec)                   
            q_y_scaled = q_y_mid*abs(scaling_vec)
            q_x_scaled_w = q_x_mid*abs(scaling_vec_w)
            q_y_scaled_w = q_y_mid*abs(scaling_vec_w)
            
            for w in range(width):
                # Determine the x and y coordinates at locations perpendicular to plume
                loc_off_plume_x[k,0,w] = locations_plume[k,0,0] - ((mid_width-w)*q_y_scaled_w)
                loc_off_plume_y[k,0,w] = locations_plume[k,0,1] + ((mid_width-w)*q_x_scaled_w)
        
            for i in range(1,total_steps):
                mid_x += q_x_scaled             # Move the location along the streamline in x direction
                mid_y += q_y_scaled             # Move the location along the streamline in y direction
                locations_plume[k,i,0] = mid_x
                locations_plume[k,i,1] = mid_y

                # Obtain q_x, q_y and temperature at new location along plume streamline
                vA = self.get_data_at_location(k,locations_plume[k,i,0],locations_plume[k,i,1])
                plume_data[k,i,0] = vA[0]
                plume_data[k,i,1] = vA[1]
                plume_data[k,i,2] = vA[2]

                # Determine new velocity magnitude and scale movement distance along plume
                vec_length = math.sqrt(plume_data[k,i,0]**2 + plume_data[k,i,1]**2 ) 
                scaling_vec = scaling_length_l/vec_length
                scaling_vec_w = scaling_length_w/vec_length
                q_x_scaled = plume_data[k,i,0]*abs(scaling_vec)
                q_y_scaled = plume_data[k,i,1]*abs(scaling_vec)
                q_x_scaled_w = plume_data[k,i,0]*abs(scaling_vec_w)
                q_y_scaled_w = plume_data[k,i,1]*abs(scaling_vec_w)

                for w in range(width):
                    loc_off_plume_x[k,i,w] = locations_plume[k,i,0] - ((mid_width-w)*q_y_scaled_w)
                    loc_off_plume_y[k,i,w] = locations_plume[k,i,1] + ((mid_width-w)*q_x_scaled_w)


        print(f"Determine values within plume domain")
        for i in range(total_samples):
            for j in range(total_steps):
                for k in range(width):
                    x = loc_off_plume_x[i,j,k] 
                    y = loc_off_plume_y[i,j,k] 
                    vA = self.get_data_at_location(i,x,y)
                    temperature[i,j,k] = vA[2]
                    Vmax[i,j,k] = math.sqrt(vA[0]**2 + vA[1]**2 ) 
                    qx[i,j,k] = vA[0]
                    qy[i,j,k] = vA[1]


        return temperature, Vmax, qx, qy, loc_off_plume_x, loc_off_plume_y

    

    def get_data_at_location(self,k,mid_x,mid_y):
        vA = np.zeros(3)
        xGrid = np.arange(0, 130, 2)
        yGrid = np.arange(0, 130, 2)
        for j in range(3):
            result = self.dataset_tensor[k,j,:,:].flatten()
            f = interpolate.interp2d(xGrid, yGrid, result, kind='cubic')
            vA[j] = f(mid_y,mid_x)

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