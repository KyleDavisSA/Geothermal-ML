"""
Loads the PFLOTRAN VTK output files into a pytorch readable format

"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
import h5py
import json
import meshio
import numpy as np

def load_data_vtk_train(imageSize, nData, analyticalSolution=False):
    '''
    imageSize - number of pixels in each direction
    nData - number of data sets to read in
    '''
    
    input_data = np.zeros((ndata,2,imageSize,imageSize))
    results_data = np.zeros((ndata,1,imageSize,imageSize))
    for i in range(0,ndata):
        mesh = meshio.read("results/pflotran-vel-" + str(i) + ".vtk")
        data = meshio.read("results/pflotran-" + str(i) +".vtk")
        for k in range(0,imageSize):
          for j in range(0,imageSize):
            input_data[i,0,k,j] = mesh.cell_data["Vlx"][0][j + k*imageSize]
            input_data[i,1,k,j] = mesh.cell_data["Vly"][0][j + k*imageSize]
            results_data[i,0,k,j] = data.cell_data["Temperature"][0][j + k*imageSize] - 10
    
    return input_data, results_data
