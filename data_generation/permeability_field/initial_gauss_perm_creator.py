# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:11:25 2020

@author: Fabian Böttcher - Modified by Kyle Davis
"""

import os
import numpy as np
from h5py import File
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from FyeldGenerator import generate_field
from matplotlib import use
import random


# use("Agg")


# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    c = np.random.normal(loc=0, scale=1, size=shape)
    return a * c + 1j * b


def rescaleArray(array):
    a_min = np.min(array)
    a_max = np.max(array)
    norm_array = (1 * (array - a_min) / (a_max - a_min)) - 0
    return norm_array


# helper functions
def normalize(array):
    a_min = np.min(array)
    a_max = np.max(array)
    norm_array = (array - a_min) / (a_max - a_min)
    return norm_array


def initial_permeability_variation(x, y):
    print(x)
    """calculate example permeability field for calibration testing"""
    # output_values = rbf(x*(n-1),y*(n-1))
    # var_val = 4 * x * (1 - x) * np.cos(3 * np.pi * x) * np.sin(3 * np.pi * y)
    # output_values = rescaleArray(output_values)
    # output_values[:,1] = rescaleArray(output_values[:,1])
    var_val = rescaleArray(output_values)
    return -20 + (3 * var_val)
    # return -19


###########################################
# variables for setup #####################
###########################################

# Load file with cell center locations (x,y)
cell_centers_path = "grid_cell_centers.dat"

# mean initial permeability value
# const_permeability = 1e-9

# scaling factor to increase permeability field differences
# scale_factor = 2
plot = False

# permeability dataset file name
# has to match the link in the PFLOTRAN input file
filename = "permeability.h5"


x = []
y = []
# fieldRBF = []
# fieldRBFOut = []

xOut = []
yOut = []

width = 4
WidthSmall = width * 1.2
n = 6  # Number of points on coarse grid
nOut = 64  # Number of points in PFLOTRAN grid

nOutput = []

plt.figure()

for i in range(0, n):
    for j in range(0, n):
        nOutput = random.uniform(-1, 1)
        if i == 0:
            x.append((-1) * (WidthSmall / n))
        elif i == n - 1:
            x.append((i + 1) * (WidthSmall / n))
        else:
            x.append(i * (WidthSmall / n))
        if j == 0:
            y.append(-1 * (WidthSmall / n))
        elif j == n - 1:
            y.append((j + 1) * (WidthSmall / n))
        else:
            y.append(j * (WidthSmall / n))
        # y.append(j)
        # field.append(0)

for i in range(0, nOut):
    for j in range(0, nOut):
        xOut.append(i * (width / nOut))
        yOut.append(j * (width / nOut))

xOut = np.asarray(xOut)
yOut = np.asarray(yOut)
# field = np.asarray(x)

# Create shapes for generating the permeabilityy field
shape = (n, n)
shapeOut = (nOut, nOut)

field = generate_field(distrib, Pkgen(2), shape)
print("Min field: ", field.min())
# np.interp(field, (field.min(), field.max()), (-2, +2))
field[:, 0] = rescaleArray(field[:, 0])
field[:, 1] = rescaleArray(field[:, 1])
print("Min field: ", field.min())


print("Max X coarse: ", max(x))
print("Max Y coarse: ", max(y))
print("Min X coarse: ", min(x))
print("Min Y coarse: ", min(y))
print("Max X fine: ", max(xOut))
print("Max Y fine: ", max(yOut))
print("Min X fine: ", min(xOut))
print("Min Y fine: ", min(yOut))
print(x)
print(y)
print(field)
# input()
# for i in range(0,n):
#    x[i] *= 10
#    y[i] *= 10

# New method: Generate random values at each coarse mesh point
count = 0
for i in range(0, n):
    for j in range(0, n):
        field[i, j] = random.uniform(-1, 1)
        count += 1

# Create RBF interpolation field
rbf = Rbf(x, y, field, function="thin_plate")
# Evaluate RBF on PFLOTRAN mesh
output_values = rbf(xOut, yOut)


# fieldOut = generate_field(distrib, Pkgen(4), shapeOut)
# Replace PFLOTRAN field output with RBF interpolation values
# count = 0
# for i in range(0, nOut):
#     for j in range(0, nOut):
#         fieldOut[i, j] = output_values[count]
#         count += 1

# print(max(fieldOut[0]))
# print(max(fieldOut[1]))
# print(min(fieldOut[0]))
# print(min(fieldOut[1]))

# plt.imshow(field, cmap="jet")

# plt.savefig("field.png", dpi=400)
# plt.close()

# plt.figure()
# plt.imshow(fieldOut, cmap="jet")
# plt.savefig("fieldOut.png", dpi=400)
# plt.close()

# exit()

# END #####################################


def generate_perm_field():
    #######################################
    # initial permeability dataset creation
    #######################################

    # get cell center coordinates for permeability interpolation
    if os.path.isfile(cell_centers_path):
        cell_center_coords = np.loadtxt(cell_centers_path)
        perm_grid = initial_permeability_variation(
            normalize(cell_center_coords[:, 0]), normalize(cell_center_coords[:, 1])
        )
        # Transform field values into the permeability field range
        perm_grid = np.exp(perm_grid)
        # perm_grid_log = np.log(perm_grid)
        # perm_grid = np.exp(perm_grid_log.mean() + ((perm_grid_log - perm_grid_log.mean()) * scale_factor))
        print("perm_grid: ", perm_grid)
        if plot:
            plt.scatter(cell_center_coords[:, 0], cell_center_coords[:, 1], c=perm_grid)
            plt.show()
        print("allocating cell index array....")
        iarray = np.arange(1, cell_center_coords.shape[0] + 1, 1)

        print("setting cell indices....", len(iarray))
        # add cell ids to file
        dataset_name = "Cell Ids"
        h5file = File(filename, mode="w")
        h5dset = h5file.create_dataset(dataset_name, data=iarray)
        # add permeability to file
        dataset_name = "Permeability"
        h5dset = h5file.create_dataset(dataset_name, data=perm_grid)
        h5file.close()

        # debug output
        print("min: %e" % np.max(perm_grid))
        print("max: %e" % np.min(perm_grid))
        print("ave: %e" % np.mean(perm_grid))
        print("\n")
        print("done with dummy permeability field ")
        print("(used in initial simulation run to create dummy real world observations)")
    else:
        print("Error: No cell centers file found.")


if __name__ == "__main__":
    generate_perm_field()
