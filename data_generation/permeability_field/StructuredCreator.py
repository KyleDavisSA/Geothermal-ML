# -*- coding: utf-8 -*-
"""
@author: Kyle Davis

Creates a structured grid for PFLOTRAN simulations
"""

import os
import numpy as np
from h5py import File
import matplotlib.pyplot as plt


xWidth = 128	# Width in meters in X direction
yWidth = 128	# Width in meters in Y direction
zWidth = 1
xGrid = 64	# Number of grid points in meters in X direction
yGrid = 64	# Number of grid points in meters in Y direction
zGrid = 1

coords = np.zeros((xGrid*yGrid,3))

cellXWidth = xWidth/xGrid
cellYWidth = yWidth/yGrid
cellZWidth = zWidth/zGrid

totalCells = xGrid*yGrid*zGrid
cellID = 0
print("CELLS ", xGrid*yGrid)
for j in range(0,yGrid):
	for i in range(0,xGrid):
		xloc = 0.5*cellXWidth + i*cellXWidth
		yloc = 0.5*cellYWidth + j*cellYWidth
		zloc = 0.5*cellZWidth
		coords[i + j*xGrid, 0] = xloc
		coords[i + j*xGrid, 1] = yloc
		coords[i + j*xGrid, 2] = zloc
		volume = 1
		cellID = (i+1) + j*yGrid
		print(cellID, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  volume)
		
		
		
print("CONNECTIONS")
for j in range(0,yGrid):
	for i in range(0,xGrid):
		if (i < (xGrid - 1)):
			cellID_1 = (i+1) + j*yGrid
			cellID_2 = (i+1) + j*yGrid + 1
			xloc = cellXWidth + i*cellXWidth
			yloc = 0.5*cellYWidth + j*cellYWidth
			zloc = 0.5*cellZWidth
			faceArea = 1
			print(cellID_1, "  ", cellID_2, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  faceArea)
		
		if (j < (yGrid - 1)):
			cellID_1 = (i+1) + j*yGrid
			cellID_2 = (i+1) + j*yGrid + xGrid
			xloc = 0.5*cellXWidth + i*cellXWidth
			yloc = cellYWidth + j*cellYWidth
			zloc = 0.5*cellZWidth
			faceArea = 1
			print(cellID_1, "  ", cellID_2, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  faceArea)


print("NorthBC")
for i in range(0,xGrid):
	cellID = (xGrid*(yGrid-1)) + i + 1
	xloc = 0.5*cellXWidth + i*cellXWidth
	yloc = yWidth
	zloc = 0.5*cellZWidth
	faceArea = 1
	print(cellID, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  faceArea)
	
print("SouthBC")
for i in range(0,xGrid):
	cellID = i + 1
	xloc = 0.5*cellXWidth + i*cellXWidth
	yloc = 0
	zloc = 0.5*cellZWidth
	faceArea = 1
	print(cellID, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  faceArea)
	
print("WestBC")
for i in range(0,yGrid):
	cellID = 1 + i*xGrid
	xloc = 0
	yloc = 0.5*cellYWidth + i*cellYWidth
	zloc = 0.5*cellZWidth
	faceArea = 1
	print(cellID, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  faceArea)
	
print("EastBC")
for i in range(0,yGrid):
	cellID = i*xGrid +xGrid
	xloc = xWidth
	yloc = 0.5*cellYWidth + i*cellYWidth
	zloc = 0.5*cellZWidth
	faceArea = 1
	print(cellID, "  ", xloc, "  ", yloc, "  ", zloc, "  ",  faceArea)



def initial_permeability_variation(x, y):
    """calculate example permeability field for calibration testing"""
    var_val = x*0 + 1e-9 + y*0
    return var_val
    
#perm_grid = initial_permeability_variation(coords[:,0], coords[:,1])

#for i in range(0,xGrid*yGrid):
	#print(coords[i,0], coords[i,1], coords[i,2])



cell_centers_path = "grid_cell_centers.dat"
cell_center_coords = np.loadtxt(cell_centers_path)
#perm_grid = initial_permeability_variation(cell_center_coords[:, 0], cell_center_coords[:, 1])
scale_factor = 2
#perm_grid_log = np.log(perm_grid)
#perm_grid = np.exp(perm_grid_log.mean() + ((perm_grid_log - perm_grid_log.mean()) * scale_factor))

#print(cell_center_coords)
#plt.scatter(cell_center_coords[:, 0], cell_center_coords[:, 1], c=perm_grid)
#plt.show()

iarray = [] #np.arange(1, cell_center_coords.shape[0] + 1, 1)
perm_grid = []
for i in range(xGrid*yGrid):
	iarray.append(i+1)
	perm_grid.append(0.000000001)


filename = 'permeability.h5'

print(iarray)
print(perm_grid)

print('setting cell indices....')
# add cell ids to file
dataset_name = 'Cell Ids'
h5file = File(filename, mode='w')
h5dset = h5file.create_dataset(dataset_name, data=iarray)
# add permeability to file
dataset_name = 'Permeability'
h5dset = h5file.create_dataset(dataset_name, data=iarray)
h5file.close()

# debug output
print('min: %e' % np.max(perm_grid))
print('max: %e' % np.min(perm_grid))
print('ave: %e' % np.mean(perm_grid))
    
