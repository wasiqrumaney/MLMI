import numpy as np
import h5py
from voxelgrid import VoxelGrid

"""
The script read the point cloud files and saves the as numpy array .npy with dimensions 
(N, M, M, M)

with M=16 the result is ~25 MB
with M=64 the result is ~1.5 GB  
TODO: if too large, break-up into multiple files 

for M=16 you can read alternatively read the file full_dataset_vectors.h5
This contains data augmentation and noise

with h5py.File('full_dataset_vectors.h5', 'r') as dataset:
    x_train = dataset["X_train"][:]       # (10000, 4096)
    x_test = dataset["X_test"][:]         # (10000,)
    y_train = dataset["y_train"][:]       # (2000, 4096)
    y_test = dataset["y_test"][:]         # (2000,)

x_train_16 = x_train.reshape(-1,16,16,16)    #(10000,16,16,16)
"""

M = 16                   # pixels per dimension
INCLUDE_TEST = False      #include also test data
SAVE_NAME = 'voxels_3DMNIST_{}.npy'.format(M)


voxels_list = list()
with h5py.File("train_point_clouds.h5", "r") as hf:
    for i in range(len(hf)):
        voxels_list.append( VoxelGrid(hf[str(i)]["points"][:], x_y_z=[M, M, M]).vector.astype('uint8') )
        
if INCLUDE_TEST:
    with h5py.File("test_point_clouds.h5", "r") as hf:
        for i in range(len(hf)):
            voxels_list.append( VoxelGrid(hf[str(i)]["points"][:], x_y_z=[M, M, M]).vector.astype('uint8') )
            
voxels_npy = np.stack(voxels_list, axis=0)   #numpy array shape (N,M,M,M)
np.save(SAVE_NAME,voxels_npy)


#Load the file
#data = np.load(SAVE_NAME)