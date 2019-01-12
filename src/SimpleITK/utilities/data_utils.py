"""Data utilities """
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from skimage.transform import resize
import torch


class VertSegDataset(Dataset):
    """VertSegV1 dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string) : Path to the csv file containing the scores.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.scores =  pd.read_csv(csv_file,header=1)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        """ Returns the length of the dataset"""
        return len(self.scores)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.scores.iloc[idx, 0])
        #print(image_name)
        mask_name = image_name.replace("\image","/mask")+str(".mhd")
        #print(mask_name)
        sample = sitk.ReadImage(mask_name)

        if self.transform:
            sample = self.transform(sample)

        return sample

class ISOResample(object):#, new_spacing=[1,1,1]):
    """Resamples a given scan with new given spacing.
    Args:
    new_size(tuple): new scan size (x,y,z)
    new_spacing(list or tuple): new spacing between scan pixels (x,y,z)
    """
    def __init__(self,new_spacing=[1,1,1]):
        assert isinstance(new_spacing, (list, tuple))
        #self.new_size    = new_size
        self.new_spacing = new_spacing
    #to make the ISOResample a callable class, i.e no need to pass the parameters every time it is called     
    def __call__(self, sample):
        
        size = np.array(sample.GetSize())
        spacing = np.array(sample.GetSpacing())
        physical_ext = size*spacing
        new_size = [int(round(physical_ext[0])),
                    int(round(physical_ext[1])),
                    int(round(physical_ext[2]))]
        
        resampleFilter = sitk.ResampleImageFilter()
        
        res_img = resampleFilter.Execute(sample, new_size, sitk.Transform(), 
                                     sitk.sitkNearestNeighbor, sample.GetOrigin(),
                                     self.new_spacing, sample.GetDirection(), 0, sample.GetPixelID())
        return res_img 
    
class MinMaxCrop(object):
    """ Crops the mask keeping the non-zero pixels only"""
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample_nda = sitk.GetArrayViewFromImage(sample)
        nnz_idx = np.where(sample_nda>0)
        #NB: the axis order of the image are reversed in the array  (x,y,z) -> (z,y,x)
        min_z, max_z = min(nnz_idx[0]), max(nnz_idx[0])
        min_y, max_y = min(nnz_idx[1]), max(nnz_idx[1])
        min_x, max_x = min(nnz_idx[2]), max(nnz_idx[2])

        cut_mask = sample[min_x:max_x,min_y:max_y,min_z:max_z]
        
        return cut_mask 

    
class Resize(object):
    """Resizes the scan after croping"""
    
    def __init__(self,new_size=(74,74,74)):
        assert isinstance(new_size, (list, tuple))
        self.new_size = new_size

    def __call__(self,sample):
        sample_nda  = sitk.GetArrayViewFromImage(sample)
        cut_resized = resize(sample_nda, self.new_size,mode='reflect', preserve_range=True, clip=True)
        
        return cut_resized

def ToTensor(sample):
    image = sample
    return torch.from_numpy(image)

# plots the slice at the middle for each axis
# img should be a numpy array
def plot_middle_slices(img):
    #print("Image shape ",img.shape)
    _=fig = plt.figure()
    _=plt.subplot(1,3,1)
    _=plt.imshow(img[img.shape[0]//2,:,:])
    _=plt.subplot(1,3,2)
    _=plt.imshow(img[:,img.shape[1]//2,:])
    _=plt.subplot(1,3,3)
    _=plt.imshow(img[:,:,img.shape[2]//2])