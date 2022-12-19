#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:13:18 2022

@author: munirdin

"""
# import packages 

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random 
import os
from PIL import Image
from matplotlib import image
import glob

##############################################################################################

import GPUtil
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

try:
    gpuid = GPUtil.getFirstAvailable(order='first', maxLoad=0.5, maxMemory=0.5)[0]
    print("Using GPU")
except:
    gpuid = -1 # no gpu
    print("No GPU available")

gpuid=1
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuid)
nvidiaconfig = ConfigProto()
nvidiaconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
nvidiaconfig.gpu_options.allow_growth = False
session = InteractiveSession(config=nvidiaconfig)
##########################################################################################
# load the data 
def random_load_data(orginal_path, label_path, bonemap_path):
    
    patient_number = bonemap_path[-43:-34] # get patient number 
    orginal_data  = nib.load(os.path.join(orginal_path, patient_number + "_0000.nii" )).get_fdata()
    label_data  = nib.load(os.path.join(label_path, patient_number + ".nii" )).get_fdata()
    bonemap_data  = nib.load(bonemap_path).get_fdata()
    return orginal_data, label_data, bonemap_data, patient_number


def random_slice_plot(data):
    
    n = random.randint(0, data.shape[2]) # generate random int betwen 0 and max horizontal slice number
    plt.imshow(data[:,:,n].T, cmap='gray')
    return plt.show()  


def get_coords(bonemap_data):

    coords = np.where(bonemap_data > 0) # get coordinates where bigger than 0
    array_coords = np.array(coords) # put the data into nparray
    return array_coords 
    
# get slice from orginal data
def get_xyz_crop(orginal_data, label_data, bonemap_data, patient_number):

    coords = get_coords (bonemap_data) # call get_coords function 
    random_samples = random.sample(range((len(coords[1])-1)),100)     
    for i in random_samples:
           
        xyz = coords[:,i]
        x0 = int(xyz[0])
        y0 = int(xyz[1])
        z0 = int(xyz[2])
                        
        if  56 < x0 < 700 and 56 < y0 < 700 : # this controls the random point is not near to corner of the image
                    
            slice_bonemap = bonemap_data[:,:,z0]
            slice_label = label_data[:,:,z0]
            slice_img = orginal_data[:,:,z0]
            crop_bonemap = slice_bonemap[x0-112:x0+112,(y0-112):(y0+112)]
            crop_label = slice_label[x0-112:x0+112,(y0-112):(y0+112)]
                    
            if  crop_label.sum() == 0.0 and (crop_bonemap.sum()/ 10000) > 0.3:  
                crop_img = slice_img[(x0-112):(x0+112),(y0-112):(y0+112)]
                image.imsave(os.path.join(save_img_path,f"Nolesion_{patient_number[:5]}_{z0}.png"), crop_img, cmap ='gray')
                print(f"lesion_{patient_number[:5]}_{z0}.png saved" )
                          
    return 
    

###################################################---RUN---###############################################################

# paths for datasets and alle files in NIFTI format 

bonemap_path ='/path/to/bone_map'
orginal_path ="path/to/CT scans/"
label_path= "path/to/label" 
save_img_path = "/path/to/save/imgs"

# get all paths as a list 
def final_run():
    
    bonemap_path_list = glob.glob( os.path.join(bonemap_path, "*.nii"))
    for i in range (len(bonemap_path_list)):
        orginal_data, label_data, bonemap_data, patient_number = random_load_data (orginal_path, label_path, bonemap_path_list[i])
        get_xyz_crop(orginal_data, label_data, bonemap_data, patient_number)
        print(f"patient number:{patient_number} done with extraction")


final_run() # call the function 
    
#Close the GPU
from numba import cuda
cuda.close()
