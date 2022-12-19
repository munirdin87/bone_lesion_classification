#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 07:59:59 2022

@author: munirdin

"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random 
import os
from PIL import Image
from matplotlib import image
import glob
from skimage.measure import label, regionprops
from skimage import filters

################################################--GPU_ON--#################################
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
def load_data (orginal_path, label_path):
    
    patient_number = label_path[-13:-4] # get patient number 
    orginal_data  = nib.load(os.path.join(orginal_path, patient_number + "_0000.nii" )).get_fdata()
    label_data  = nib.load(label_path).get_fdata()
    return orginal_data,label_data , patient_number

def regionprops_and_crop (image_data, label_data, patient_number ):
    
   
    for i in range(label_data.shape[2]):
        threshold = filters.threshold_otsu(label_data[:,:,i]) 
        mask = label_data[:,:,i] > threshold
        slice_label = label(mask)
        regions = regionprops(slice_label)
        
        for props in regions:
            x0, y0 = props.centroid
            randomInt = random.randint(10, 60)# shifted the leasion with random int
            x0 = int(x0+randomInt)
            y0 = int(y0+randomInt)
            z0 = i
            if  57 < x0 < 710 and 57 < y0 < 710 :
                slice_label = label_data[:,:,z0]
                slice_im = image_data [:,:,z0]
                crop_label = slice_label[(x0-112):(x0+112),(y0-112):(y0+112)]
                crop_im    = slice_im [(x0-112):(x0+112), (y0-112):(y0+112)] # crop it 

                if crop_im.shape == (224,224)  and crop_label.sum() > 0:
                    image.imsave(os.path.join(save_img_path,f"lesion_{patient_number[:6]}{x0}_{y0}_{z0}.png"), crop_im, cmap ='gray')
                    print(f"lesion_{patient_number[:6]}{x0}_{y0}_{z0}.png saved")         
    return 


    
#####################################################  RUN #######################################
orginal_path ="path/to/CT scans" # .nii file
label_path= "path/to/label/file" # .nii file 
save_img_path = "/path/to/save/imgs"

def final_run(label_path, orginal_path):
    
    label_path_list = glob.glob( os.path.join(label_path, "*.nii"))
    for i in range (len(label_path_list)):
        
        image_data, label_data, patient_number = load_data (orginal_path, label_path_list[i])
        regionprops_and_crop (image_data, label_data, patient_number)
                  
    return 

final_run(label_path, orginal_path) # call the function 


    #Close the GPU
from numba import cuda
cuda.close()
