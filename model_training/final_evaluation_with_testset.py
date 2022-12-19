#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:09:09 2022

@author: munirdin
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure, img_as_float
from  skimage.util import random_noise
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input 
from tensorflow.keras.applications.inception_v3 import preprocess_input as ICP_preprocess_input 
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input 
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess_input 
from tensorflow.keras.models import  load_model
from model_evaluation import loss_acc_plot, metrics
'''
# enable GPU
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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
nvidiaconfig = ConfigProto()
nvidiaconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
nvidiaconfig.gpu_options.allow_growth = False
session = InteractiveSession(config=nvidiaconfig)
'''
#################################################################################################
test_data_path = "/path/to/testdata"

test_gen = ImageDataGenerator(preprocessing_function = effnet_preprocess_input, width_shift_range=0.2, height_shift_range=0.2)
test_data = test_gen.flow_from_directory(test_data_path, 
                                         class_mode='categorical',
                                         batch_size=8, 
                                         target_size=(224,224), 
                                           shuffle=False)

model_path = "/path/to/saved/model"

model =load_model(model_path)

prediction = model.predict(test_data) # predicton on test data

# call the evaluation function that is created in model-1
metrics(prediction = prediction, data = test_data, model_name = "model_name")
