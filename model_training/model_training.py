#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:58:30 2022

@author: munirdin
"""
# import packages
'''
Please be sure that you have GPU on your server. 
select the model from importen transfer learning models
select the preprocessing function for the chosen model.
define the all paths

'''


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure, img_as_float
from  skimage.util import random_noise
from models import vgg16_model, inceptionV3, resnet50, efficientNetB7
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input 
from tensorflow.keras.applications.inception_v3 import preprocess_input as ICP_preprocess_input 
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input 
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess_input 
from model_evaluation import loss_acc_plot, metrics

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


# set the directory based on the folder where you saved dataset

train_data_dir = "/path/to/train_data"
val_data_dir = "/path/to/validation_data"

epochs = 40
batch_size =32
input_shape =192
preprocessing_function = effnet_preprocess_input # this should be changed before train pre-trained model. 
  
train_datagen = ImageDataGenerator(
                                     rotation_range=45,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                    #width_shift_range=0.25,
                                    #height_shift_range=0.25
                                    fill_mode="reflect",
                                    preprocessing_function= preprocessing_function)


val_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)

train_data = train_datagen.flow_from_directory(batch_size=batch_size,
                                                 directory=train_data_dir,
                                                 shuffle=True,
                                                 target_size=(input_shape,input_shape),
            
                                                 class_mode='categorical', 
                                                 seed=42)
                                     
val_data = val_gen.flow_from_directory(batch_size=batch_size,
                                                 directory=val_data_dir,
                                                 shuffle=False,
                                                 target_size=(input_shape,input_shape), 
                                                 class_mode='categorical',
                                                 seed=42)


def train_model(model_name, train_data, val_data, train_data_dir , input_shape, batch_size):
  
    
    train_dataSize = train_data.samples
    val_dataSize = val_data.samples    
    model = model_name(input_shape=(input_shape, input_shape,3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, 
                                                     name='categorical_crossentropy'),
        
        metrics=[tf.keras.metrics.CategoricalAccuracy(
        name='accuracy')])
    earlyStopping = EarlyStopping(monitor = 'val_loss', 
                              mode = 'min', 
                              verbose = 1, 
                              patience = 3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     factor=0.2,verbose = 1,
                                                     patience=2, 
                                                     min_lr=0.00000001)
    checkpoint_filepath = '/path/to/model/save'
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                          
                                                                   monitor='val_accuracy',
                                                                   mode='max',
                                                                   save_best_only=True)
    history = model.fit(train_data, epochs=epochs, 
                        
                        steps_per_epoch=train_dataSize // batch_size,
                        validation_data=val_data, 
                        validation_steps =val_dataSize//batch_size,
                        callbacks =[reduce_lr, earlyStopping, checkpoint_callback])
    
    return model, history





model, history =  train_model(model_name = efficientNetB7, 
                              train_data= train_data, 
                              val_data= val_data, 
                              train_data_dir= train_data_dir, 
                              input_shape = input_shape, 
                              batch_size = batch_size)
np.save('path/to/save/training/history.npy',history.history)


loss_acc_plot (history, model_name = "efficientNetB7" )


prediction = model.predict(val_data) # predicton on test data

# call the evaluation function that is created in model-1
metrics(prediction = prediction, data = val_data, model_name = "efficientNetB7")

from numba import cuda
cuda.close()
