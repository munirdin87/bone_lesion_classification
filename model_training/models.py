#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:19:29 2022

@author: munirdin
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as ICP_preprocess_input 
from  tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB7


def vgg16_model(input_shape):
    
    base_model = VGG16(include_top=False, input_shape=input_shape, pooling ='avg')

    for layer in base_model.layers:
        layer.trainable= False 
    x = base_model.output
    x = Dense(64, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def inceptionV3(input_shape):
    base_model = InceptionV3(include_top=False,input_shape=input_shape) 
    for layer in base_model.layers:
        layer.trainable= True 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def resnet50 (input_shape):
    base_model = ResNet50(include_top=False,input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable= False    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model 


def efficientNetB7 (input_shape):

    base_model = EfficientNetB7(include_top=False, weights='imagenet',input_shape = input_shape, pooling='avg')
    for layer in base_model.layers:
        layer.trainable= True   
    x = base_model.output
    x = Dense(64, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
