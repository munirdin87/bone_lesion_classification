#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:09:27 2022

@author: munirdin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the history 

#history_vgg16=np.load('/wecare/home/munirdin/Desktop/lesion_classification/model_training/history/InceptioV3_history.npy',allow_pickle='TRUE').item()
#history_resnet50 =np.load('/wecare/home/munirdin/Desktop/lesion_classification/model_training/history/InceptioV3_history.npy',allow_pickle='TRUE').item()
history_vgg=np.load("/wecare/home/munirdin/Desktop/lesion_classification/model_training/history/history of frozen_layers/vgg16_224_history.npy",allow_pickle='TRUE').item()
history_incp=np.load("/wecare/home/munirdin/Desktop/lesion_classification/model_training/history/history of frozen_layers/InceptioV3_224.npy",allow_pickle='TRUE').item()
history_resnet=np.load("/wecare/home/munirdin/Desktop/lesion_classification/model_training/history/history of frozen_layers/resnet_history_224.npy",allow_pickle='TRUE').item()
history_effnet=np.load("/wecare/home/munirdin/Desktop/lesion_classification/model_training/history/history of frozen_layers/efficientnet_history_224.npy",allow_pickle='TRUE').item()


history_vgg = pd.DataFrame(history_vgg)
history_incp = pd.DataFrame(history_incp)
history_resnet= pd.DataFrame(history_resnet)
history_effnet= pd.DataFrame(history_effnet)


df_vgg = history_vgg[["val_accuracy" ]]
df_vgg.columns=["VGG-16"]

df_incp = history_incp[["val_accuracy" ]]
df_incp.columns=["InceptionV3"]

df_resnet = history_resnet[["val_accuracy" ]]
df_resnet.columns=["ResNet50"]

df_effnet = history_effnet[["val_accuracy" ]]
df_effnet.columns=["EfficientNetB7"]


combine_df = pd.concat([df_vgg, df_incp, df_resnet, df_effnet], axis=1)

combine_df.plot(xlabel ="epochs", ylabel ='Val_ac'), plt.title("Validation accuracy")
plt.show()