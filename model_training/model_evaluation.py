#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:34:15 2022

@author: munirdin
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report
import scikitplot as skplt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# plot of training and validation curves
def loss_acc_plot (history, model_name):
	# plot loss
	plt.figure(figsize = (15,5))
	plt.subplot(121)
	plt.title(f'LOSS OF {model_name}')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(122)
	plt.title(f'ACCURACY OF {model_name}')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	plt.legend(['training', 'validation'])
	plt.show()
    

def metrics (prediction, data, model_name ):
    
    #################################--Accuracy and AUC score-######################################
  # Classification report test set
    class_report = classification_report(data.classes, prediction.argmax(axis=1))
    print(f"classification_report of test dataset with {model_name}:\n")
    print(class_report)
    print("#"*60)
    #################################--Accuracy and AUC score-######################################
       # Accuracy test set 
    acc = accuracy_score( data.classes, prediction.argmax(axis=1))
    from sklearn.metrics import confusion_matrix
    print(f'validation_accuracy of {model_name}: {acc}')
   
    tn, fp, fn, tp = confusion_matrix(y_true=data.classes, y_pred = prediction.argmax(axis=1)).ravel()
    false_positive_rate = round(fp / (fp + tn), 2)
    false_negative_rate = round(fn/(fn+tp), 2)
    print(f'false_positive_rate of {model_name}: {false_positive_rate}')
    print(f'false_negative_rate of {model_name}: {false_negative_rate}')

  
    ####################################--Confusion matrix --###################################
    
    skplt.metrics.plot_confusion_matrix(y_true = data.classes, 
                                        y_pred= prediction.argmax(axis=1),
                                        figsize = (10, 7))
  
    ##############################----ROC----#########################################
    skplt.metrics.plot_roc(data.classes, prediction, figsize = (8,7))
    plt.title (f"ROC curve on final testdata with {model_name}")
    plt.show()

    return 

