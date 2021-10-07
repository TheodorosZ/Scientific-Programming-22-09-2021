# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:11:08 2021

@author: maart
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Splitting function using stratified K-fold
def statKfold(Xdata, ylabels, Nsplits):
    skf = StratifiedKFold(n_splits=Nsplits)
    dictTrain = {}
    dictTest = {}
    dictTrainlabels = {}
    dictTestlabels = {}
    dictIDX = 0
    for train_index, test_index in skf.split(Xdata, ylabels):
        #print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain = Xdata.iloc[train_index]
        Xtest = Xdata.iloc[test_index]
        ylabels = pd.DataFrame(ylabels)
        ytrain = ylabels.iloc[train_index]
        ytest = ylabels.iloc[test_index]
        dictTrain[dictIDX] = Xtrain
        dictTest[dictIDX] = Xtest
        dictTrainlabels[dictIDX] = ytrain
        dictTestlabels[dictIDX] = ytest
        dictIDX = dictIDX + 1
    return dictTrain, dictTest, dictTrainlabels, dictTestlabels


# scale by mean-centering function
def scale_meanC(Xtrain, Xtest):
        data = Xtrain
        mean_vector = data.mean(axis=0)
        
        #print(data.shape)
        
        mean_centered_data = np.divide(data, mean_vector)
        
        mean_centered_test = np.divide(Xtest, mean_vector)
        return mean_centered_data, mean_centered_test

# pareto scaling function
def scale_pareto(Xtrain, Xtest):
    data = Xtrain
    mean_vector = data.mean(axis=0)
    mean_centering = np.divide(data, mean_vector)
    
    # calculate std and the square root of std
    std_vector = np.std(data, axis=0)
    root_std = np.sqrt(std_vector)
    
    # paretoscaling
    scaled_data = np.divide(mean_centering, root_std)
    
    # test data
    mean_centered_test = np.divide(Xtest, mean_vector)
    test_scaled_data = np.divide(mean_centered_test, root_std)
    return scaled_data, test_scaled_data

# autoscaling function
def scale_auto(Xtrain, Xtest):
    data = Xtrain
    mean_vector = data.mean(axis=0)
    mean_centering = np.divide(data, mean_vector)
    
    # calculate std
    std_vector = np.std(data, axis=0)
    
    # autoscaling
    scaled_data = np.divide(mean_centering, std_vector)
    
    # test data
    mean_centered_test = np.divide(Xtest, mean_vector)
    test_scaled_data = np.divide(mean_centered_test, std_vector)
    
    return scaled_data, test_scaled_data
