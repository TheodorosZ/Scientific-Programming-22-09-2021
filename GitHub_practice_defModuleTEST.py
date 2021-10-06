# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:11:08 2021

@author: maart
"""
import pandas as pd
import numpy as np

def defScale(Xtrain, Xtest, scaletype):
    if scaletype == 'mean-center':
        data = Xtrain
        mean_vector = data.mean(axis=1)
        
        print(data.shape())
        
        mean_centered_data = np.divide(data, mean_vector)
        
        mean_centered_test = np.divide(Xtest, mean_vector)
        return mean_centered_data, mean_centered_test
    if scaletype == 'auto':
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
    if scaletype == 'pareto':
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
            
# def scale_pareto(Xtrain, Xtest):
#     data = Xtrain
#     mean_vector = data.mean(axis=0)
#     mean_centering = np.divide(data, mean_vector)
    
#     # calculate std and the square root of std
#     std_vector = np.std(data, axis=0)
#     root_std = np.sqrt(std_vector)
    
#     # paretoscaling
#     scaled_data = np.divide(mean_centering, root_std)
    
#     # test data
#     mean_centered_test = np.divide(Xtest, mean_vector)
#     test_scaled_data = np.divide(mean_centered_test, root_std)
#     return scaled_data, test_scaled_data

# def scale_auto(Xtrain, Xtest):
#     data = Xtrain
#     mean_vector = data.mean(axis=0)
#     mean_centering = np.divide(data, mean_vector)
    
#     # calculate std
#     std_vector = np.std(data, axis=0)
    
#     # autoscaling
#     scaled_data = np.divide(mean_centering, std_vector)
    
#     # test data
#     mean_centered_test = np.divide(Xtest, mean_vector)
#     test_scaled_data = np.divide(mean_centered_test, std_vector)
    
#     return scaled_data, test_scaled_data
