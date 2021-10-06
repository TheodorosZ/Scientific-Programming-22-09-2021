# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:11:08 2021

@author: maart
"""
import pandas as pd
import numpy as np

def scale_meanC(X):
    data = X
    mean_vector = data.mean(axis=1)
    
    print(data.shape())
    
    mean_centered_data = np.divide(data, mean_vector)
    return mean_centered_data
    
def scale_pareto(X):
    data = X
    mean_vector = data.mean(axis=0)
    mean_centering = np.divide(data, mean_vector)
    
    # calculate std and the square root of std
    std_vector = np.std(data, axis=0)
    root_std = np.sqrt(std_vector)
    
    # paretoscaling
    scaled_data = np.divide(mean_centering, root_std)
    return scaled_data

def scale_auto(X):
    data = X
    mean_vector = data.mean(axis=0)
    mean_centering = np.divide(data, mean_vector)
    
    # calculate std
    std_vector = np.std(data, axis=0)
    
    # paretoscaling
    scaled_data = np.divide(mean_centering, std_vector)
    return scaled_data