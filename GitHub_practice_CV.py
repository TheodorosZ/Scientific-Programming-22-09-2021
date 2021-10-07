# -*- coding: utf-8 -*-
"""
Script GitHub practice 
Scientific Programming 2021 (MSB1015)
Author: MEA Koot
StudentID: I6084689

This script accompanies the MSB1011 Classication report: 
    MSB1011_ML_classification_report_Koot_i6084689

This The script is created and run with Python version 3.8 
in the free and open source ‘Scientific Python Development Environment’ 
(Spyder), accessed through Anaconda 3.0

NOTE: In order to run the script, edits have to made at steps 0.2 Instructions
are provided at the step itself
"""
# =============================================================================
# ##### 0. GETTING STARTED
# =============================================================================
# import tools
import platform
import sys 
import pandas as pd
import numpy as np 
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.io import loadmat
# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# Check Python version
print(platform.python_version()) 
#%% 
# =============================================================================
# ##### 0.2 Loading the pipelines and loading and setting the data #####
# =============================================================================
"""
# =============================================================================
##### !!IMPORTANT!! EDIT HERE:
   1. At step 0.2.1, on line ***, between the quotion makrs within sys.path.append(r'),
       insert the directory to the folder where you have stored function files called:
       nestedCV_KfeatKoot, best_modelKoot, and final_modelKoot
   2. At step 0.2.2, on line *** set directory to the FOLDER where the datafile(s) are stored
   3. At step 0.2.3, on line ***, between the quotation marks in 
   ext_labels = loadmat(''), insert the filename of the file that contains the labels 
   for the otherwise unlabeled set (called XValidation in this script).
   This should be a matlab file and the filename should end with .mat
   4. At step 0.2.4, on line ***, after 
   yValidation = 
   call the labels for the otherwise unlabeled set from ext_labels. 
   For example, if the ext_labels is a dictionary (dict) and the key of the labels
   is ctest, then:  yValidation = ext_labels[ctest].
   TO PREVENT ERROR: 
   !-> Make sure that the 3 function files are in the SAME folder as this script 
   (this should be the default in the unzipped folder)
   !-> Make sure that the data file is stored IN A FOLDER that is stored IN the
   the folder that stores the this script. 
   (this should be the default in the unzipped folder)
   !-> Make sure that you have stored the matlab file (.mat) that contains 
   the labels for the otherwise unlabeled set (called XValidation in this script)
   in the same folder that stores Dataset1.mat (in the unzipped folder said 
   subfolder is called 'Data').
   !-> if the file that holds the unkown labels is the same as the name of the
   available datafile (called 'Dataset1.mat'), then it is easiest for this code
   to change the name of the unknown labels file.
# =============================================================================
"""
### 0.2.1 Load the pipelines ###
#sys.path.append(r'')
#EXAMPLE:sys.path.append(r'C:\Users\username\Documents\Machine Learning Algorithms\MSB1011ClassifierKoot')
sys.path.append(r'C:\Users\maart\OneDrive\Academic\MSB1015 - Scientific Programming\Scientific-Programming-22-09-2021')
# load
from GitHub_practice_defModule import statKfold, scale_meanC, scale_pareto, scale_auto

### 0.2.2 set the directory to load the data ###
#os.chdir(r'')
#EXAMPLE: os.chdir(r'C:\Users\username\Documents\Machine Learning Algorithms\MSB1011ClassifierKoot\Data')
os.chdir(r'C:\Users\maart\OneDrive\Academic\MSB1015 - Scientific Programming\Scientific-Programming-22-09-2021')

### 0.2.3 load  the data ###
#data = loadmat('Dataset1.mat') #ML version
X = pd.read_excel(r'Data_tocheck.xlsx', header=None)
# define labels
y = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

#%% 
# =============================================================================
# ##### 1. INSPECT DATA #####
# =============================================================================
### 1.1 check the shape of the expression data and labels ###
print(X.shape, y.shape)
    
### 1.2 Sum labels to check the balance in the distribution of the target classes ###
    # The datalabels are 1 (class1) and -1 (class2).
    # Therefore the closer the sum is to zero, the better the balance between the classes
group_diff = sum(y)

n_class1 = len(X)/2 + group_diff/2   
n_class2 =  len(X)/2 - group_diff/2

print("Data inspection: There are {} less class 1 samples than class 2 samples.".format(-1*group_diff))
print("Data inspection: Thus, {} out of 50 samples are class 1 ({}%), and {} are controls ({}%)".format(n_class1, n_class1*2, n_class2, n_class2*2))
#%% 
# =============================================================================
# ##### 2. DATA PREPERATION ##### 
# =============================================================================
### 2.1 Change the the control labels from -1 to 0 for easier handling of data ###
y = np.where(y!=1, 0, y)
y = y.ravel()


NfoldOuter = 4
# call split funtion to make outer splits
outerSplit = statKfold(X, y, NfoldOuter)
(dictTrainOut, dictTestOut, dictTrainlabelsOut, dictTestlabelsOut) = outerSplit


### 2.3 Inner cross validation splits ###
NfoldInner = 3
dictCVtrainAllinn= {}
dictCVtestAllinn= {}
dictCVtrainAllLabelsinn= {}
dictCVtestAllLabelsinn= {}
dictCVfolds = {}
for Outerfold in range(NfoldOuter):
    # call split function to make inner splits in each outer test set
    innerSplit = statKfold(dictTrainOut[Outerfold], dictTrainlabelsOut[Outerfold], NfoldInner)
    (dictTrainInner, dictTestInner, dictTrainlabelsInner, dictTestlabelsInner) = innerSplit
    dictCVfolds[Outerfold] = innerSplit
    dictCVtrainAllinn[Outerfold]= dictTrainInner
    dictCVtestAllinn[Outerfold]= dictTestInner
    dictCVtrainAllLabelsinn[Outerfold]= dictTrainlabelsInner
    dictCVtestAllLabelsinn[Outerfold]= dictTestlabelsInner

#%%   
# =============================================================================
# ##### 3. APPLY SCALING AND SET UP AND RUN THE MODEL #####
# =============================================================================
### 3.0 initialize classifiers ###
# Support Vector Machine  (aka Support Vector CLassifier)
CLFsvc = SVC(C = 0.001, random_state = 23, probability = True, kernel = 'linear')  
'''Note that this specific type of classiciation and the paramater values/types choosen here 
are not chosen based on a proper rationale (or let alone a grid search, 
because that would be implemented within the inner cross validation fold). 
We just picked something to use for this excercise.'''

dictCVaccMC  = {}
dictCVaccPareto  = {}
dictCVaccAuto  = {}
for outerFold in range(NfoldOuter): 
    nested_scoresMC = list()
    nested_scoresPareto = list()
    nested_scoresAuto = list()
    for innerFold in range(NfoldInner):
        X_train = dictCVfolds[outerFold][0][innerFold]
        X_test = dictCVfolds[outerFold][1][innerFold]
        y_train = dictCVfolds[outerFold][2][innerFold]
        y_test = dictCVfolds[outerFold][3][innerFold]
        
        ### 3.1 inner CV train on mean-centered data ### 
        resultMC = scale_meanC(X_train, X_test)
        (trainMC, testMC) = resultMC
        CLFfit = CLFsvc.fit(trainMC, y_train)
        y_pred = CLFfit.predict(testMC)
        # evaluate the model
        Acc = accuracy_score(y_test, y_pred)
        # store the result
        nested_scoresMC.append(Acc)
        #perfomance = modelfit.best_score_
        
        ### 3.2 inner CV train on pareto scaled data ### 
        resultPareto = scale_pareto(X_train, X_test)
        (trainPareto, testPareto) = resultPareto
        CLFfit = CLFsvc.fit(trainPareto, y_train)
        y_pred = CLFfit.predict(testPareto)
        # evaluate the model
        Acc = accuracy_score(y_test, y_pred)
        # store the result
        nested_scoresPareto.append(Acc)
        ### 3.3 inner CV train on auto-scaled data ### 
        resultAuto = scale_auto(X_train, X_test)
        (trainAuto, testAuto) = resultAuto
        CLFfit = CLFsvc.fit(trainAuto, y_train)
        y_pred = CLFfit.predict(testAuto)
        # evaluate the model
        Acc = accuracy_score(y_test, y_pred)
        # store the result
        nested_scoresAuto.append(Acc)
        
    dictCVaccMC[outerFold] = nested_scoresMC
    dictCVaccPareto[outerFold] = nested_scoresPareto
    dictCVaccAuto[outerFold] = nested_scoresAuto

