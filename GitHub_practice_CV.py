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
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC

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
from GitHub_practice_defModule import scale_meanC, scale_pareto, scale_auto
from GitHub_practice_nestedCV_KfeatKoot import Clf_pipeKbest

### 0.2.2 set the directory to load the data ###
#os.chdir(r'')
#EXAMPLE: os.chdir(r'C:\Users\username\Documents\Machine Learning Algorithms\MSB1011ClassifierKoot\Data')
os.chdir(r'C:\Users\maart\OneDrive\Academic\MSB1015 - Scientific Programming\Scientific-Programming-22-09-2021')

### 0.2.3 load  the data ###
#data = loadmat('Dataset1.mat') #ML version
X = pd.read_excel(r'Data_tocheck.xlsx')
# define labels
y = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

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
n_class1 = 100 + group_diff/2   
n_class2 =  100 - group_diff/2
print("Data inspection: There are ", *group_diff, " more class 1 samples than class 2 samples.")
print("Data inspection: Thus,", *n_class1, " out of 50 samples are patients (", *n_class2/2, "%), and ", *n_class2, " are controls (", *n_class2/2, "%)")
#%% 
# =============================================================================
# ##### 2. DATA PREPERATION ##### 
# =============================================================================
### 2.1 Change the the control labels from -1 to 0 for easier handling of data ###
y = np.where(y!=1, 0, y)
y = y.ravel()
#%%
# =============================================================================
# ##### 3 MODEL PARAMETER AND FEATURE SELECTION #####
# =============================================================================
### 3.1 initialize classifiers ###
# seperately for each pipeline
CLFsvc =   SVC(random_state = 23, probability = True, kernel = 'linear')  
CLFlr = LogisticRegression(random_state = 23, n_jobs = -1)

# =============================================================================
# #### 3.2 Classifier pipeline with hyperparameter C optimization ####    
# =============================================================================
## 3.2.1 initialize cross-validation K folds ##
cv_outer =  StratifiedKFold(n_splits=5, shuffle=True, random_state=23) # returns 5 stratified folds
cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=23) # returns 4 stratified folds

## 3.2.2 initialize paramater grid to search for the best regulatization parameter C
param_svc = {'C': [.0001, .001, .005, .01, .1, .5, 1],

             } 
param_lr = {'C': [.0001, .001, .005, .01, .1, .5, 1]
            }
#%%
### 3.2.3 Run nested CV pipelines for all features ###
results_SVC = Clf_pipeKbest(Clf = CLFsvc, 
                            ParaGrid = param_svc, 
                            Refit = 'Accuracy', # consider another classification perfomance parameter, because the data is not balanced
                            CV_outer = cv_outer, 
                            CV_inner = cv_inner, 
                            X = X, 
                            y = y, 
                            Kfeat = 'all') 

results_LR = Clf_pipeKbest(Clf = CLFlr, 
                           ParaGrid = param_lr, 
                           Refit = 'Accuracy', # consider another classification perfomance parameter, because the data is not balanced
                           CV_outer = cv_outer, 
                           CV_inner = cv_inner, 
                           X = X, y = y, 
                           Kfeat = 'all') 
#%%
'''
## =============================================================================
## #### 4. NESTED CROSS-VALIDATION RESULTS ####
## =============================================================================

# store accuracy scores in variables
accSVC = [results_SVC400[1], results_SVC300[1], results_SVC200[1], results_SVC100[1]]
accLR = [results_LR400[1], results_LR300[1], results_LR200[1], results_LR100[1]]

# create variables with feature number to identify barcharts
titles = ['SVC nested CV accuracy all features', 
          'SVC nested CV accuracy 300 best features', 
          'SVC nested CV accuracy 200 best features', 
          'SVC nested CV accuracy 100 best features', 
          'Log regression nested CV accuracy all features', 
          'Log regression nested CV accuracy 300 best features', 
          'Log regression nested CV accuracy 200 best features', 
          'Log regression nested CV accuracy 100 best features']
fold = ['fold 1','fold 2','fold 3','fold 4','fold 5']


# set dictionary for barplot text style 
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

# loop over CV accuracy scores for different # best features
figSVCcv = plt.figure(figsize=(20,10))
for i in range(4):
    score = accSVC[i]
    mean  = np.mean(score)
    sd = np.std(score)
    sns.set(font_scale = 1.3)
    plt.subplot(2,2,i+1)
    ax = sns.barplot(fold, score)
    ax.set_title(titles[i], fontsize=20)
    ax.set_ylabel('outer accuracy for best model', fontsize=20)
    ax.set_ylim(0,1)
    ax.text(0.9, 0.9, str('mean accuracy: %.3f (SD=%.3f)' % (mean, sd)), fontdict=font)
plt.show()

figSVCcv.savefig('barplot_SVC_nestedCV_accuracy.png')  

figLRcv = plt.figure(figsize=(20,10))
for i in range(4):
    score = accLR[i]
    mean  = np.mean(score)
    sd = np.std(score)
    plt.subplot(2,2,i+1)
    ax = sns.barplot(fold, score)
    ax.set_title(titles[i+4], fontsize=20)
    ax.set_ylabel('outer accuracy for best model', fontsize=20)
    ax.set_ylim(0,1)
    ax.text(0.9, 0.9, str('mean accuracy: %.3f (SD=%.3f)' % (mean, sd)), fontdict=font)
plt.show()

figLRcv.savefig('barplot_LR_nestedCV_accuracy.png')  

# The mean accuracy is the highest for SVC on the 200 best features
# On 3 out of 5 folds the value for the best C = 0.01
print('SVC on 200 best features with C=0.01 is chosen as the best model.')
#%%
# =============================================================================
# ##### 5. FIT THE BEST MODEL TO FULL OUTER SET #####
# =============================================================================

# 5.1 initialize best model 
svc_best =  SVC(random_state = 23, probability = True, kernel = 'linear', C = 0.01)  

# 5.2 set the stratified K-fold
cv_validate = cv_outer
# =============================================================================
# #### 5.3 run the model through validation pipeline ####
# =============================================================================
result_bestCLF = BestModel_pipe(best_clf = svc_best, cv = cv_validate, X = X, y = y, Kfeat = 200)

### 5.4 vizualize results ###
score = result_bestCLF[0]
meanfit  = np.mean(score)
sdfit = np.std(score)
print('The mean accuracy for the best model fit is %.3f (SD=%.3f)' % (meanfit, sdfit))
print('Thus, the accuracy is exprected to lay around 0.725, the')
figBESTCLF = plt.figure(figsize=(20,10))
ax = sns.barplot(fold, score)
ax.set_title('Accuracy best model outer fit (SVC, C=.01, Kbest = 200)', fontsize=20)
ax.set_ylabel('accuracy for best model', fontsize=20)
ax.set_ylim(0,1)
ax.text(0.9, 0.9, str('mean accuracy: %.3f (SD=%.3f)' % (meanfit, sdfit)), fontdict=font)
plt.show()

figBESTCLF.savefig('barplot_SVC_best_model_on_outer_5fold.png')  
#%% 
# =============================================================================
# ##### 6. APPLY MODEL TO UNLABELED DATA  #####
# =============================================================================
"""
Note this step is redundant when the label for the otherwise unlabeled validation data are available 
"""
# 6.1 initialize best model 
trainCLF =  SVC(random_state = 23, probability = True, kernel = 'linear', C = 0.01)  
# 6.2 fit the best model with the final pipeline 
resultFIT = train_bestClfpipe(clf = trainCLF,
                              traindata = X, 
                              labels = y, 
                              unlabeled_data = XValidation, 
                              Kfeat = 200)

#%%
# =============================================================================
# ##### 7. LOAD LABELS FOR THE (UNTIL NOW) UNLABELED DATA AND APPLY AGAIN #####
# =============================================================================
yValidation = np.random.randint(2,size = 1000)

# 7.1 initialize best model 
finalCLF =  SVC(random_state = 23, probability = True, kernel = 'linear', C = 0.01)
finalResults = final_predictionClf(final_clf = finalCLF, 
                                   traindata = X, 
                                   trainlabels = y, 
                                   preddata = XValidation, 
                                   predlabels = yValidation, 
                                   Kfeat = 200)

print('Accuracy of the classifier on the final validation data =%.3f' % (finalResults['accuracy']))

'''