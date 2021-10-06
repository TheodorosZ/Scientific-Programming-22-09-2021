# -*- coding: utf-8 -*-
"""
This function accompanies the classification script:
    SCRIPT_MSB1011_classification_Koot_I6084689
"""
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

def train_bestClfpipe(clf, traindata, labels, unlabeled_data, Kfeat):
    scaler = StandardScaler().fit(traindata)
    X_train = scaler.transform(traindata)
    X_predict = scaler.transform(unlabeled_data) 
    
    ## select K best features
    feat_select = SelectKBest(k = Kfeat)
    featurefit = feat_select.fit(X_train, labels.ravel())
    mask = featurefit.get_support(indices=True)
    X_traindf = pd.DataFrame(X_train)
    features_select = X_traindf.iloc[:,mask]
    X_trainK = featurefit.transform(X_train)

    ## train the model 
    CLFfit= clf.fit(X_trainK, labels.ravel())

# =============================================================================
# #### apply the model to predict labels of unlabeled data ####
# =============================================================================
    # Select features in test set, based on train set fit
    X_predictK = featurefit.transform(X_predict)
    # predict labels
    ypred = CLFfit.predict(X_predictK)
    probs = CLFfit.predict_proba(X_predictK)[:,1]
    return ypred, probs, mask, features_select