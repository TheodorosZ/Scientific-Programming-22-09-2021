# -*- coding: utf-8 -*-
"""
This function accompanies the classification script:
    SCRIPT_MSB1011_classification_Koot_I6084689
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest

def BestModel_pipe(best_clf, cv, X, y, Kfeat):
    nested_scores = list()
    for train_ix, test_ix in cv.split(X, y):
        # split data 
        X_train_unscaled, X_test_unscaled = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # standardize data
        scaler = StandardScaler().fit(X_train_unscaled)
        X_train = scaler.transform(X_train_unscaled)
        X_test = scaler.transform(X_test_unscaled) 
        
        ## select K best features
        feat_select = SelectKBest(k = Kfeat)
        featurefit = feat_select.fit(X_train, y_train.ravel())
        mask = featurefit.get_support(indices=True)
        X_traindf = pd.DataFrame(X_train)
        features_select = X_traindf.iloc[:,mask]
        X_trainK = featurefit.transform(X_train)

        ## train the model 
        CLFfit= best_clf.fit(X_trainK, y_train.ravel())
        
# =============================================================================
#         ## Evaluate model on the hold out dataset ##
# =============================================================================
        # Select features in test set, based on train set fit
        X_testK = featurefit.transform(X_test)
        # predict test labels
        ypred = CLFfit.predict(X_testK)
        probs = CLFfit.predict_proba(X_testK)[:,1]
        # evaluate the model
        acc = accuracy_score(y_test, ypred)
        # store the result
        nested_scores.append(acc)

    return nested_scores, ypred, probs, mask, features_select