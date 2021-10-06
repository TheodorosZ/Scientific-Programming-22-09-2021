# -*- coding: utf-8 -*-
"""
This function accompanies the classification script:
    [...]
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.feature_selection import SelectKBest

import sys 
sys.path.append(r'C:\Users\maart\OneDrive\Academic\MSB1015 - Scientific Programming\Scientific-Programming-22-09-2021')
# from GitHub_practice_defModuleTEST import scale_meanC, scale_pareto, scale_auto
from GitHub_practice_defModuleTEST import defScale

def Clf_pipeKbest(Clf, ParaGrid, CV_outer, CV_inner, Refit, X, y, scaletype):
    nested_accscores = list()
    i = 1
    dict_resultALL = {}
    for train_ix, test_ix in CV_outer.split(X, y):
        # split data 
        X_train_unscaled, X_test_unscaled = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        
        ## parameter optimization ##
        multi_score = {'AUC': 'roc_auc',
                       'Accuracy': 'accuracy'}
        # define search
        search_cv = GridSearchCV(estimator=Clf, param_grid=ParaGrid, scoring=multi_score, n_jobs=-1, cv=CV_inner, refit = Refit, return_train_score = True)
        # execute search
        search_result = search_cv.fit(X_train_unscaled, y_train.ravel())
        # results of inner cross validation
        inner_results = search_result.cv_results_
        # get the best performing model fit on the whole training set
        best_model = search_result.best_estimator_
        # best hyperparameters
        best_param = search_result.best_params_
        # best inner accuracy scores 
        best_score = search_result.best_score_
        
        #store results of the model selection
        dict_result = dict(inner_result = inner_results, bestparameters = best_param, bestscore = best_score,  bestmodel = best_model,  trainsamples = train_ix)

# =============================================================================
#         ## Evaluate model on the hold out dataset ##
# =============================================================================

        # standardize data
        scaler = defScale(X_train_unscaled, X_test_unscaled, scaletype)
        # unpack output tuple
        (scaled_Xtrain, scaled_Xtest) = scaler
        
        # predict test labels
        ypred = best_model.predict(scaled_Xtest)
        probs = best_model.predict_proba(scaled_Xtest)[:,1]
        # evaluate the model
        acc = accuracy_score(y_test, ypred)
        # store the result
        nested_accscores.append(acc)
        # save results
        dict_resultALL['fold' + str(i)] = dict_result
        i = i + 1 
    locals().update(dict_resultALL)
    return dict_resultALL, nested_accscores, ypred, probs

