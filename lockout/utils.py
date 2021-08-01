# ==================================================================================================
# Author: Wilmer Arbelo-Gonzalez
# 
# This module provides general python tools
# --------------------------------------------------------------------------------------------------
import os
import pickle
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ==================================================================================================
def read_list(file_path):
    """
    Input:
    - file_path: path to file (str or path)
    
    Output:
    - itemlist: list read from file (python list)
    """
    itemlist = None
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            itemlist = pickle.load(file)
    return itemlist


# ==================================================================================================
def save_list(itemlist, file_path):
    """
    Input:
    - itemlist:  list to be saved (python list)
    - file_path: path to file (str or path)
    
    Output:
    - list saved as a pickle file
    """
    with open(file_path, 'wb') as file:
        pickle.dump(itemlist, file)

 
# ==================================================================================================
def save_data(X, xtrain, xvalid, xtest, Y, ytrain, yvalid, ytest, folder):
    """
    Input:
    - (X, xtrain, xvalid, xtest, Y, ytrain, yvalid, ytest) as Pandas 
      DataFrames
    - folder: where data is to be saved (str or path)
    
    Output:
    - Data saved as the following .cvs files:
      'X.csv', 'xtrain.csv', 'xvalid.csv', 'xtest.csv'
      'Y.csv', 'ytrain.csv', 'yvalid.csv', 'ytest.csv'
    """
# Save X, xtrain, xvalid, and xtest
    X.to_csv(os.path.join(folder, 'X.csv'), 
             index=False, header=False)
    xtrain.to_csv(os.path.join(folder, 'xtrain.csv'), 
                  index=False, header=False)
    xvalid.to_csv(os.path.join(folder, 'xvalid.csv'), 
                  index=False, header=False)
    xtest.to_csv(os.path.join(folder, 'xtest.csv'), 
                 index=False, header=False)
    
# Save Y, ytrain, yvalid, and ytest
    Y.to_csv(os.path.join(folder, 'Y.csv'), 
             index=False, header=False)
    ytrain.to_csv(os.path.join(folder, 'ytrain.csv'), 
                  index=False, header=False)
    yvalid.to_csv(os.path.join(folder, 'yvalid.csv'), 
                  index=False, header=False)
    ytest.to_csv(os.path.join(folder, 'ytest.csv'), 
                 index=False, header=False)


# ==================================================================================================
def split_data(dfX, dfy, seed1=None, seed2=None, test_size1=0.2, test_size2=0.25, 
               stratify=False):
    """
    Input:
    -Predictors (dfX) and targets (dfy)
    -Integer seeds for the data split
    -Test sizes for the data split
    
    Output:
    -Dataframes with training, validation, and testing subsets: 
     xtrain, xvalid, xtest, ytrain, yvalid, ytest
    """
    if stratify == False:
        stratify1 = None
    else:
        stratify1 = dfy
    xtrain_valid, x_test, ytrain_valid, y_test = train_test_split(dfX, dfy, 
                                                 test_size=test_size1, 
                                                 random_state=seed1,
                                                 stratify=stratify1)

    if stratify == False:
        stratify2 = None
    else:
        stratify2 = ytrain_valid
    x_train, x_valid, y_train, y_valid = train_test_split(xtrain_valid, 
                                                          ytrain_valid, 
                                                          test_size=test_size2, 
                                                          random_state=seed2, 
                                                          stratify=stratify2)
    return x_train, x_valid, x_test, y_train, y_valid, y_test
