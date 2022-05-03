# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:56:46 2022

@author: mw
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import glob



"""
Variables
"""
# Set wd = 'your_working_directory'
wd = r'C:/Users/mw/OneDrive/Desktop/Predictive Maintenance with LSTM Project'
data_path = r'/Data/Durst_data'

d_filtered_hist = pd.read_csv(wd+data_path+'/durst_filtered_label_printhead_history.csv')
p_hist_by_time = pd.read_csv(wd+data_path+'/printhead_history_train_data_by_time.csv')
    

"""
Functions
"""


# LOAD
# Reading data files
#def read_data(wd, data_path):
#    # Returns dataframes of all files in the durst folder
#    file_path = wd+data_path
#    files = glob.glob(file_path+'//*', recursive = True)
#    for file in files:
#        file = pd.read_csv(file)
#    return file



"""
Script
"""
if __name__ == '__main__':

    # LOAD
    # Setting working directory
    os.chdir(wd)
    # File paths for train and test data files
    #read_data(wd, data_path)