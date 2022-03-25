# -*- coding: utf-8 -*-
"""
Predictive Maintenance using Sensor Data and LSTM
https://towardsdatascience.com/lstm-for-predictive-maintenance-on-pump-sensor-data-b43486eb3210

@author: Pascalle Banaszek
"""

"""Imports and set variables
"""
import sys
import os
import pandas as pd
import numpy as np


#Set wd = 'your_working_directory'
wd = 'C:/Users/mw/OneDrive/Desktop/Predictive Maintenance with LSTM Project'


#Set target Variable
#'machine_status' or 'RUL' (not set up for RUL yet)
target_variable = 'machine_status' #'RUL'



"""
Functions
"""

#LOAD
#Reading data files
def read_data(path):
    dat=pd.read_csv(path)
    #senorname=pd.Series(dat.keys()[2:-1])
    senorname=dat.keys()[6:-2]
    return dat, senorname


#ENCODE
#Encoding the target data classifications - machine_status
#https://github.com/JanderHungrige/PumpSensor/blob/main/Sensor_analysis.py
    def Vorverarbeitung_Y(dat):
        from sklearn import preprocessing
#1: Label Mapping
        le = preprocessing.LabelEncoder()
        le.fit(dat)
        encoded_y=le.transform(dat)
#2: Get the Label map
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)
        return pd.DataFrame(encoded_y,columns=['target'])


#DROP
#Formatting data by dropping unnecessary columns
def col_drop(df, droptime = False):
#1: Drop the op columns
    newdata = df.drop(df.filter(regex='op').columns, axis=1)
#2: Remove sensors where unique values is less than 2 (or some number)
    nunique = newdata.nunique()
    cols_to_drop = nunique[nunique < 2].index
    cols_to_drop = cols_to_drop.values
    if droptime == True:
        cols_to_drop = np.append(cols_to_drop, ["Unnamed: 0","id", "t", "machine_status"])
    #can drop sensor 5 too
    newdata = newdata.drop(cols_to_drop, axis=1)
    return newdata

def time_drop(df):
#1: Need to remove certain columns from the time series created data and split the x and y variables
    data_y=df.iloc[:,-1] #Get the target data out before removing unwanted data
#2: Remove sensors(t), and target
    to_remove_list = df.loc[:, df.columns.str.endswith('(t)')].columns #remove all non shifted elements again. so we retreive elements and shifted target  
    data_x=df.drop(to_remove_list, axis=1) #remove sensors(t) 
    data_x.drop(data_x.columns[len(data_x.columns)-1], axis=1, inplace=True)# remove target(t-n)
    return data_x, data_y


#PAD
#Padding the dataframe with 0's so all id's(timeseries) have same amount of observations
def pad0(df, col):
#1: Create a new index, indexing by id (so id 1 and all rows, id 2 and all rows)
    df = df.set_index([col, df.groupby(col).cumcount()])
#2: Adds indices so that each index is the same size
    index = pd.MultiIndex.from_product(df.index.levels, names=df.index.names)
#3: Applies changes to original data frame and pads with 0's
    output = df.reindex(index, fill_value=0).reset_index(level=1, drop=True).reset_index()
#4: Check that each timeseries has the same number of rows    
    freq = output.groupby(['id'])['id'].count()
    if freq.nunique() != 1:
        print("Error with padding")
    split_segment = freq.iloc[0]
    return output, split_segment


#TIMESERIES
#Prepare data to be time series data
#Adds columns for t-n and t+n time lagged variables
    #Starting with using one lagged variable (all sensors at t-1) to predict machine_status at t
    #Can move on to using more lagged variables to predict a sequence like t through t+5
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, namen = list(),list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        namen +=[(df.columns[j]+'(t-%d)' %(i)) for j in range (n_vars)]
        #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            namen +=[(df.columns[j]+'(t)') for j in range (n_vars)]
        else:
            namen +=[(df.columns[j]+'(t+%d)' %(i)) for j in range (n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns=namen
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


#SPLIT
#Splitting training data into training and validation sets
    #Ensuring a each set has complete (and not split) timeseries data    
    #each padded timeseries will have "split_segment" number of variables
#For now, train(75%): 75 groups@ 362 rows= [0,27150]obs and validate(25%): 25groups@ 362rows= [27150,]
#In future, can do k-fold cross-validation with each fold containing 362obs
#https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
def split(data_x, data_y, split_segment, percent_train = .75):
    i = int(split_segment*(100*percent_train))
    #y = int(split_segment*(100*(1-percent_train)))
    print(i)
    
    x_train = data_x.iloc[0:i,:]
    x_val = data_x.iloc[i:,:]
    y_train = data_y[0:i]
    y_val = data_y[i:]

    return x_train, x_val, y_train, y_val



"""
Script
"""
if __name__ == '__main__': 
    
    #LOAD
    #Setting working directory
    sys.path.insert(0, wd + '/PumpSensors') #this allows me to pull in functions from the PumpSensor github
    os.chdir(wd)
    #File paths for train and test data files
    dp_train = "Data/Train_labeled_FD001.txt"
    dp_test = "Data/Test_FD001.txt"
    #Reading datasets for train and test data files, and sensor label files
    data_train, sensorname_train = read_data(dp_train)
    data_test, sensorname_test = read_data(dp_test)
    
    
    #ENCODE
    #Encoding the target data classifications - machine_status
    #Uses one-hot encoding so there is no ranking or importance in the discreet class labels
    #Outputs vector labeled Target
    encoded_y = Vorverarbeitung_Y(data_train["machine_status"])
    #add encoded y value to the original
    encoded_train =pd.concat([data_train,encoded_y],axis=1)#.reindex(data.index)
    
    #DROP1
    #Formatting by dropping sensors with unchanging data
    drop_train = col_drop(encoded_train)

  
    #PAD
    #Pad each timeseries wiht 0's so they all have the same length
    #Using split segment to ensure training and validation sets are split so they 
    pad_train, split_segment = pad0(drop_train, "id")
    #print(pad_train.columns)
    #print(split_segment)


    #DROP2
    #Dropping columns unnecessary for Training (timeseries columns, non-encoded target)
    drop2_train = col_drop(pad_train, True)    
    #print(drop2_train.columns)
    
    
    #TIMESERIES
    #Default n_in = 1 , n_out = 1 (gives t-1 and t)
    train_series = series_to_supervised(drop2_train)
    
    
    #DROP3
    #Need to remove certain columns from the time series created data and split the x and y variables
    data_x, data_y = time_drop(train_series)
    
    
    #SPLIT
    #Splitting training data into training and validation sets
        #Ensuring a each set has complete timeseries data
        #each padded timeseries will have an equal number of variables (print("split_segment"))           
    x_train, x_val, y_train, y_val = split(data_x, data_y, split_segment)
        

      
    
    #SCALE
    
    
    
    
    
    
    #RESHAPE
    #TRAIN
    #LOAD
    #PREDICT
    #EVALUATE
    
    
    
    




  
   

    
    
    
    #How to deal with irregular timesteps
    #Why are the timesteps irregular? Missing data, only counts to a break, etc
    #Main options to deal with this:
        #Pad 'missing' values with 0's
        #Pad missing values with last value (broken, etc) https://www.tutorialspoint.com/python_pandas/python_pandas_reindexing.htm
        #Encode the data 1 with being available, and 0 for missing rows
    #Other options, but riskier
        #Feed in with one batch at a time (window size of 1)
        #Extrapolate missing values
    
    #https://towardsdatascience.com/predictive-maintenance-with-lstm-siamese-network-51ee7df29767
    #file:///C:/Users/mw/Downloads/sensors-21-00972-v2.pdf
    #Using an autoencoder
    #https://stats.stackexchange.com/questions/312609/rnn-for-irregular-time-intervals
    #https://dl.acm.org/doi/pdf/10.1145/3097983.3097997
    #Pad the shorter sequences with zeros, Here's an example of how to do it: https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py#L46
    #Another solution would be to feed sequences to your model one sequence at a time (batch_size=1). Then differences in sequence lengths would be irrelevant.
