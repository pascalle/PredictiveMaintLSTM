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
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt

# Set wd = 'your_working_directory'
wd = 'C:/Users/mw/OneDrive/Desktop/Predictive Maintenance with LSTM Project'

# Set target Variable
# 'machine_status' or 'RUL' (not set up for RUL yet)
target_variable = 'machine_status'  # 'RUL'
# Set timesteps either default = 1 or = split_segment
# Set percent train (.75 or .8)


"""
Functions
"""


# LOAD
# Reading data files
def read_data(path):
    dat = pd.read_csv(path)
    # senorname=pd.Series(dat.keys()[2:-1])
    senorname = dat.keys()[6:-2]
    return dat, senorname


# ENCODE
# Encoding the target data classifications - machine_status
# https://github.com/JanderHungrige/PumpSensor/blob/main/Sensor_analysis.py
def Vorverarbeitung_Y(dat):
    # from sklearn import preprocessing
    # 1: Label Mapping
    le = preprocessing.LabelEncoder()
    le.fit(dat)
    encoded_y = le.transform(dat)
    # 2: Get the Label map
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    return pd.DataFrame(encoded_y, columns=['hot_target'])


# DROP
# Formatting data by dropping unnecessary columns
def col_drop(df, droptime=False):
    # 1: Drop the op columns
    newdata = df.drop(df.filter(regex='op').columns, axis=1)
    # 2: Remove sensors where unique values is less than 2 (or some number)
    nunique = newdata.nunique()
    cols_to_drop = nunique[nunique < 2].index
    cols_to_drop = cols_to_drop.values
    if droptime == True:
        cols_to_drop = np.append(cols_to_drop, ["Unnamed: 0(t-1)", "id(t-1)", "t(t-1)"])
    # can drop sensor 5 too
    newdata = newdata.drop(cols_to_drop, axis=1)
    return newdata


def time_drop(df):
    # 1: Need to remove certain columns from the time series created data and split the x and y variables
    data_y = df.iloc[:, -1]  # Get the target data out before removing unwanted data
    data_y = data_y.astype(str)
    # 2: Remove sensors(t), and target
    to_remove_list = df.loc[:, df.columns.str.endswith(
        '(t)')].columns  # remove all non shifted elements again. so we retreive elements and shifted target
    data_x = df.drop(to_remove_list, axis=1)  # remove sensors(t)
    data_x.drop(data_x.columns[len(data_x.columns) - 1], axis=1, inplace=True)  # remove target(t-n)
    return data_x, data_y


# TIMESERIES
# Prepare data to be time series data
# Adds columns for t-n and t+n time lagged variables
# Starting with using one lagged variable (all sensors at t-1) to predict machine_status at t
# Can move on to using more lagged variables to predict a sequence like t through t+5
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, namen = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        namen += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            namen += [(df.columns[j] + '(t)') for j in range(n_vars)]
        else:
            namen += [(df.columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = namen
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# PAD
# Padding the dataframe with 0's so all id's(timeseries) have same amount of observations
def pad0(df, col):
    # 1: Create a new index, indexing by id (so id 1 and all rows, id 2 and all rows)
    df = df.set_index([col, df.groupby(col).cumcount()])
    # 2: Adds indices so that each index is the same size
    index = pd.MultiIndex.from_product(df.index.levels, names=df.index.names)
    # 3: Applies changes to original data frame and pads with 0's
    output = df.reindex(index, fill_value=0).reset_index(level=1, drop=True).reset_index()
    # 4: Check that each timeseries has the same number of rows
    freq = output.groupby([col])[col].count()
    if freq.nunique() != 1:
        print("Error with padding")
    split_segment = freq.iloc[0]
    return output, split_segment


# SPLIT
# Splitting training data into training and validation sets
# Ensuring a each set has complete (and not split) timeseries data
# each padded timeseries will have "split_segment" number of variables
# For now, train(75%): 75 groups@ 362 rows= [0,27150]obs and validate(25%): 25groups@ 362rows= [27150,]
# In future, can do k-fold cross-validation with each fold containing 362obs
# https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
def split(data_x, data_y, split_segment, percent_train=.75):
    i = int(split_segment * (100 * percent_train))

    x_train = data_x[0:i].values
    x_val = data_x[i::].values

    y_train = data_y[0:i].values
    y_val = data_y[i::].values

    x_train.astype('float32')
    x_val.astype('float32')

    return x_train, x_val, y_train, y_val


# ONEHOT ENCODING
# Encoding the target variable for training the model
def one_hot(train_Y, val_Y):
    from sklearn.preprocessing import OneHotEncoder
    oneHot = OneHotEncoder()
    oneHot.fit(train_Y.reshape(-1, 1))

    train_Y_Hot = oneHot.transform(train_Y.reshape(-1, 1)).toarray()
    val_Y_Hot = oneHot.transform(val_Y.reshape(-1, 1)).toarray()
    return train_Y_Hot, val_Y_Hot


# SCALE
# Scaling the sets between 0-1
# Using MinMaxScalar so the padding of zeros does not affect the scale
def scaling(data):
    # from sklearn.preprocessing import MinMaxScaler
    scaler = preprocessing.MinMaxScaler().fit(data)
    scaled_features = scaler.transform(data)
    return scaled_features


# RESHAPE
# Reshaping the data to be used for LSTM [batch, timesteps, features]
# Batch size: number of samples used per iteration before performing the weight update
# In this case, a batch size could be the total number of ids, the timesteps are the amount of observations within
# each id, and the features are the number of columns (sensors plus RUL)
def reshape_for_Lstm(data, timesteps=1):
    samples = int(np.floor(data.shape[0] / timesteps))  # batch
    data = data.reshape((samples, timesteps, data.shape[1]))  # batch, timesteps, sensors (features)
    return data


# TRAIN
def model_setup_Fapi(in_shape):
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    units = 75  # originally 42

    inputs = tf.keras.Input(shape=(in_shape[1], in_shape[2]))
    x = LSTM(units, activation='relu', input_shape=(in_shape[1], in_shape[2]), return_sequences=True)(inputs)
    x = LSTM(units, activation='relu', input_shape=(in_shape[1], in_shape[2]), return_sequences=True)(x)
    out_signal = Dense(1, name='signal_out')(x)
    out_class = Dense(4, activation='softmax', name='class_out')(x)  # Use softmax for classification problems

    model = tf.keras.Model(inputs=inputs, outputs=[out_signal, out_class])

    model.compile(loss={'signal_out': 'mean_squared_error',
                        'class_out': 'categorical_crossentropy'},
                  optimizer='adam',
                  metrics={'class_out': 'acc'})

    print(model.summary())

    return model


# PLOT
# Plotting results from the validation set
def plot_training(history, what='loss', saving=False, name='training'):
    fig = plt.figure()
    plt.plot(history[0])
    plt.plot(history[1])
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    if what == 'loss':
        plt.title('model loss')
        plt.ylabel('loss')
    elif what == 'acc':
        plt.title('model Acc')
        plt.ylabel('Accuracy')
    if saving == True:
        fig.savefig(name + '_' + what + '.png', format='png', dpi=300, transparent=True)

    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    if saving == True:
        fig.savefig(name + '_ACC.png', format='png', dpi=300, transparent=True)
    plt.show()


"""
Script
"""
if __name__ == '__main__':

    # LOAD
    # Setting working directory
    sys.path.insert(0, wd + '/PumpSensors')  # this allows me to pull in functions from the PumpSensor github
    os.chdir(wd)
    # File paths for train and test data files
    dp_train = "Data/Train_labeled_FD001.txt"
    dp_test = "Data/Test_FD001.txt"
    # Reading datasets for train and test data files, and sensor label files
    data_train, sensorname_train = read_data(dp_train)
    data_test, sensorname_test = read_data(dp_test)

    # ENCODE
    # Encoding the target data classifications - machine_status
    # Uses one-hot encoding so there is no ranking or importance in the discreet class labels
    # Outputs vector labeled Target
    ###encoded_y = Vorverarbeitung_Y(data_train["machine_status"])
    # add encoded y value to the original
    ###encoded_train =pd.concat([data_train,encoded_y],axis=1)#.reindex(data.index)

    # DROP1
    # Formatting by dropping sensors with unchanging data
    drop_train = col_drop(data_train)  # encoded_train

    # TIMESERIES
    # Default n_in = 1 , n_out = 1 (gives t-1 and t)
    train_series = series_to_supervised(drop_train)

    # PAD
    # Pad each timeseries wiht 0's so they all have the same length
    # Using split segment to ensure training and validation sets are split so they
    pad_train, split_segment = pad0(train_series, "id(t-1)")
    # print(pad_train.columns)
    # print(split_segment)

    # DROP2
    # Need to remove certain columns from the time series created data and split the x and y variables
    data_x, data_y = time_drop(pad_train)
    '''data_y is the non-encoded y'''
    # Dropping columns unnecessary for Training (timeseries columns, non-encoded target)
    data_x = col_drop(data_x, True)
    # print(drop2_train.columns)

    # SPLIT
    # Splitting training data into training and validation sets
    # Ensuring a each set has complete timeseries data
    # each padded timeseries will have an equal number of variables (print("split_segment"))
    x_train, x_val, y_train, y_val = split(data_x, data_y, split_segment)

    # SCALE
    # Scaling the sets between 0-1
    x_train = scaling(x_train)
    x_val = scaling(x_val)

    # ENCODING
    # ONEHOT ENCODING
    # Encoding padded 0's as a different category which should be ok because they always come after a broken status
    y_train_hot, y_val_hot = one_hot(y_train, y_val)
    # Changing Y class to numeric classes
    y_train = Vorverarbeitung_Y(y_train).to_numpy()
    y_val = Vorverarbeitung_Y(y_val).to_numpy()

    # RESHAPE
    '''Try timesteps = 1 (default), or split_segment'''
    timesteps = split_segment
    train_X = reshape_for_Lstm(x_train, timesteps)
    val_X = reshape_for_Lstm(x_val, timesteps)

    # Don't reshape the Y values
    #samples=int(np.floor(y_train.shape[0]/timesteps)) #batch
    #y_train=y_train.flatten()
    #y_train=y_train.reshape((samples,1))
    #samples=int(np.floor(y_val.shape[0]/timesteps)) #batch
    #y_val=y_val.reshape((samples,1))
    #samples=int(np.floor(y_train_hot.shape[0]/timesteps)) #batch
    #y_train_hot=y_train_hot.reshape((samples,1))
    #samples=int(np.floor(y_val_hot.shape[0]/timesteps)) #batch
    #y_val_hot=y_val_hot.reshape((samples,1))

    y_train = reshape_for_Lstm(y_train, timesteps)
    y_val = reshape_for_Lstm(y_val, timesteps)

    y_train_hot = reshape_for_Lstm(y_train_hot, timesteps)
    y_val_hot = reshape_for_Lstm(y_val_hot, timesteps)
    '''When timesteps = split_segment, need to reshape the y as well
    Also should make this a part of a function to automate the choice of timesteps'''

    # TRAIN
    inputshape_X = (train_X.shape)
    batch_X = train_X.shape[0]
    model = model_setup_Fapi(inputshape_X)

    # for model.fit in the y_train, he origianlly has the code as [train_Y, train_Y_Hot], [val_Y,val_Y_Hot], but I'm going to start with it
    # as just using the encoded y (which for now is y_train, y_val)

    history = model.fit(train_X, [y_train, y_train_hot], epochs=20, batch_size=32,
                        validation_data=(val_X, [y_val, y_val_hot]), shuffle=False)


    Future = 1
    plot_training([history.history['class_out_loss'], history.history['val_class_out_loss']],
                  what='loss',
                  saving=True,
                  name=('training_' + str(Future)))
    plot_training([history.history['class_out_acc'], history.history['val_class_out_acc']],
                  what='acc',
                  saving=True,
                  name=('training_' + str(Future)))

'''Need to update the saving information based on the input from RESHAPE'''
'''
    # SAVE
    if timesteps == 1 | timesteps == None:
        model.save('./model/Pump_LSTM_timestep_1_' + str(Future))
    else:
        model.save('./model/Pump_LSTM_timestep_' + str(timesteps) + '_' + str(Future))

    # LOAD
    if timesteps == 1 | timesteps == None:
        model = tf.keras.models.load_model('./model/Pump_LSTM_timestep_1_' + str(Future))
    else:
        model = tf.keras.models.load_model('./model/Pump_LSTM_timestep_' + str(timesteps) + '_' + str(Future))
'''
    # PREDICT
    # EVALUATE

    # How to deal with irregular timesteps
    # Why are the timesteps irregular? Missing data, only counts to a break, etc
    # Main options to deal with this:
    # Pad 'missing' values with 0's
    # Pad missing values with last value (broken, etc) https://www.tutorialspoint.com/python_pandas/python_pandas_reindexing.htm
    # Encode the data 1 with being available, and 0 for missing rows
    # Other options, but riskier
    # Feed in with one batch at a time (window size of 1)
    # Extrapolate missing values

    # https://towardsdatascience.com/predictive-maintenance-with-lstm-siamese-network-51ee7df29767
    # file:///C:/Users/mw/Downloads/sensors-21-00972-v2.pdf
    # Using an autoencoder
    # https://stats.stackexchange.com/questions/312609/rnn-for-irregular-time-intervals
    # https://dl.acm.org/doi/pdf/10.1145/3097983.3097997
    # Pad the shorter sequences with zeros, Here's an example of how to do it: https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py#L46
    # Another solution would be to feed sequences to your model one sequence at a time (batch_size=1). Then differences in sequence lengths would be irrelevant.
