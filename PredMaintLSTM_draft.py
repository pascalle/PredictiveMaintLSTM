# -*- coding: utf-8 -*-
"""
Predictive Maintenance using Sensor Data and LSTM
https://towardsdatascience.com/lstm-for-predictive-maintenance-on-pump-sensor-data-b43486eb3210

@author: Pascalle Banaszek
"""

import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#import matplotlib.pyplot as plt
#from sklearn import preprocessing
# adding Folder_2 to the system path
sys.path.insert(0, 'C:/Users/mw/OneDrive/Desktop/Predictive Maintenance with LSTM Project/PumpSensors')
import Sensor_learning as psl
import Sensor_analysis as psa


'''Setting working directory'''
#wd = 'your_working_directory'
wd = 'C:/Users/mw/OneDrive/Desktop/Predictive Maintenance with LSTM Project'
os.chdir(wd)

'''Set target Variable'''
#'machine_status' or 'RUL' (not set up for RUL yet)
target_variable = 'machine_status' #'RUL'

#File paths for train and test data files
dp_train = "Data/Train_labeled_FD001.txt"
dp_test = "Data/Test_FD001.txt"


'''Reading and Exploring Data'''
#Reading train and test data files
def read_data(path):
    dat=pd.read_csv(path)
    #senorname=pd.Series(dat.keys()[2:-1])
    senorname=dat.keys()[6:-2]
    return dat, senorname

#Datasets for train and test data files, and sensor label files
data_train, sensorname_train = read_data(dp_train)
data_test, sensorname_test = read_data(dp_test)
#print(sensorname_train)
#print(sensorname_test)
print(data_train.shape)
print(data_train.keys())
print(data_train.head())
print(data_train.head(1))
print(data_train[["id","t","sensor_1", "machine_status"]])

xdt = data_train[["id","t","sensor_1", "machine_status"]]
print(xdt[xdt["id"]==1]) 
xdt = xdt[xdt["id"]==1]
print(xdt.head(1))



xdt.plot(x="t", y = "machine_status")


sensor_1 = data_train[["id","t","sensor_1"]]
sensor_1 = sensor_1[(sensor_1["id"] == 1)] # & (sensor_1["t"]  <= 20)] #sensor_1[sensor_1["id"] <= 5]
print(sensor_1)
sensor_1.plot(x ='t', y='sensor_1', c= "id", cmap= "viridis", kind = 'scatter')	#c= "t"


#Discovering that there are 3 variables for machine status: [normal (9631), recovering(8000), broken(3000)]
#print( 'status options: ');  print( data_train['machine_status'].unique()); print() # Get the unique values for class
#print (data_train['machine_status'].value_counts()); print() # Count the Classes to see how many we got from each
    
#Finding number of NA and null values by column
#No missing data except for test target data (machine status and RUL)
#print(data_train.isna().sum(axis=0))
#print(data_train.isnull().sum(axis=0))
#print(data_test.isna().sum(axis=0))
#print(data_test.isnull().sum(axis=0))


#Encoding the target variable (first machine_status, later RUL)
#To be discreet labels instead of text
#Using code from PumpSensor file Sensor Analysis
'''made with only training data for now'''
encoded_y = psa.Vorverarbeitung_Y(data_train[target_variable]) 
Values=pd.concat([data_train[sensorname_train],encoded_y],axis=1)#.reindex(data.index)
sensorname=Values.keys()[:-1]


#Plotting the target data
psa.plot_Y(encoded_y,col='target', saving=True , name='Klassen')


'''Adding Lag (windowed/shifted) time variables'''
Future=1

data_win = psl.series_to_supervised(Values, n_in=Future, n_out=1)
to_remove_list =['sensor'+str(n)+'(t)' for n in range(1,len(Values.columns)+1)] #now remove all non shifted elements again. so we retreive elements and shifted target
#to_remove_list_2 =['sensor'+str(n)+'(t-'+ str(i)+')' for n in range(1,len(data_scaled.columns)+1) for i in range(1,Future)] #now remove all non shifted elements again. so we retreive elements and shifted target
#to_remove_list=to_remove_list_1+to_remove_list_2
data_y=data_win.iloc[:,-1] #Get the target data out before removing unwanted data
data_x=data_win.drop(to_remove_list, axis=1) #remove sensors(t)
data_x.drop(data_x.columns[len(data_x.columns)-1], axis=1, inplace=True)# remove target(t-n)

    
'''Splitting and formatting training, validation, and test sets'''
#Splitting dataset for cross validation
train_X, test_X, train_Y, test_Y = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
train_X, val_X, train_Y, val_Y   = train_test_split(train_X, train_Y, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

#One hot encoding for the target variable
#Encoding the discreet variables for the class of the target so it doesn't get confused
#With rank or importance
def one_hot(train_Y,val_Y,test_Y):    
    
    oneHot=OneHotEncoder()
    oneHot.fit(train_Y.values.reshape(-1,1))
    
    train_Y_Hot=oneHot.transform(train_Y.values.reshape(-1,1)).toarray()
    val_Y_Hot  =oneHot.transform(val_Y.values.reshape(-1,1)).toarray()
    test_Y_Hot =oneHot.transform(test_Y.values.reshape(-1,1)).toarray()
    
    return train_Y_Hot,val_Y_Hot,test_Y_Hot

train_Y_Hot,val_Y_Hot,test_Y_Hot = one_hot(train_Y,val_Y,test_Y)


#Scaling the data between 0 and 1
def scaling(data):
    scaler=MinMaxScaler().fit(data) 
    scaled_features=scaler.transform(data) 
    return scaled_features
    
train_X = scaling(train_X)
test_X  = scaling(test_X)
val_X  = scaling(val_X)

    
    
'''Training the LSTM model'''

#Had to update the in_shape indices so created new function here
def model_setup_seq(in_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    
    model = Sequential()
    model.add(LSTM(32,activation='relu', input_shape=(None,in_shape[1]), 
                   return_sequences=True)  )#,
                   # kernel_regularizer=tf.keras.regularizers.L1L2(0.01,0.01)))
    #model.add(Dropout(0.3))
    model.add(LSTM(32,activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


#Had to update the in_shape indices so created new function here
def model_setup_Fapi(in_shape):
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    
    inputs= tf.keras.Input(shape=(None,in_shape[1]))
    x=LSTM(42,activation='relu', input_shape=(None,in_shape[1]),return_sequences=True)(inputs)
    x=LSTM(42,activation='relu')(x)
    out_signal=Dense(1, name='signal_out')(x)
    out_class=Dense(3,activation='softmax', name='class_out')(x)
    
    model=tf.keras.Model(inputs=inputs, outputs=[out_signal,out_class])
    
    model.compile(loss={'signal_out':'mean_squared_error',
                        'class_out' :'categorical_crossentropy'},
                         optimizer='adam',
                         metrics={'class_out':'acc'})
    
    print(model.summary())
    return model




Train=True
inputshape_X=(train_X.shape)
print(inputshape_X)

'''Getting error here but I can't figure it out
#Need to update the input to be the right size
#ValueError: Input 0 is incompatible with layer model_4: expected shape=(None, 12378, 21), found shape=(None, 21)
#I thought I fixed the shape to work with the new data (shape=(in_shape[0],in_shape[1]) instead of [1] and [2]) but it's not working'
'''

if Train==True:
    #model=model_setup_seq(inputshape_X)
    #history = model.fit(train_X, train_Y, epochs=80, batch_size=32, validation_data=(val_X, val_Y), shuffle=False)

    model=model_setup_Fapi(inputshape_X)
    history = model.fit(train_X, [train_Y, train_Y_Hot], epochs=20, batch_size=32, validation_data=(val_X, [val_Y,val_Y_Hot]), shuffle=False)
    psl.plot_training([history.history['class_out_loss'],history.history['val_class_out_loss']],
                  what='loss',
                  saving=True,
                  name=('training_'+ str(Future)))  
    psl.plot_training([history.history['class_out_acc'],history.history['val_class_out_acc']],
                  what='acc',
                  saving=True,
                  name=('training_'+ str(Future))) 
    model.save('./model/Pump_LSTM_Fapi_4_'+ str(Future))
    
# ...OR LOAD THE MODELl  
else:  
    model=tf.keras.models.load_model('./model/Pump_LSTM_Fapi')







'''Trying something here'''

inputshape_X=(train_X.shape)
model=model_setup_Fapi(inputshape_X)
history = model.fit(train_X, [train_Y, train_Y_Hot], epochs=70, batch_size=32, validation_data=(val_X, [val_Y,val_Y_Hot]), shuffle=False)
plot_training(history,saving=True,name=('training'+ str(Future)))  
model.save('./model/Pump_LSTM_Fapi_2_'+ str(Future))


print(train_X[0])


