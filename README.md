# PredictiveMaintLSTM

Please refer to LSTM_main.py

Following this tutorial: https://towardsdatascience.com/lstm-for-predictive-maintenance-on-pump-sensor-data-b43486eb3210

Takes time series sensor data of 20 sensors and uses LSTM to predict failure (machine_status = broken).

1. Loads data
2. Encodes target variable using one-hot encoding to ensure there is no ranking or importance to the discreet labels.
3. Drops sensors with redundant data (sensors with no more than 2 unique values).
4. Pads the matrix with 0's. LSTM requires equal amount of observations in each time series, but that is not the case in the given data. The first method used is to create the additional observations as 0.
5. Converts data to time series, that is to take given observations and align them to a future target, one time step in the future.
6. Splits data into training and validation sets such that each set contains only complete time series, no time series are split between both sets.

7. Scale
8. Reshape
9. Train
10. Load
11. Predict
12. Evaluate
