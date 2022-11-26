import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling3D, Flatten
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv1D, Conv3D
import matplotlib.pyplot as plt
import matplotlib as mpl

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    if isinstance(data, list):
        n_vars = 1
    else:
        n_vars = data.shape[1]
    if isinstance(data, pd.DataFrame):
        pass
    else:
        data = pd.DataFrame(data)
    cols, names = list(), list()
    print(n_vars)
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        print(i)
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    #cols_to_use = names[:len(names) - (n_out)]
    #agg = agg[cols_to_use]
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def Min_max_scal(data):
	array = data.values
	values_ = array.astype('float32')
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled = scaler.fit_transform(values_)
	return scaled

def reshape_data_single_lag(reframed, train_percentage, test_percentage, valid_percentage):
	# split into train and test sets
	values = reframed.values
	# Sizes
	train_size = int(len(reframed) * train_percentage)
	test_size = int(len(reframed) * test_percentage)
	valid_size = int(len(reframed) * valid_percentage)

	train = values[:train_size]
	test = values[train_size:train_size + test_size]
	val = values[train_size + test_size:]

	# split into input and outputs
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]
	val_X, val_y = val[:, :-1], val[:, -1]
	# print(train_X.shape)

	### this reshape below is we using it for univariate timeseries
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, val_X.shape, val_y.shape)

	return train_X, train_y, test_X, test_y, val_X, val_y



def LSTM_model(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    model.add(LSTM(90, return_sequences = True,  input_shape=(train_X.shape[1], train_X.shape[2])))  # 1 , 2
    model.add(Dropout(0.75))
    model.add(LSTM(70, return_sequences = True,  input_shape=(train_X.shape[1], train_X.shape[2])))  # 1 , 2
    model.add(Dropout(0.75))
    model.add(LSTM(50, return_sequences = True,  input_shape=(train_X.shape[1], train_X.shape[2])))  # 1 , 2
    model.add(Dropout(0.50))
    model.add(LSTM(30, return_sequences = True,  input_shape=(train_X.shape[1], train_X.shape[2])))  # 1 , 2
    model.add(Dropout(0.50))
    model.add(LSTM(10, return_sequences = False,  input_shape=(train_X.shape[1], train_X.shape[2])))  # 1 , 2
    model.add(Dropout(0.10))
    #model.add(LSTM(15, return_sequences = False ))
    #model.add(Dropout(0.5))
    #model.add(LSTM(30, return_sequences = False ))
    #model.add(Dropout(0.2))
    #model.add(LSTM(15, return_sequences = False ))
    #model.add(Dense(50))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_X, train_y, epochs=25, batch_size=8, validation_data=(test_X, test_y),verbose=2, shuffle=False)
    return model

def plot_train_test_loss(model):
    mpl.rcParams['figure.figsize'] = (12, 8)
    mpl.rcParams['axes.grid'] = False
    # plot history
    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='test')
    plt.legend()
    plt.show()