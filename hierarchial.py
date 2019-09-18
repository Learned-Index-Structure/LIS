import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from numpy import loadtxt
import tensorflow as tf
import datetime
import math


path = 'utils/newData0.1.csv'

def import_data(path):

    data = np.loadtxt(path, dtype=float, delimiter=' ')
    key = data[:, 1]
    value = data[:, 0]

    key = np.reshape(key, (-1,1))
    value = np.reshape(value, (-1,1))

    print(key.shape)
    print(value.shape)
    return key, value

def train_individual_model(keys, values, epochs=100, batch_size=50,  verbose=0, validation_split=0.2):

    model = Sequential()
    model.add(Dense(32, activation=tf.nn.relu, input_shape=(1,)))
    # model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse' , 'mse'])
    model.fit(keys, values , epochs=epochs, batch_size=batch_size,  verbose=verbose, validation_split=validation_split)

    return model

def train(keys, values , stages, threshold):
    model = []
    m = len(stages)
    n = len(keys)
    print(n)

    tmp_keys = []
    tmp_keys.append([])
    tmp_keys[0].append([])
    tmp_keys[0][0] = keys

    tmp_values = []
    tmp_values.append([])
    tmp_values[0].append([])
    tmp_values[0][0] = values

    for i in range(m):

        print("-"*40)
        print("Optimizing layer - " + str(i+1))
        print("-"*40)

        model.append([])

        layer_keys = []
        layer_values = []

        if (i < m-1):
            tmp_keys.append([])
            tmp_values.append([])
            for q in range(stages[i+1]):
                layer_keys.append([])
                layer_values.append([])

        for j in range(stages[i]):
            print("Training for layer - " + str(i+1) + ", and model - " + str(j+1))
            model[i].append(train_individual_model(tmp_keys[i][j] , tmp_values[i][j]))
            if (i < m-1):

                predictions = np.floor(((model[i][j].predict(tmp_keys[i][j]))*stages[i+1])/n)
                predictions = predictions.astype('int') 

                for p in range(len(tmp_keys[i][j])): 
                    layer_keys[predictions[p][0]].append(tmp_keys[i][j][p][0])
                    layer_values[predictions[p][0]].append(tmp_values[i][j][p][0])

        if (i < m-1):
            for q in range(stages[i+1]):
                tmp_keys[i+1].append(np.reshape(np.asarray(layer_keys[q]), (-1,1)))
                tmp_values[i+1].append(np.reshape(np.asarray(layer_values[q]), (-1,1)))

    return model


def test_model(keys, values, stages, model):
    print("-"*40)
    print("Testing model")
    print("-"*40)
    
    m = len(stages)
    n = len(keys)

    for ind in range(len(keys)):
        if(ind % int((len(keys)/20)))==0:
            start = datetime.datetime.now()
            model_number = 0
            for i in range(m):
                if (i < m-1):
                    model_number = np.floor(((model[i][model_number].predict(keys[ind]))*stages[i+1])/n)
                    model_number = model_number.astype('int')
                    model_number = model_number[0][0]
                else:
                    pred = model[i][model_number].predict(keys[ind])
                    end = datetime.datetime.now()
                    diff = end - start
                    print("key=%s, Original=%s, Predicted=%s, Time=%s" % (keys[ind], values[ind], pred, diff))

stages = [1,5, 10]
threshold = 0

all_keys, all_values = import_data(path)
models = train(all_keys, all_values, stages, 0)
test_model(all_keys, all_values, stages, models)