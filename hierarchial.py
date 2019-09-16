import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from numpy import loadtxt
import tensorflow as tf
import datetime

path = 'utils/newData0.1.csv'

def import_data(path):

    data = np.loadtxt(path, dtype=float, delimiter=' ')
    key = data[:, 1]
    value = data[:, 0]

    key = np.reshape(key, (-1,1))
    value = np.reshape(value, (-1,1))

    #Normalization
    scalar_key = MinMaxScaler()
    scalar_value = MinMaxScaler()

    scalar_key.fit(key)
    scalar_value.fit(value)

    scale_key = scalar_key.transform(key)
    scale_value = scalar_value.transform(value)

    return scale_key, scale_value

def train_individual_model(keys, values):

    model = Sequential()
    model.add(Dense(32, activation=tf.nn.relu, input_shape=(1,)))
    # model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse' , 'mse'])
    model.fit(keys, values , epochs=100, batch_size=50,  verbose=0, validation_split=0.2)

    return model

def train(keys, values , stages, threshold):
    model = []
    m = len(stages)
    n = len(keys[0,:])
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
        tmp_keys.append([])
        tmp_values.append([])
        for j in range(stages[i]):
            print("Calculating for layer - " + str(i+1) + ", and stage - " + str(j+1))
            model[i].append(train_individual_model(tmp_keys[i][j] , tmp_values[i][j]))
            if (i < m-1):
                for q in range(stages[i+1]):
                    tmp_keys[i+1].append([])
                    tmp_values[i+1].append([])
                for p in range(len(tmp_keys[i][j])): 
                    pred = (model[i][j].predict(tmp_keys[i][j][p]))*stages[i+1]
                    tmp_keys[i+1][int(pred)].append(tmp_keys[i][j][p][0])
                    tmp_values[i+1][int(pred)].append(tmp_values[i][j][p][0])

    return model


def test_model(keys, values, stages, model):
    print("-"*40)
    print("Testing model")
    print("-"*40)
    
    m = len(stages)

    for ind in range(len(keys)):
        if(ind % int((len(keys)/20)))==0:
            start = datetime.datetime.now()
            model_number = 0 
            for i in range(m):
                if (i < m-1):
                    model_number = int(model[i][model_number].predict(keys[ind]*stages[i+1]))
                else:
                    pred = model[i][model_number].predict(keys[ind])
                    end = datetime.datetime.now()
                    diff = end - start
                    print("key=%s, Original=%s, Predicted=%s, Time=%s" % (keys[ind], values[ind], pred, diff))

stages = [1,10]
threshold = 0

all_keys, all_values = import_data(path)
models = train(all_keys, all_values, stages, 0)
test_model(all_keys, all_values, stages, models)