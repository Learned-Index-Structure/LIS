import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from numpy import loadtxt
import tensorflow as tf
import datetime
import math
from concurrent.futures import ThreadPoolExecutor


path = 'data/sample.txt'

def import_data(path):

    data = np.loadtxt(path, dtype=float, delimiter=' ')
    key = data[:, 1]
    value = data[:, 0]

    key = np.reshape(key, (-1,1))
    value = np.reshape(value, (-1,1))

    print(key.shape)
    print(value.shape)
    return key, value

def train_individual_model(layerNumber, modelNumber, keys, values, epochs=100, batch_size=32,  verbose=0, validation_split=0.1):
    print("Training model(" + str(layerNumber) + ", " + str(modelNumber) + ")")
    #TODO: create sessions equal to max threads in the thread pool
    sess = tf.compat.v1.Session()
    with sess.as_default():
        if layerNumber == 0:
            model = Sequential()
            model.add(Dense(32, activation=tf.nn.relu, input_shape=(1,)))
            model.add(Dense(32, activation=tf.nn.relu))
            model.add(Dense(1))
            model.compile(optimizer='RMSprop', loss='mse', metrics=['mse' , 'mse'])
            model.fit(keys, values , epochs=epochs, batch_size=batch_size,  verbose=verbose, validation_split=validation_split)
        else:
            model = Sequential()
            # model.add(Dense(32, activation=tf.nn.relu, input_shape=(1,)))
            #model.add(Dense(32, activation=tf.nn.relu))
            model.add(Dense(1))
            model.compile(optimizer='RMSprop', loss='mse', metrics=['mse' , 'mse'])
            model.fit(keys, values , epochs=200, batch_size=8,  verbose=verbose, validation_split=0.2)


    # model._make_predict_function()
    print("Finished training model(" + str(layerNumber) + ", " + str(modelNumber) + ")")
    return (model, (sess, tf.compat.v1.get_default_graph()))

def train(keys, values , stages, threshold):
    tf.debugging.set_log_device_placement(True)
    model = []
    #TODO: change variable name
    graphs = []
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

    executor = ThreadPoolExecutor(max_workers=1)

    for i in range(m):

        print("-"*40)
        print("Optimizing layer - " + str(i))
        print("-"*40)

        model.append(stages[i]*[None])
        graphs.append(stages[i]*[None])

        layer_keys = []
        layer_values = []

        if (i < m-1):
            tmp_keys.append([])
            tmp_values.append([])
            for q in range(stages[i+1]):
                layer_keys.append([])
                layer_values.append([])

        futures = []

        for j in range(stages[i]):
            if i > 0:
                futures.append(executor.submit(train_individual_model, i, j, tmp_keys[i][j] , tmp_values[i][j]))
            else:
                start_train = datetime.datetime.now()
                model[i][j],graphs[i][j] = train_individual_model(i, j, tmp_keys[i][j] , tmp_values[i][j])
                end_train = datetime.datetime.now()
                print("Model for layer 0 trained. Time taken = " + str(end_train - start_train))

            if (i < m-1):
                with graphs[i][j][0].as_default():
                    predictions = np.floor(((model[i][j].predict(tmp_keys[i][j]))*stages[i+1])/n)
                    predictions = predictions.astype('int')
                    predictions[predictions >= stages[i+1]] = stages[i+1]-1
                    predictions[predictions < 0] = 0

                    for p in range(len(tmp_keys[i][j])):
                        layer_keys[predictions[p][0]].append(tmp_keys[i][j][p][0])
                        layer_values[predictions[p][0]].append(tmp_values[i][j][p][0])

        if (i < m-1):
            for q in range(stages[i+1]):
                tmp_keys[i+1].append(np.reshape(np.asarray(layer_keys[q]), (-1,1)))
                tmp_values[i+1].append(np.reshape(np.asarray(layer_values[q]), (-1,1)))

        if i > 0:
            print("\nWaiting for models to get trained...")
            start_train = datetime.datetime.now()
            for j in range(len(futures)):
                model[i][j],graphs[i][j] = futures[j].result()
            end_train = datetime.datetime.now()
            print("All models for layer " + str(i) + " trained. Time taken = " + str(end_train - start_train))

    return (model,graphs)


def test_model(graphs, keys, values, stages, model):
    print("-"*40)
    print("Testing model")
    print("-"*40)
    
    m = len(stages)
    n = len(keys)

    toterror = 0
    t = 0

    for ind in range(len(keys)):
        if(ind % int((len(keys)/20)))==0:
            t+=1
            start = datetime.datetime.now()
            model_number = 0
            for i in range(m):
                g = graphs[i][model_number][0]
                with g.as_default():
                    with graphs[i][model_number][1].as_default():
                        if (i < m-1):
                            model_number = np.floor(((model[i][model_number].predict(keys[ind]))*stages[i+1])/n)
                            model_number = model_number.astype('int')
                            model_number = model_number[0][0]
                        else:
                            pred = model[i][model_number].predict(keys[ind])
                            toterror += abs(values[ind] - pred)
                            end = datetime.datetime.now()
                            diff = end - start
                            print("key=%s, Original=%s, Predicted=%s, Time=%s" % (keys[ind], values[ind], pred, diff))

    print("\nTotal error in the prediction = " + str(toterror))
    print("Average error = " + str(toterror/t))

stages = [1,10]
threshold = 0

all_keys, all_values = import_data(path)
models,graphs = train(all_keys, all_values, stages, 0)
test_model(graphs, all_keys, all_values, stages, models)