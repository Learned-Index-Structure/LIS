#!/usr/bin/python3

import tensorflow as tf
from keras.layers import Dense
import numpy as np

def train_individual_model(keys, values):
    model = tf.keras.Sequential()
    model.add(Dense(32, activation=tf.nn.relu, input_shape=(1,)))
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dense(1))
    model.fit(keys, values)
    return model

def train(all_data, stages, threshold):
    model = []
    m = len(stages)
    n = len(all_data[0,:])
    for i in range(m):
        model.append([])
        for j in range(stages[i]):
            start = (n/stages[i])*j
            end = (n/stages[i])*(j+1)
            model[i].append(train_individual_model(all_data[0, start:end], all_data[1, start:end]))

    #TODO: Do error checking and put B-tree models where error is higher than threshold