import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import decimal
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def hiddenLayer(prob):
    n1 = n2 = n3 = 2048.0 / prob
    n = [n1, n2, n3, prob]
    return n

def arch(prob):
    n1 = n2 = n3 = 2048
    n = [n1, n2, n3, 1 - prob]
    return n

def dropoutLayer(layer, input_shape, x_train, y_train):
    model = Sequential([
        Conv2D(28, kernel_size=(3, 3), input_shape=input_shape, activation="relu"),
        Flatten(),
        Dense(layer[0], activation=tf.nn.relu),
        Dropout(layer[3]),
        Dense(layer[1], activation=tf.nn.relu),
        Dropout(layer[3]),
        Dense(layer[2], activation=tf.nn.relu),
        Dropout(layer[3]),
        Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    dropoutHistory = model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=10, batch_size=32)

    accuracy = dropoutHistory.history['accuracy']
    trainError = 100.0 - 100.0 * accuracy[-1]

    valAccuracy = dropoutHistory.history['val_accuracy']
    testError = 100.0 - 100.0 * valAccuracy[-1]

    finalError = [testError, trainError, layer[3]]
    return finalError

def pRange(start, stop, step):
    start_dec = decimal.Decimal(str(start))
    stop_dec = decimal.Decimal(str(stop))
    step_dec = decimal.Decimal(step)
    return (float(start_dec + i * step_dec) for i in range(int((stop_dec - start_dec) / step_dec) + 1))

def allPRun():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    input_shape = (28, 28, 1)

    test_error, train_error = [], []

    probValues = list(pRange(0, 0.9, '0.1'))[1:] + [0.99]

    for i in probValues:
        constLayer = arch(i)

        errorFig = dropoutLayer(constLayer, input_shape, x_train, y_train)
        test_error.append(errorFig[0])
        train_error.append(errorFig[1])
    
    print("test error: ", test_error)
    print("train error: ", train_error)

allPRun()
