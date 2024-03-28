import argparse
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from CNN_Model import CNNModel
from CNN_LSTM_model import CNN_LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

def get_CNN_model(k1_size, pool1_size, conv_size, learnRate):
    model = CNNModel(k1_size, pool1_size, conv_size)
    sgd = keras.optimizers.SGD(learning_rate=learnRate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def get_CNN_LSTM(num_cells, dropout, learnRate):
    model = CNN_LSTM(num_cells, dropout)
    sgd = keras.optimizers.SGD(learning_rate=learnRate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def tune_params(model_name, data_dir):
    assert (model_name == 'CNN' or model_name == 'CNN_LSTM'), "Invalid Model Name"

    # Get data from saved csv
    trainX = np.load(os.path.join(data_dir, 'train_X.npy'))
    trainY = np.load(os.path.join(data_dir, 'train_y.npy'))

    # Define model and param grid
    if model_name == 'CNN':
        build_fn = get_CNN_model
        k1_size = [25, 50, 75]
        pool1_size = [4, 8, 16]
        conv_size = [4, 8, 16]
        learnRate = [1e-2, 1e-3, 1e-4]
        batchSize=[64]
        epochs=[20]

        grid = {
            "k1_size": k1_size,
            "pool1_size": pool1_size,
            "conv_size": conv_size,
            "learnRate": learnRate,
            "batch_size": batchSize,
            "epochs": epochs
        }
    else:
        build_fn = get_CNN_LSTM
        num_cells = [64, 128, 256]
        dropout = [0.3, 0.4, 0.5]
        learnRate = [1e-2, 1e-3, 1e-4]
        batchSize=[64]
        epochs=[20]

        grid = {
            "num_cells": num_cells,
            "dropout": dropout,
            "learnRate": learnRate,
            "batch_size": batchSize,
            "epochs": epochs
        }
    
    # search grid
    model = KerasClassifier(build_fn=build_fn, verbose = 2)
    searcher = RandomizedSearchCV(estimator=model, cv=5, param_distributions=grid, scoring='accuracy')
    searchResults = searcher.fit(trainX, np.argmax(trainY, axis=1))

    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("Best score is {:.2f} using {}".format(bestScore,	bestParams))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required = True)
    args = parser.parse_args()

    tune_params(args.model_name, args.data_dir)