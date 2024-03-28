import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from CNN_Model import CNNModel
from CNN_LSTM_model import CNN_LSTM
from CNN_Transformer_Model import CNN_Transformer

def train_model(data_dir, chkpt_dir, model_name):
    assert (model_name == 'CNN' or model_name == 'CNN_LSTM' or model_name == 'CNN_Transformer'), 'Invalid Model Name'
    loss = 'categorical_crossentropy'
    if model_name == 'CNN':
        model = CNNModel()
    elif model_name == 'CNN_LSTM':
        model = CNN_LSTM()
    else:
        model = CNN_Transformer()
        loss = 'sparse_categorical_crossentropy'
    
    EPOCHS = 20 # Unsure of correct param

    # Set up model
    sgd = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
    
    # Get data from saved csv
    trainX = np.load(os.path.join(data_dir, 'train_X.npy'))
    trainY = np.load(os.path.join(data_dir, 'train_y.npy'))

    # Set up checkpoints
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(chkpt_dir, 'model.ckpt'),
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=True
    )

    # Train
    model.fit(trainX, trainY, epochs = EPOCHS, batch_size=64, callbacks=[model_checkpoint_callback], verbose =2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required = True)
    parser.add_argument("--chkpt_dir", type=str, required = True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_dir, args.chkpt_dir, args.model_name)