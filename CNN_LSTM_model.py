import tensorflow as tf
from tensorflow import keras
from CNN_Model import CNNModel

class CNN_LSTM(CNNModel):
    def __init__(self, num_cells=128, dropout=0.3, **kwargs):
        super().__init__(dropout=dropout, **kwargs)
        self.rnn_layer = tf.keras.layers.RNN(keras.layers.LSTMCell(num_cells))
    
    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        hidden1 = self.hidden1(input_layer)
        hidden1 = self.dropout(hidden1, training= training)
        hidden2 = self.hidden2(hidden1)
        hidden2 = self.hidden2(hidden2)
        hidden2 = self.hidden2(hidden2)
        final_CNN_layer = self.output_layer(hidden2)
        final_CNN_layer = self.dropout(final_CNN_layer, training = training)
        output_layer = self.rnn_layer(final_CNN_layer)
        output_layer = self.dropout(output_layer, training = training)
        output_layer = self.dense2(output_layer)
        return output_layer