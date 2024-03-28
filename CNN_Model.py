import tensorflow as tf
from tensorflow import keras

# CNN based model based on the representation learning part of the TinySleepNet model
# Default param values are based on tuning where we acheived 82.34% test accuracy

class CNNModel(keras.Model):
    def __init__(self,k1_size=25, pool1_size=16, conv_size=16, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=k1_size, strides=k1_size//8, activation='relu', padding='same', input_shape=(3000, 1))
        self.hidden1 = tf.keras.layers.MaxPooling1D(pool_size=pool1_size, strides =pool1_size, padding="same")
        self.hidden2 = tf.keras.layers.Conv1D(filters=128, kernel_size=conv_size, strides=1, activation='relu')
        self.output_layer = tf.keras.layers.MaxPooling1D(pool_size=pool1_size//2, strides=pool1_size//2, padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.dense1 = tf.keras.layers.Dense(700)
        self.dense2 = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        hidden1 = self.hidden1(input_layer)
        hidden1 = self.dropout(hidden1, training= training)
        hidden2 = self.hidden2(hidden1)
        hidden2 = self.hidden2(hidden2)
        hidden2 = self.hidden2(hidden2)
        output_layer = self.output_layer(hidden2)
        output_layer = self.flatten(output_layer)
        output_layer = self.dropout(output_layer, training=training)
        output_layer = self.dense1(output_layer)
        output_layer = self.dense2(output_layer)
        return output_layer