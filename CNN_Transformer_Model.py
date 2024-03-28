from typing_extensions import dataclass_transform
import tensorflow as tf
from tensorflow import keras

class CNN_Transformer(keras.Model):
    def __init__(self, head_size=256, num_heads=4, ff_dim=4, dropout=0.4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units=mlp_units
        #CNN Layers
        self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=6, activation='relu', padding='same',input_shape=(3000,1))
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=8, strides =8, padding="same")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4, padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(500)
        self.reshape = tf.keras.layers.Reshape((500,1))

        # Normalization and Attention
        self.layernormalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.multiheadatten = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        # Feed Forward
        self.conv3 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')

        # Transformer
        self.globalavgpool = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')
        self.mlpdropout = tf.keras.layers.Dropout(mlp_dropout)

        # Output
        self.output_layer = tf.keras.layers.Dense(5, activation='softmax')


    def call(self, input, training=None):
        x = input
        for _ in range(self.num_transformer_blocks):
            shape = x.shape[-1]
            x = self.layernormalization(x)
            x = self.multiheadatten(x, x)
            x = self.dropout2(x, training=training)
            res = x + input

            x = self.layernormalization(res)
            x = self.conv3(x)
            x = self.dropout2(x, training = training)
            x = tf.keras.layers.Conv1D(filters=shape, kernel_size=1)(x)
            x = x + res
        x = self.globalavgpool(x)
        for dim in self.mlp_units:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
            x = self.mlpdropout(x, training=training)
        output = self.output_layer(x)
        return output



        

