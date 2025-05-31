import tensorflow as tf
from tensorflow.keras import layers

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # … (encoder + decoder from before) …
    def call(self, x):
        x = self.encoder(x)
        return self.decoder(x)