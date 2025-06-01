import tensorflow as tf
from tensorflow.keras import layers

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder: 3× Conv → ReLU → AvgPool
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(128, kernel_size=4, padding='same', activation='relu',
                          input_shape=(224, 224, 3)),
            layers.AveragePooling2D(pool_size=2, strides=2),    # 224→112
            layers.Conv2D(256, kernel_size=4, padding='same', activation='relu'),
            layers.AveragePooling2D(pool_size=2, strides=2),    # 112→56
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.AveragePooling2D(pool_size=2, strides=2),    #  56→28
        ])
        # Decoder: 3× ConvTranspose → ReLU (last: Sigmoid)
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),                                      #  28→56
            layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'),
            layers.ReLU(),                                      #  56→112
            layers.Conv2DTranspose(3,   kernel_size=5, strides=2, padding='same',
                                   activation='sigmoid'),       # 112→224
        ])

    def call(self, x):
        x = self.encoder(x)
        return self.decoder(x)