import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Enhanced encoder with 5 convolutional blocks as described
        # Each block: Conv3x3 + BatchNorm + ReLU + MaxPool
        self.conv1 = layers.Conv2D(64, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2, 2)  # 224→112
        
        self.conv2 = layers.Conv2D(128, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2, 2)  # 112→56
        
        self.conv3 = layers.Conv2D(256, 3, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2, 2)  # 56→28
        
        self.conv4 = layers.Conv2D(512, 3, padding='same')
        self.bn4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(2, 2)  # 28→14
        
        self.conv5 = layers.Conv2D(512, 3, padding='same')
        self.bn5 = layers.BatchNormalization()
        self.pool5 = layers.MaxPooling2D(2, 2)  # 14→7
        
        # Latent representation compressed to 128 dimensions via adaptive global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.latent_dense = layers.Dense(128, activation='relu')
        self.latent_bn = layers.BatchNormalization()
        
        # Reshape for decoder
        self.reshape = layers.Reshape((7, 7, 128))
        
        # Decoder with transposed convolutions and skip connections
        self.upconv1 = layers.Conv2DTranspose(512, 3, strides=2, padding='same')
        self.bn_up1 = layers.BatchNormalization()
        
        self.upconv2 = layers.Conv2DTranspose(256, 3, strides=2, padding='same')
        self.bn_up2 = layers.BatchNormalization()
        
        self.upconv3 = layers.Conv2DTranspose(128, 3, strides=2, padding='same')
        self.bn_up3 = layers.BatchNormalization()
        
        self.upconv4 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')
        self.bn_up4 = layers.BatchNormalization()
        
        self.upconv5 = layers.Conv2DTranspose(32, 3, strides=2, padding='same')
        self.bn_up5 = layers.BatchNormalization()
        
        # Final output layer
        self.output_conv = layers.Conv2D(3, 3, padding='same', activation='sigmoid')
        
        # Skip connection layers for residual connections
        self.skip_conv1 = layers.Conv2D(32, 1, padding='same')
        self.skip_conv2 = layers.Conv2D(64, 1, padding='same')
        self.skip_conv3 = layers.Conv2D(128, 1, padding='same')
        self.skip_conv4 = layers.Conv2D(256, 1, padding='same')

    def call(self, x, training=None):
        # Encoder with skip connection storage
        # Block 1
        e1 = self.conv1(x)
        e1 = self.bn1(e1, training=training)
        e1 = tf.nn.relu(e1)
        e1_pooled = self.pool1(e1)  # 224→112
        
        # Block 2
        e2 = self.conv2(e1_pooled)
        e2 = self.bn2(e2, training=training)
        e2 = tf.nn.relu(e2)
        e2_pooled = self.pool2(e2)  # 112→56
        
        # Block 3
        e3 = self.conv3(e2_pooled)
        e3 = self.bn3(e3, training=training)
        e3 = tf.nn.relu(e3)
        e3_pooled = self.pool3(e3)  # 56→28
        
        # Block 4
        e4 = self.conv4(e3_pooled)
        e4 = self.bn4(e4, training=training)
        e4 = tf.nn.relu(e4)
        e4_pooled = self.pool4(e4)  # 28→14
        
        # Block 5
        e5 = self.conv5(e4_pooled)
        e5 = self.bn5(e5, training=training)
        e5 = tf.nn.relu(e5)
        e5_pooled = self.pool5(e5)  # 14→7
        
        # Latent representation (128 dimensions)
        latent = self.global_avg_pool(e5_pooled)
        latent = self.latent_dense(latent)
        latent = self.latent_bn(latent, training=training)
        
        # Reshape for decoder
        # We need to expand the 128-dim latent to spatial dimensions
        # Use learned dense layers to create 7x7x128 feature map
        latent_spatial = tf.expand_dims(tf.expand_dims(latent, 1), 1)  # [B, 1, 1, 128]
        latent_spatial = tf.tile(latent_spatial, [1, 7, 7, 1])  # [B, 7, 7, 128]
        
        # Decoder with skip connections
        # Decoder block 1: 7x7 → 14x14
        d1 = self.upconv1(latent_spatial)
        d1 = self.bn_up1(d1, training=training)
        d1 = tf.nn.relu(d1)
        # Skip connection from e4 (14x14x512 → 14x14x512)
        d1 = d1 + e4_pooled
        
        # Decoder block 2: 14x14 → 28x28
        d2 = self.upconv2(d1)
        d2 = self.bn_up2(d2, training=training)
        d2 = tf.nn.relu(d2)
        # Skip connection from e3 (28x28x256)
        skip4 = self.skip_conv4(e3_pooled)
        d2 = d2 + skip4
        
        # Decoder block 3: 28x28 → 56x56
        d3 = self.upconv3(d2)
        d3 = self.bn_up3(d3, training=training)
        d3 = tf.nn.relu(d3)
        # Skip connection from e2 (56x56x128)
        skip3 = self.skip_conv3(e2_pooled)
        d3 = d3 + skip3
        
        # Decoder block 4: 56x56 → 112x112
        d4 = self.upconv4(d3)
        d4 = self.bn_up4(d4, training=training)
        d4 = tf.nn.relu(d4)
        # Skip connection from e1 (112x112x64)
        skip2 = self.skip_conv2(e1_pooled)
        d4 = d4 + skip2
        
        # Decoder block 5: 112x112 → 224x224
        d5 = self.upconv5(d4)
        d5 = self.bn_up5(d5, training=training)
        d5 = tf.nn.relu(d5)
        # Skip connection from input features (224x224x32)
        skip1 = self.skip_conv1(e1)
        # Resize skip1 to match d5 if needed
        skip1_resized = tf.image.resize(skip1, tf.shape(d5)[1:3])
        d5 = d5 + skip1_resized
        
        # Final output
        output = self.output_conv(d5)
        
        return output