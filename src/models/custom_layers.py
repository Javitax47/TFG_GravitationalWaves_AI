# src/models/custom_layers.py

import tensorflow as tf
from tensorflow.keras import layers

# ==============================================================================
# CAPAS PARA CNN 2D (ResNet)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut
        if self.conv_shortcut:
            self.shortcut_conv = layers.Conv2D(filters, 1, strides=stride, kernel_initializer='he_normal',
                                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            self.shortcut_bn = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2 = layers.BatchNormalization()
        
        self.add = layers.Add()
        self.out_relu = layers.Activation('relu')

    def call(self, inputs):
        shortcut = inputs
        if self.conv_shortcut:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.add([shortcut, x])
        x = self.out_relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "conv_shortcut": self.conv_shortcut,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class RecomputeGradLayer(layers.Layer):
    def __init__(self, recomputable_layer, **kwargs):
        super().__init__(**kwargs)
        self.recomputable_layer = recomputable_layer

    def call(self, inputs):
        return tf.recompute_grad(self.recomputable_layer)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "recomputable_layer": tf.keras.layers.serialize(self.recomputable_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        recomputable_layer_config = config.pop("recomputable_layer")
        recomputable_layer = tf.keras.layers.deserialize(recomputable_layer_config)
        return cls(recomputable_layer=recomputable_layer, **config)

# ==============================================================================
# CAPAS PARA VISION TRANSFORMER (ViT)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

# ==============================================================================
# CAPAS PARA 1D CNN BAYESIANA
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class MCDropout(layers.Dropout):
    """Mantiene el Dropout activado durante la inferencia para calcular incertidumbre."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)