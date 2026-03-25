# src/models/backbones/transformer_1d.py

import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="Custom")
class PatchEncoder1D(layers.Layer):
    """
    Convierte la serie temporal 1D en 'Tokens' y añade Codificación Posicional.
    Usa una Conv1D con stride = kernel_size para extraer parches no superpuestos.
    """
    def __init__(self, num_patches, projection_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        
        # Proyección lineal (Extrae el parche y lo convierte en un vector)
        self.projection = layers.Conv1D(
            filters=projection_dim, 
            kernel_size=patch_size, 
            strides=patch_size, 
            padding="valid"
        )
        # Codificación posicional
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, inputs):
        # inputs shape: (batch, 8192, 1)
        patches = self.projection(inputs) # shape: (batch, num_patches, projection_dim)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "patch_size": self.patch_size
        })
        return config


def get_vit1d_backbone(inputs, patch_size=64, num_heads=4, transformer_layers=4, projection_dim=128):
    """
    Devuelve el tronco del Vision Transformer adaptado a Series Temporales 1D.
    """
    # Calculamos el número de secuencias (Ej: 8192 / 64 = 128 tokens)
    sequence_length = inputs.shape[1]
    num_patches = sequence_length // patch_size
    
    # 1. Codificación
    encoded_patches = PatchEncoder1D(num_patches, projection_dim, patch_size)(inputs)

    # 2. Bloques Transformer
    for _ in range(transformer_layers):
        # Capa de Normalización 1 + Atención
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip Connection
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Capa de Normalización 2 + Feed Forward (MLP)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip Connection
        encoded_patches = layers.Add()([x3, x2])

    # 3. Global Average Pooling (Colapso del tiempo para extraer el ADN final)
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAveragePooling1D()(representation)
    
    return x