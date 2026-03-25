import tensorflow as tf
from tensorflow.keras import layers
from src.models.custom_layers import Patches, PatchEncoder

def get_vit_backbone(inputs, patch_size=16, num_heads=4, transformer_layers=4, projection_dim=128):
    """Devuelve el tronco del Vision Transformer (hasta el GlobalAveragePooling)."""
    input_shape = inputs.shape[1:] # Ignorar batch_size
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = layers.GlobalAveragePooling1D()(representation)
    return x