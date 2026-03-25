import tensorflow as tf
from tensorflow.keras import layers
from src.models.custom_layers import ResidualBlock, RecomputeGradLayer

def get_resnet18_backbone(inputs, l2_reg=1e-4, use_checkpointing=False):
    """Devuelve el tronco de la ResNet-18 (hasta el GlobalAveragePooling)."""
    x = layers.Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    filters = 64
    block_configs = [2, 2, 2, 2]
    
    for i, num_blocks in enumerate(block_configs):
        if i > 0: filters *= 2
        for j in range(num_blocks):
            stride = 2 if i > 0 and j == 0 else 1
            res_block_layer = ResidualBlock(filters, stride=stride, conv_shortcut=True, name=f'block_{i+1}_layer_{j+1}')
            
            if use_checkpointing:
                x = RecomputeGradLayer(res_block_layer)(x)
            else:
                x = res_block_layer(x)
        
    x = layers.GlobalAveragePooling2D()(x)
    return x