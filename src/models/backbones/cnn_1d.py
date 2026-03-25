import tensorflow as tf
from tensorflow.keras import layers
from src.models.custom_layers import MCDropout

def get_cnn1d_backbone(inputs, filters=[64, 128, 256, 512], dropout_rate=0.2):
    """Devuelve el tronco de la CNN-1D para series temporales puras."""
    x = inputs
    
    for f in filters:
        x = layers.Conv1D(filters=f, kernel_size=16, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv1D(filters=f, kernel_size=16, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = MCDropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling1D()(x)
    return x