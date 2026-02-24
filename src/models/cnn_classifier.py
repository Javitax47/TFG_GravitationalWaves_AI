# src/models/cnn_classifier.py

import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualBlock(layers.Layer):
    """
    Bloque residual de ResNet implementado como una capa de Keras.
    """
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
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

class RecomputeGradLayer(layers.Layer):
    """Capa que envuelve una capa o función con tf.recompute_grad."""
    def __init__(self, recomputable_layer, **kwargs):
        super().__init__(**kwargs)
        self.recomputable_layer = recomputable_layer

    def call(self, inputs):
        # Esta función le dice a TensorFlow que no guarde las activaciones de esta capa,
        # sino que las recalcule durante el paso hacia atrás.
        return tf.recompute_grad(self.recomputable_layer)(inputs)

    def get_config(self):
        config = super().get_config()
        # No necesitamos guardar la capa interna en el JSON,
        # ya que el modelo se reconstruye de otra forma.
        # Simplemente devolvemos la configuración base.
        return config


# ### CAMBIO CLAVE 1: use_checkpointing ahora es True por defecto ###
def build_resnet18_classifier(input_shape, dense_units=256, dropout_rate=0.4, l2_reg=1e-4, use_checkpointing=True):
    input_tensor = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    filters = 64
    block_configs = [2, 2, 2, 2]
    
    for i, num_blocks in enumerate(block_configs):
        if i > 0:
            filters *= 2
        for j in range(num_blocks):
            stride = 2 if i > 0 and j == 0 else 1
            
            res_block_layer = ResidualBlock(
                filters, stride=stride, conv_shortcut=True, name=f'block_{i+1}_layer_{j+1}'
            )

            # ### CAMBIO CLAVE 2: La lógica de checkpointing está ahora activa ###
            if use_checkpointing:
                # Envolvemos el bloque residual para ahorrar memoria VRAM.
                x = RecomputeGradLayer(res_block_layer)(x)
            else:
                # Se ejecuta de la forma normal si se desactiva.
                x = res_block_layer(x)
        
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    output_tensor = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    
    # ### CAMBIO CLAVE 3: Mensaje de confirmación ###
    print(f"Modelo ResNet-18 construido. Gradient Checkpointing: {'ACTIVADO' if use_checkpointing else 'DESACTIVADO'}")
    return model