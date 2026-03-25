# src/models/heads.py

import tensorflow as tf
from tensorflow.keras import layers
from src.models.custom_layers import MCDropout

def get_detection_head(x, dense_units, dropout_rate, activation, l2_reg=None, name='head_detection'):
    """Cabeza para Detección Binaria (Señal vs Ruido)."""
    # Usamos name_prefix en las capas internas para que en Multi-Task no choquen los nombres
    x = layers.Dense(dense_units, activation=activation, kernel_regularizer=l2_reg, name=f'dense_{name}')(x)
    x = layers.Dropout(dropout_rate, name=f'drop_{name}')(x)
    return layers.Dense(1, activation='sigmoid', name=name)(x)

def get_classification_head(x, dense_units, dropout_rate, activation, name='head_classification'):
    """Cabeza para Clasificación Multiclase (Ruido, BBH, BNS, NSBH)."""
    x = layers.Dense(dense_units, activation=activation, name=f'dense_{name}')(x)
    x = layers.Dropout(dropout_rate, name=f'drop_{name}')(x)
    # 4 neuronas de salida con Softmax (Probabilidades que suman 1)
    return layers.Dense(4, activation='softmax', name=name)(x)

def get_masses_head(x, dense_units, dropout_rate, name='head_masses'):
    """Cabeza para Regresión Bayesiana de Masas (M1, M2)."""
    x = layers.Dense(dense_units, activation='relu', kernel_initializer='he_normal', name=f'dense1_{name}')(x)
    x = MCDropout(dropout_rate, name=f'mcdrop1_{name}')(x)
    x = layers.Dense(dense_units // 2, activation='relu', kernel_initializer='he_normal', name=f'dense2_{name}')(x)
    x = MCDropout(dropout_rate, name=f'mcdrop2_{name}')(x)
    # 2 neuronas Lineales (Para predecir números continuos sin límite)
    return layers.Dense(2, activation='linear', name=name)(x)