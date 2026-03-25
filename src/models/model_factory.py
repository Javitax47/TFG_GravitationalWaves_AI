# src/models/model_factory.py

import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.backbones.transformer_1d import get_vit1d_backbone

# Importamos los Backbones (Extraen el ADN de la señal)
from src.models.backbones.cnn_2d import get_resnet18_backbone
from src.models.backbones.vit_2d import get_vit_backbone
from src.models.backbones.cnn_1d import get_cnn1d_backbone

# Importamos las Cabezas (Toman la decisión)
from src.models.heads import get_detection_head, get_classification_head, get_masses_head
from src.models.custom_layers import MCDropout

def build_grid_model(architecture_name, tasks, input_shape, hps_dict=None):
    """
    Ensambla el modelo bajo demanda.
    - architecture_name: 'CNN-2D', 'ViT-2D', 'CNN-1D', 'ViT-1D'
    - tasks: lista de tareas ej. ['detection'], ['detection', 'masses', 'classification']
    - input_shape: (256, 512, 1) para 2D, o (8192, 1) para 1D
    """
    tf.keras.backend.clear_session()
    inputs = layers.Input(shape=input_shape)
    
    # Valores por defecto de hiperparámetros si no se pasan
    hps = hps_dict or {}
    dense_units = hps.get('dense_units', 256)
    dropout_rate = hps.get('dropout_rate', 0.2)
    l2_reg_val = hps.get('l2_reg', 1e-4)

    # ==========================================================================
    # 1. INSTANCIAR EL TRONCO (BACKBONE)
    # ==========================================================================
    if architecture_name == "CNN-2D":
        x = get_resnet18_backbone(inputs, l2_reg=l2_reg_val, use_checkpointing=False)
        activation = 'relu'
        l2_reg = tf.keras.regularizers.l2(l2_reg_val)
        
    elif architecture_name == "ViT-2D":
        x = get_vit_backbone(
            inputs, 
            patch_size=16, 
            num_heads=hps.get('num_heads', 4), 
            transformer_layers=hps.get('transformer_layers', 4), 
            projection_dim=hps.get('projection_dim', 128)
        )
        activation = 'gelu'
        l2_reg = None
        
    elif architecture_name == "CNN-1D":
        x = get_cnn1d_backbone(inputs, dropout_rate=dropout_rate)
        activation = 'relu'
        l2_reg = None
        
    elif architecture_name == "ViT-1D":
        # Extraemos el tronco del Transformer 1D
        x = get_vit1d_backbone(
            inputs,
            patch_size=64, # Ajustable según experimentación
            num_heads=hps.get('num_heads', 4),
            transformer_layers=hps.get('transformer_layers', 4),
            projection_dim=hps.get('projection_dim', 128)
        )
        activation = 'gelu'
        l2_reg = None
        
    else:
        raise ValueError(f"Arquitectura {architecture_name} no reconocida.")

    # ==========================================================================
    # 2. INSTANCIAR LAS CABEZAS (HEADS)
    # ==========================================================================
    outputs =[]
    
    # ----------------------------------------------------------------------
    # PROTECCIÓN ZERO-BREAK: Mantener nombres idénticos para modelos antiguos
    # ----------------------------------------------------------------------
    if len(tasks) == 1 and tasks[0] == 'detection' and architecture_name in["CNN-2D", "ViT-2D"]:
        # Replicamos literalmente el final de tus antiguos CNN y ViT
        x = layers.Dense(dense_units, activation=activation, kernel_regularizer=l2_reg)(x)
        x = layers.Dropout(dropout_rate)(x)
        out = layers.Dense(1, activation='sigmoid')(x) # Keras lo llamará dense_1 automáticamente
        outputs.append(out)
        
    elif len(tasks) == 1 and tasks[0] == 'masses' and architecture_name == "CNN-1D":
        # Replicamos el estimador de masas antiguo
        x = layers.Dense(dense_units, activation='relu', kernel_initializer='he_normal')(x)
        x = MCDropout(dropout_rate)(x)
        x = layers.Dense(dense_units // 2, activation='relu', kernel_initializer='he_normal')(x)
        x = MCDropout(dropout_rate)(x)
        out = layers.Dense(2, activation='linear', name='mass_predictions')(x)
        outputs.append(out)
        
    # ----------------------------------------------------------------------
    # MODO MULTI-TASKING (La IA General)
    # ----------------------------------------------------------------------
    else:
        if 'detection' in tasks:
            outputs.append(get_detection_head(x, dense_units, dropout_rate, activation, l2_reg))
            
        if 'classification' in tasks:
            outputs.append(get_classification_head(x, dense_units, dropout_rate, activation))
            
        if 'masses' in tasks:
            outputs.append(get_masses_head(x, dense_units, dropout_rate))

    # ==========================================================================
    # 3. ENSAMBLAJE FINAL
    # ==========================================================================
    model = models.Model(inputs=inputs, outputs=outputs)
    
    task_str = " + ".join([t.capitalize() for t in tasks])
    print(f"\n[+] Fábrica de Modelos: {architecture_name} ({task_str}) ensamblado con éxito.")
    
    return model