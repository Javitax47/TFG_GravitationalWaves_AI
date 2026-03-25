# src/training/train_master.py

import os
import sys
import glob
import json
import numpy as np
import tensorflow as tf
import keras_tuner as kt

# --- CONFIGURACIÓN DEL ENTORNO ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={conda_prefix}'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

ARCHITECTURE = os.environ.get('ARCHITECTURE', 'CNN-2D')
if 'ViT' not in ARCHITECTURE:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# --- RUTAS DEL PROYECTO ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.model_factory import build_grid_model
from src.data_processing.tfrecord_parsers import create_grid_dataset

# ==============================================================================
# CONFIGURACIÓN DEL EXPERIMENTO (Vía Variables de Entorno)
# ==============================================================================
# El script Bash inyectará estas variables. Si no existen, usamos valores por defecto.
MODE = os.environ.get('EXECUTION_MODE', 'train').lower()
ARCHITECTURE = os.environ.get('ARCHITECTURE', 'CNN-2D')
DIMENSION = os.environ.get('DIMENSION', '2D')
# Convertimos el string de tareas (ej. "detection,masses") en una lista de Python
TASKS_STR = os.environ.get('TASKS', 'detection')
TASKS =[t.strip() for t in TASKS_STR.split(',')]

print(f"\n{'='*70}")
print(f"🚀 INICIANDO EXPERIMENTO GRID: {ARCHITECTURE} | MODO: {MODE.upper()} 🚀")
print(f"   Tareas Activas: {TASKS}")
print(f"   Dimensión de Entrada: {DIMENSION}")
print(f"{'='*70}")

# --- RUTAS DINÁMICAS BASADAS EN EL EXPERIMENTO ---
# Usamos un sufijo único para no pisar las carpetas de otros modelos
EXP_SUFFIX = f"{ARCHITECTURE.lower()}_{'_'.join(TASKS)}"

if DIMENSION == '2D':
    INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/processed/tfrecord_chunks')
    TARGET_SHAPE = (256, 512, 1)
else:
    INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/processed/tfrecord_1d_chunks')
    TARGET_SHAPE = (8192, 1)

HP_CONFIG_PATH = os.path.join(PROJECT_ROOT, f'trained_models/best_hps_{EXP_SUFFIX}.json')
TUNER_DIR = os.path.join(PROJECT_ROOT, f'keras_tuner_{EXP_SUFFIX}')
BACKUP_DIR_BASE = os.path.join(PROJECT_ROOT, f'trained_models/backup_{EXP_SUFFIX}')
LOGS_DIR = os.path.join(PROJECT_ROOT, f'logs/{EXP_SUFFIX}')

# --- HIPERPARÁMETROS GLOBALES ---
EPOCHS = 150 if len(TASKS) == 1 else 250  
PATIENCE = 20
NUM_FOLDS = 5

# Ajuste Inteligente del Lote
if 'ViT' in ARCHITECTURE:
    BATCH_SIZE = 16
else:
    # Si es CNN, 32 está bien para Single-Task, pero bajamos a 16 si hay múltiples cabezas
    BATCH_SIZE = 32 if len(TASKS) == 1 else 16

# Funciones de utilidad para memoria y backups (Iguales que las que ya usabas)
import gc
class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()

class ResumableModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, state_file, **kwargs):
        super().__init__(filepath, **kwargs)
        self.state_file = state_file
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.best = json.load(f).get('best', np.Inf if self.monitor_op == np.less else -np.Inf)
            except Exception: pass
    def _save_model(self, epoch, batch, logs):
        super()._save_model(epoch, batch, logs)
        try:
            with open(self.state_file, 'w') as f: json.dump({'best': float(self.best)}, f)
        except Exception: pass

# ==============================================================================
# MOTOR DE COMPILACIÓN MULTI-TASK
# ==============================================================================
def compile_grid_model(model, hps_dict):
    """Asigna la función de pérdida y métrica correcta a cada cabeza activa."""
    lr = hps_dict.get('learning_rate', 1e-4 if 'ViT' in ARCHITECTURE else 5e-4)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4) if 'ViT' in ARCHITECTURE else tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Diccionarios dinámicos para Keras
    losses = {}
    loss_weights = {}
    metrics = {}

    if 'detection' in TASKS:
        # En Single-Task Keras no usa nombres de diccionario
        key = 'head_detection' if len(TASKS) > 1 else model.output_names[0]
        losses[key] = 'binary_crossentropy'
        loss_weights[key] = 1.0
        metrics[key] = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]

    if 'classification' in TASKS:
        key = 'head_classification' if len(TASKS) > 1 else model.output_names[0]
        # Usamos Sparse porque las etiquetas son enteros (0,1,2,3)
        losses[key] = 'sparse_categorical_crossentropy'
        loss_weights[key] = 0.5  # Le damos un poco menos de peso que a la detección pura
        metrics[key] = ['accuracy']

    if 'masses' in TASKS:
        key = 'head_masses' if len(TASKS) > 1 else model.output_names[0]
        from src.training.custom_losses import MaskedHuberLoss
        losses[key] = MaskedHuberLoss(mask_value=0.0)
        loss_weights[key] = 0.5
        metrics[key] = ['mae']

    # Compilación dinámica
    if len(TASKS) == 1:
        model.compile(optimizer=optimizer, loss=list(losses.values())[0], metrics=list(metrics.values())[0], jit_compile=False)
    else:
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics, jit_compile=False)

    return model

# ==============================================================================
# MODO TUNE Y TRAIN
# ==============================================================================
def run_tuner(train_ds, val_ds):
    print("\n--- MODO TUNE: BÚSQUEDA DE HIPERPARÁMETROS ---")
    
    def model_builder(hp):
        tf.keras.backend.clear_session()
        # Hiperparámetros de búsqueda unificados
        hps_dict = {
            'dense_units': hp.Int('dense_units', min_value=128, max_value=512, step=128),
            'dropout_rate': hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1),
            'learning_rate': hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])
        }
        if 'ViT' in ARCHITECTURE:
            hps_dict['projection_dim'] = hp.Choice('projection_dim', values=[64, 128])
            hps_dict['num_heads'] = hp.Choice('num_heads', values=[2, 4])
            hps_dict['transformer_layers'] = hp.Int('transformer_layers', min_value=2, max_value=6, step=2)
        else:
            hps_dict['l2_reg'] = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
            
        model = build_grid_model(ARCHITECTURE, TASKS, TARGET_SHAPE, hps_dict)
        return compile_grid_model(model, hps_dict)

    # Determinar qué métrica monitorizar para Early Stopping
    if len(TASKS) == 1:
        monitor_metric = 'val_loss' if 'masses' in TASKS else 'val_accuracy'
    else:
        monitor_metric = 'val_loss' # En Multi-Task, la pérdida global es la suma de las pérdidas ponderadas

    tuner = kt.Hyperband(
        hypermodel=model_builder, objective=monitor_metric, max_epochs=20, factor=3,
        directory=TUNER_DIR, project_name='grid_search', overwrite=False
    )
    
    tuner.search(train_ds, validation_data=val_ds, epochs=20,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=5), ClearMemoryCallback()],
                 verbose=2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    os.makedirs(os.path.dirname(HP_CONFIG_PATH), exist_ok=True)
    with open(HP_CONFIG_PATH, 'w') as f: json.dump(best_hps.values, f, indent=4)

def run_kfold_training(all_chunk_files):
    print(f"\n--- MODO TRAIN: K-FOLD CROSS-VALIDATION ({NUM_FOLDS} Folds) ---")
    from sklearn.model_selection import KFold
    
    if not os.path.exists(HP_CONFIG_PATH):
        # Fallback de seguridad si no se ha hecho Tuning
        print(f"⚠️ No hay Tuning previo. Usando Hiperparámetros por defecto.")
        hps_values = {}
    else:
        with open(HP_CONFIG_PATH, 'r') as f: hps_values = json.load(f)

    np.random.seed(42)
    all_files_copy = all_chunk_files.copy()
    np.random.shuffle(all_files_copy)
    val_end = int(0.8 * len(all_files_copy))
    train_val_files = np.array(all_files_copy[:val_end])

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_files)):
        fold_model_path = os.path.join(PROJECT_ROOT, f'trained_models/{EXP_SUFFIX}_fold_{fold+1}.keras')
        fold_backup_dir = f"{BACKUP_DIR_BASE}_fold_{fold+1}"
        fold_state_file = os.path.join(PROJECT_ROOT, f'trained_models/checkpoint_state_{EXP_SUFFIX}_fold_{fold+1}.json')

        if os.path.exists(fold_model_path) and not os.path.exists(fold_backup_dir):
            print(f"✅ Fold {fold+1} completado. Saltando...")
            continue

        print(f"\n⭐ ENTRENANDO FOLD {fold + 1}/{NUM_FOLDS} ⭐")
        train_files = train_val_files[train_idx].tolist()
        val_files = train_val_files[val_idx].tolist()

        train_ds = create_grid_dataset(train_files, DIMENSION, TASKS, BATCH_SIZE, is_training=True, use_mixup=True, use_time_shift=True)
        val_ds = create_grid_dataset(val_files, DIMENSION, TASKS, BATCH_SIZE, is_training=False)

        model = build_grid_model(ARCHITECTURE, TASKS, TARGET_SHAPE, hps_dict=hps_values)
        model = compile_grid_model(model, hps_dict=hps_values)
        
        monitor_metric = 'val_loss' if len(TASKS) > 1 or 'masses' in TASKS else 'val_accuracy'
        
        callbacks =[
            tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=7, min_lr=1e-7),
            ResumableModelCheckpoint(filepath=fold_model_path, state_file=fold_state_file, monitor=monitor_metric, save_best_only=True),
            tf.keras.callbacks.BackupAndRestore(backup_dir=fold_backup_dir)
        ]

        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
        tf.keras.backend.clear_session()
        import gc; gc.collect()

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    all_chunk_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, '*.tfrecord')))
    if not all_chunk_files:
        sys.exit(f"❌ Error: No hay datos en {INPUT_DATA_DIR}")

    if MODE == 'tune':
        np.random.seed(42)
        np.random.shuffle(all_chunk_files)
        val_end = int(0.8 * len(all_chunk_files))
        train_end = int(0.8 * val_end)
        train_ds = create_grid_dataset(all_chunk_files[:train_end], DIMENSION, TASKS, BATCH_SIZE, is_training=True, use_mixup=True, use_time_shift=True)
        val_ds = create_grid_dataset(all_chunk_files[train_end:val_end], DIMENSION, TASKS, BATCH_SIZE, is_training=False)
        run_tuner(train_ds, val_ds)
        
    elif MODE == 'train':
        run_kfold_training(all_chunk_files)

if __name__ == '__main__':
    main()