# train_cnn.py (versión final leyendo TFRecords)

import os
import sys
import shutil
import tensorflow as tf
import numpy as np
import json
import glob
import keras_tuner as kt

# Desactivamos el compilador JIT (XLA) para evitar errores de entorno.
tf.config.optimizer.set_jit(False)

# --- CONTROL Y LOGGING ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- SOLUCIÓN: RUTA DE XLA ---
if 'CONDA_PREFIX' in os.environ:
    cuda_path = os.environ['CONDA_PREFIX']
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_path}'
    print("="*80)
    print(f"INFO: Ruta de CUDA para XLA establecida en: {os.environ['XLA_FLAGS']}")
    print("="*80)
else:
    print("ADVERTENCIA: No se está en un entorno Conda. La ruta de XLA no se ha establecido.")

# --- Importaciones del Proyecto ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(project_root)
from src.models.cnn_classifier import build_resnet18_classifier

# ==============================================================================
# --- CONFIGURACIÓN PRINCIPAL ---
# ==============================================================================
MODE = 'tune'
# ### CAMBIO IMPORTANTE: Apuntar al directorio con los datos TFRecord ###
INPUT_DATA_DIR = os.path.join(project_root, 'data/processed/tfrecord_chunks')
OUTPUT_MODEL_PATH = os.path.join(project_root, 'trained_models/resnet18_final.keras')
HP_CONFIG_PATH = os.path.join(project_root, 'trained_models/best_hyperparameters.json')
LOGS_DIR = os.path.join(project_root, 'logs')
TUNER_DIR = os.path.join(project_root, 'keras_tuner')

# --- Hiperparámetros ---
EPOCHS = 150
PATIENCE = 15
TUNE_EPOCHS = 20
TUNE_PATIENCE = 5
BATCH_SIZE = 1 # Un batch size conservador para máxima estabilidad

model_input_shape = (500, 970, 1)

# ==============================================================================
# --- Lógica de Carga de Datos (LEYENDO TFRECORDS) ---
# ==============================================================================

def parse_tfrecord_fn(example):
    """Función para decodificar cada registro del archivo TFRecord."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    image.set_shape(model_input_shape) # Asegurar la forma después de parsear
    
    label = tf.cast(example['label'], tf.int8)
    
    return image, label

def create_dataset(file_paths, batch_size, shuffle=True):
    """
    Crea un pipeline de tf.data optimizado para leer archivos TFRecord.
    Este es el método más rápido y eficiente en memoria.
    """
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    
    if shuffle:
        path_ds = path_ds.shuffle(len(file_paths), reshuffle_each_iteration=True)

    dataset = path_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        # Un buffer de 4000 es seguro para 32GB de RAM y da una excelente aleatorización
        dataset = dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)

    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ==============================================================================
# --- Funciones de Modo de Ejecución ---
# ==============================================================================
def build_model(hp):
    hp_dense_units = hp.Int('dense_units', min_value=128, max_value=256, step=128)
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    hp_l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
    hp_learning_rate = hp.Choice('learning_rate', values=[5e-4, 1e-4, 5e-5])
    
    model = build_resnet18_classifier(
        input_shape=model_input_shape,
        dense_units=hp_dense_units,
        dropout_rate=hp_dropout_rate,
        l2_reg=hp_l2_reg
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_tuner(train_dataset, val_dataset):
    print("\n--- INICIANDO MODO 'TUNE': BÚSQUEDA DE HIPERPARÁMETROS ---")
    
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_accuracy',
        max_epochs=TUNE_EPOCHS,
        factor=3,
        directory=TUNER_DIR,
        project_name='gw_classification',
        overwrite=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, 'tuner'))
    
    print("\nIniciando búsqueda de hiperparámetros...")
    # Keras inferirá los pasos del dataset finito, no necesitamos `steps_per_epoch`
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=TUNE_EPOCHS
        callbacks=[tensorboard_callback, tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TUNE_PATIENCE)],
        verbose=1
    )

    print("\nBúsqueda completada. Resumen de los mejores resultados:")
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nMejores hiperparámetros encontrados: {best_hps.values}")
    
    os.makedirs(os.path.dirname(HP_CONFIG_PATH), exist_ok=True)
    with open(HP_CONFIG_PATH, 'w') as f:
        json.dump(best_hps.values, f, indent=4)
    print(f"La mejor configuración ha sido guardada en: {HP_CONFIG_PATH}")

def run_training(train_dataset, val_dataset, test_dataset):
    print("\n--- INICIANDO MODO 'TRAIN': ENTRENAMIENTO FINAL ---")
    if not os.path.exists(HP_CONFIG_PATH):
        sys.exit(f"ERROR: No se encontró el archivo de hiperparámetros en {HP_CONFIG_PATH}")
    with open(HP_CONFIG_PATH, 'r') as f: hps_values = json.load(f)
    
    hps = kt.HyperParameters()
    for key, value in hps_values.items(): hps.Fixed(key, value=value)
    model = build_model(hps)
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, 'final_training'))
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n--- ENTRENAMIENTO FINAL COMPLETADO ---")
    best_model = tf.keras.models.load_model(OUTPUT_MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_dataset)
    print(f"  -> Pérdida final en Prueba: {test_loss:.4f}")
    print(f"  -> Precisión final en Prueba: {test_accuracy:.4f}")

# ==============================================================================
# --- Orquestador Principal ---
# ==============================================================================
def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Error al configurar el crecimiento de memoria: {e}")
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Política de precisión mixta ('mixed_float16') habilitada.")

    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    all_chunk_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, '*.tfrecord')))
    if not all_chunk_files:
        sys.exit(f"Error: No se encontraron archivos de datos en '{INPUT_DATA_DIR}'.\nAsegúrate de haber ejecutado `convert_to_tfrecord.py` primero.")
    
    np.random.shuffle(all_chunk_files)
    num_files = len(all_chunk_files)
    train_end, val_end = int(0.6 * num_files), int(0.8 * num_files)
    
    train_files = all_chunk_files[:train_end]
    val_files = all_chunk_files[train_end:val_end]
    test_files = all_chunk_files[val_end:]
    
    if not val_files and train_files: val_files.append(train_files.pop())
    if not test_files and train_files: test_files.append(train_files.pop())
    print(f"División de archivos: {len(train_files)} para train, {len(val_files)} para val, {len(test_files)} para test.")

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(TUNER_DIR, exist_ok=True)

    print("\nCreando pipelines de datos con tf.data...")
    train_ds = create_dataset(train_files, BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(val_files, BATCH_SIZE, shuffle=False)

    if MODE == 'tune':
        print("Modo 'TUNE' seleccionado.")
        run_tuner(train_ds, val_ds)
        
    elif MODE == 'train':
        print("Modo 'TRAIN' seleccionado.")
        test_ds = create_dataset(test_files, BATCH_SIZE, shuffle=False)
        run_training(train_ds, val_ds, test_ds)
        
    else:
        print(f"Error: Modo '{MODE}' no reconocido. Usa 'tune' o 'train'.")

if __name__ == '__main__':
    try:
        main()
        print("\n--- SCRIPT COMPLETADO CON ÉXITO ---")
    except Exception as e:
        print("\n" + "="*80)
        print("¡ERROR! Se ha detectado una excepción no controlada. Iniciando limpieza...")
        print("="*80)
        
        if os.path.exists(TUNER_DIR) and os.path.isdir(TUNER_DIR):
            try:
                shutil.rmtree(TUNER_DIR)
                print(f"  - Directorio de KerasTuner eliminado con éxito: {TUNER_DIR}")
            except OSError as err:
                print(f"  - Error al eliminar el directorio de KerasTuner: {err}")
        
        print("\nLimpieza finalizada. El error original se mostrará a continuación:")
        print("="*80 + "\n")
        
        raise e