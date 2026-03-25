# src/data_processing/convert_to_tfrecord_1d.py

import tensorflow as tf
import h5py
import os
import json
import numpy as np
from tqdm import tqdm

# --- RUTAS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MASTER_HDF5 = os.path.join(PROJECT_ROOT, 'data/processed/dataset_80000_samples.hdf5') # Cambia a 60000 si es tu caso
TFRECORD_1D_DIR = os.path.join(PROJECT_ROOT, 'data/processed/tfrecord_1d_chunks')

# Constantes de chunking (para igualar a los 2D)
CHUNK_SIZE = 3000  

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value): return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

EVENT_TYPE_MAP = {'noise': 0, 'BBH': 1, 'BNS': 2, 'NSBH': 3}

def serialize_example_1d(signal_1d, label_detect, event_type_int, mass1, mass2, snr):
    feature = {
        # Guardamos la onda 1D (8192 floats)
        'signal': _bytes_feature(tf.io.serialize_tensor(tf.cast(signal_1d, tf.float32))),
        'label': _int64_feature(label_detect),
        'event_type': _int64_feature(event_type_int),
        'mass1': _float_feature(mass1),
        'mass2': _float_feature(mass2),
        'snr': _float_feature(snr)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def main():
    os.makedirs(TFRECORD_1D_DIR, exist_ok=True)
    print(f"Abriendo archivo maestro: {MASTER_HDF5}")
    
    with h5py.File(MASTER_HDF5, 'r') as f_master:
        X_1d = f_master['X'][:]  # Ondas puras (N, 8192)
        y_labels = f_master['y'][:]
        parameters = f_master['parameters'][:]
        
        num_samples = len(y_labels)
        num_chunks = (num_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        print(f"Generando {num_chunks} chunks de datos 1D puros...")

        for chunk_idx in tqdm(range(num_chunks), desc="Escribiendo TFRecords 1D"):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min((chunk_idx + 1) * CHUNK_SIZE, num_samples)
            
            tfrecord_path = os.path.join(TFRECORD_1D_DIR, f"qtransforms_{num_samples}_samples_part_{chunk_idx+1}_of_{num_chunks}.tfrecord")
            
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for i in range(start_idx, end_idx):
                    # Extraer física
                    param_str = parameters[i].decode('utf-8')
                    param_dict = json.loads(param_str)
                    
                    ev_type_str = param_dict.get('event_type', 'noise')
                    ev_type_int = EVENT_TYPE_MAP.get(ev_type_str, 0)
                    mass1 = float(param_dict.get('mass1', 0.0))
                    mass2 = float(param_dict.get('mass2', 0.0))
                    snr = float(param_dict.get('snr', -1.0))
                    
                    # Añadimos una dimensión extra al final (8192,) -> (8192, 1) para las CNNs 1D
                    signal = np.expand_dims(X_1d[i], axis=-1)
                    
                    serialized = serialize_example_1d(signal, y_labels[i], ev_type_int, mass1, mass2, snr)
                    writer.write(serialized)

    print("\n✅ ¡TFRecords 1D generados con éxito! Los datos 2D originales están INTACTOS.")

if __name__ == '__main__':
    main()