import tensorflow as tf
import h5py
import glob
import os
import json
import re
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
HDF5_CHUNKS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/qtransform_chunks_resized')
TFRECORD_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/processed/tfrecord_chunks')
MASTER_HDF5 = os.path.join(PROJECT_ROOT, 'data/processed/dataset_80000_samples.hdf5')

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value): return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

EVENT_TYPE_MAP = {'noise': 0, 'BBH': 1, 'BNS': 2, 'NSBH': 3}

def serialize_example(image, label_detect, event_type_int, mass1, mass2, snr):
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _int64_feature(label_detect),
        'event_type': _int64_feature(event_type_int),
        'mass1': _float_feature(mass1),
        'mass2': _float_feature(mass2),
        'snr': _float_feature(snr)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

# --- SOLUCIÓN AL BUG ESTADÍSTICO ---
def get_chunk_number(filepath):
    """Extrae el número de parte para ordenar correctamente (1, 2... 10, 11)."""
    match = re.search(r'part_(\d+)_of', filepath)
    return int(match.group(1)) if match else 0

def main():
    os.makedirs(TFRECORD_OUTPUT_DIR, exist_ok=True)
    
    # Aplicamos la ordenación matemática, no alfabética
    hdf5_chunks = sorted(glob.glob(os.path.join(HDF5_CHUNKS_DIR, '*.hdf5')), key=get_chunk_number)
    
    if not hdf5_chunks:
        print(f"❌ Error: No se encontraron chunks en {HDF5_CHUNKS_DIR}")
        return

    print(f"Encontrados {len(hdf5_chunks)} trozos de imágenes.")
    
    old_tfrecords = glob.glob(os.path.join(TFRECORD_OUTPUT_DIR, '*.tfrecord'))
    for f in old_tfrecords: os.remove(f)
    print("TFRecords antiguos eliminados.")

    with h5py.File(MASTER_HDF5, 'r') as f_master:
        all_parameters = f_master['parameters'][:]
        global_sample_idx = 0
        
        for chunk_path in tqdm(hdf5_chunks, desc="Conversión Inteligente Ordenada"):
            base_name = os.path.basename(chunk_path).replace('.hdf5', '.tfrecord')
            tfrecord_path = os.path.join(TFRECORD_OUTPUT_DIR, base_name)

            with h5py.File(chunk_path, 'r') as f_chunk, tf.io.TFRecordWriter(tfrecord_path) as writer:
                images = f_chunk['X'][:]
                labels = f_chunk['y'][:]
                
                for i in range(len(labels)):
                    param_str = all_parameters[global_sample_idx].decode('utf-8')
                    param_dict = json.loads(param_str)
                    
                    ev_type_str = param_dict.get('event_type', 'noise')
                    ev_type_int = EVENT_TYPE_MAP.get(ev_type_str, 0)
                    mass1 = float(param_dict.get('mass1', 0.0))
                    mass2 = float(param_dict.get('mass2', 0.0))
                    snr = float(param_dict.get('snr', -1.0))
                    
                    serialized_example = serialize_example(
                        images[i], labels[i], ev_type_int, mass1, mass2, snr
                    )
                    writer.write(serialized_example)
                    global_sample_idx += 1
            os.remove(chunk_path)

    print(f"\n✅ ¡Física Restaurada! Los TFRecords ahora son matemáticamente perfectos.")

if __name__ == '__main__':
    main()