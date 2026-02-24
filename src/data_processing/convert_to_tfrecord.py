# convert_to_tfrecord.py
import tensorflow as tf
import h5py
import glob
import os
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Apuntar al directorio con los datos HDF5 redimensionados
HDF5_DATA_DIR = 'data/processed/qtransform_chunks_resized'
# Directorio donde se guardarán los nuevos archivos TFRecord
TFRECORD_OUTPUT_DIR = 'data/processed/tfrecord_chunks'

def _bytes_feature(value):
    """Devuelve un bytes_list a partir de un string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Devuelve un int64_list a partir de un bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    """Crea un tf.train.Example a partir de una imagen y una etiqueta."""
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def main():
    os.makedirs(TFRECORD_OUTPUT_DIR, exist_ok=True)
    hdf5_files = sorted(glob.glob(os.path.join(HDF5_DATA_DIR, '*.hdf5')))

    print(f"Encontrados {len(hdf5_files)} archivos HDF5 para convertir.")

    for hdf5_path in tqdm(hdf5_files, desc="Convirtiendo archivos a TFRecord"):
        base_name = os.path.basename(hdf5_path).replace('.hdf5', '.tfrecord')
        tfrecord_path = os.path.join(TFRECORD_OUTPUT_DIR, base_name)

        with h5py.File(hdf5_path, 'r') as f_in, tf.io.TFRecordWriter(tfrecord_path) as writer:
            images = f_in['X'][:]
            labels = f_in['y'][:]
            
            for i in range(len(labels)):
                serialized_example = serialize_example(images[i], labels[i])
                writer.write(serialized_example)

    print("\n¡Conversión a TFRecord completada!")

if __name__ == '__main__':
    main()