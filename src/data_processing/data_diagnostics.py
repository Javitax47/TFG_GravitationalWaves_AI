# src/data_processing/data_diagnostics.py

import os
import sys
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Obligamos a usar la CPU para no molestar a la GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TFRECORD_DIR = os.path.join(project_root, 'data/processed/tfrecord_chunks')
TARGET_SHAPE = (256, 512, 1)
BATCH_SIZE = 16

def raw_parse_fn(example):
    """Parsea el TFRecord pero NO lo normaliza (Datos Crudos)"""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    image.set_shape(TARGET_SHAPE)
    label = tf.cast(example['label'], tf.int8)
    return image, label

def diagnostic_parse_fn(example):
    """Devuelve la imagen cruda Y la imagen normalizada para compararlas"""
    raw_image, label = raw_parse_fn(example)
    # Aplicamos la normalización propuesta
    norm_image = tf.image.per_image_standardization(raw_image)
    return raw_image, norm_image, label

def main():
    print("="*60)
    print("🔬 INICIANDO AUDITORÍA DE DATOS (TFRECORDS) 🔬")
    print("="*60)

    files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord'))
    if not files:
        print("❌ Error: No se encontraron archivos TFRecord.")
        return

    dataset = tf.data.TFRecordDataset(files[0])
    dataset = dataset.map(diagnostic_parse_fn).batch(BATCH_SIZE)

    for raw_imgs, norm_imgs, labels in dataset.take(1):
        raw_np = raw_imgs.numpy()
        norm_np = norm_imgs.numpy()
        lbl_np = labels.numpy()

        print("\n--- 1. ANÁLISIS ESTADÍSTICO DE DATOS CRUDOS (RAW) ---")
        print(f"Rango de valores: Min = {np.min(raw_np):.4f} | Max = {np.max(raw_np):.4f}")
        print(f"Media global: {np.mean(raw_np):.4f}")
        print(f"Desviación Estándar global: {np.std(raw_np):.4f}")
        print(f"NaNs encontrados: {np.isnan(raw_np).sum()} (Debería ser 0)")
        print(f"Infinitos encontrados: {np.isinf(raw_np).sum()} (Debería ser 0)")

        print("\n--- 2. ANÁLISIS TRAS NORMALIZACIÓN (Z-Score) ---")
        print(f"Rango de valores: Min = {np.min(norm_np):.4f} | Max = {np.max(norm_np):.4f}")
        print(f"Media global: {np.mean(norm_np):.4f} (Debería ser ~0.0)")
        print(f"Desviación Estándar global: {np.std(norm_np):.4f} (Debería ser ~1.0)")

        print("\n--- 3. BALANCE DE CLASES DEL LOTE ---")
        conteos = np.bincount(lbl_np, minlength=2)
        print(f"Ruido (0): {conteos[0]} muestras | Señal (1): {conteos[1]} muestras")

        # Buscar el primer ejemplo que sea una señal (Onda Gravitacional) para graficar
        signal_indices = np.where(lbl_np == 1)[0]
        if len(signal_indices) > 0:
            idx = signal_indices[0]
            print(f"\nGenerando gráfica comparativa para la muestra {idx} (Etiqueta: SEÑAL)...")
            
            plt.figure(figsize=(16, 6))
            
            # Gráfica Cruda
            plt.subplot(1, 2, 1)
            plt.title("Q-Transform CRUDO (Lo que veía la red antes)")
            plt.imshow(raw_np[idx].squeeze(), aspect='auto', cmap='magma', origin='lower')
            plt.colorbar(label='Amplitud Cruda')
            plt.xlabel("Tiempo")
            plt.ylabel("Frecuencia")

            # Gráfica Normalizada
            plt.subplot(1, 2, 2)
            plt.title("Q-Transform NORMALIZADO (Lo que verá ahora)")
            plt.imshow(norm_np[idx].squeeze(), aspect='auto', cmap='magma', origin='lower')
            plt.colorbar(label='Z-Score (Desviaciones)')
            plt.xlabel("Tiempo")
            plt.ylabel("Frecuencia")

            plt.tight_layout()
            plt.show()
        else:
            print("\nNo se encontró ninguna señal en este lote para graficar. Ejecuta el script de nuevo.")
        break

if __name__ == '__main__':
    main()