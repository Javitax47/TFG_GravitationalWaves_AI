# src/evaluation/benchmark_master.py

import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, mean_absolute_error

# --- CONFIGURACIÓN MLOPS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={conda_prefix}'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.model_factory import build_grid_model

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'evaluation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SHAPE_2D = (256, 512, 1)
TARGET_SHAPE_1D = (8192, 1)

# ==============================================================================
# PARSER UNIVERSAL DE EVALUACIÓN (INCLUYE SNR)
# ==============================================================================
def get_benchmark_parse_fn(dimension, tasks):
    def parse_fn(example):
        feature_desc = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'event_type': tf.io.FixedLenFeature([], tf.int64),
            'mass1': tf.io.FixedLenFeature([], tf.float32),
            'mass2': tf.io.FixedLenFeature([], tf.float32),
            'snr': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0)
        }
        if dimension == '2D':
            feature_desc['image'] = tf.io.FixedLenFeature([], tf.string)
        else:
            feature_desc['signal'] = tf.io.FixedLenFeature([], tf.string)

        parsed = tf.io.parse_single_example(example, feature_desc)

        # 1. Entrada y Normalización
        if dimension == '2D':
            inputs = tf.io.parse_tensor(parsed['image'], out_type=tf.float32)
            inputs.set_shape(TARGET_SHAPE_2D)
            inputs = tf.image.per_image_standardization(inputs)
        else:
            inputs = tf.io.parse_tensor(parsed['signal'], out_type=tf.float32)
            inputs.set_shape(TARGET_SHAPE_1D)
            mean = tf.math.reduce_mean(inputs)
            std = tf.math.reduce_std(inputs)
            inputs = (inputs - mean) / tf.maximum(std, 1e-6)

        # 2. Salidas dinámicas
        outputs = {}
        if 'detection' in tasks:
            outputs['detection'] = tf.cast(parsed['label'], tf.float32)
        if 'classification' in tasks:
            outputs['classification'] = tf.cast(parsed['event_type'], tf.int64)
        if 'masses' in tasks:
            m1, m2 = parsed['mass1'], parsed['mass2']
            outputs['masses'] = tf.cast(tf.stack([tf.maximum(m1, m2), tf.minimum(m1, m2)]), tf.float32)

        return inputs, outputs, parsed['snr']
    return parse_fn

def get_test_dataset(dimension, tasks, batch_size):
    data_dir = 'tfrecord_chunks' if dimension == '2D' else 'tfrecord_1d_chunks'
    input_path = os.path.join(PROJECT_ROOT, f'data/processed/{data_dir}')
    
    all_files = sorted(glob.glob(os.path.join(input_path, '*.tfrecord')))
    if not all_files: return None
    
    np.random.seed(42)
    np.random.shuffle(all_files)
    test_files = all_files[int(0.8 * len(all_files)):]
    
    ds = tf.data.TFRecordDataset(test_files)
    ds = ds.map(get_benchmark_parse_fn(dimension, tasks), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# MOTOR DE EVALUACIÓN MULTI-TASK Y K-FOLD
# ==============================================================================
def evaluate_experiment(architecture, tasks):
    exp_suffix = f"{architecture.lower()}_{'_'.join(tasks)}"
    dimension = "2D" if "2D" in architecture else "1D"
    
    print(f"\n{'='*80}")
    print(f"🚀 BENCHMARK: {architecture} | Tareas: {tasks} 🚀")
    print(f"{'='*80}")

    hp_path = os.path.join(PROJECT_ROOT, f'trained_models/best_hps_{exp_suffix}.json')
    weights_pattern = os.path.join(PROJECT_ROOT, f'trained_models/{exp_suffix}_fold_*.keras')
    
    fold_weights = sorted(glob.glob(weights_pattern))
    if not fold_weights or not os.path.exists(hp_path):
        print(f"⚠️ Saltando {exp_suffix}: Modelos o HPs no encontrados.")
        return

    with open(hp_path, 'r') as f: hps = json.load(f)
    ordered_tasks =[t for t in ['detection', 'classification', 'masses'] if t in tasks]
    
    # --- AJUSTE DE LOTE DINÁMICO (PREVENCIÓN OOM PARA ViT) ---
    current_batch_size = 16 if 'ViT' in architecture else 32

    # Instanciamos la arquitectura
    input_shape = TARGET_SHAPE_2D if dimension == '2D' else TARGET_SHAPE_1D
    model = build_grid_model(architecture, ordered_tasks, input_shape, hps_dict=hps)

    test_ds = get_test_dataset(dimension, ordered_tasks, current_batch_size)
    if not test_ds:
        print("❌ Error: No se encontraron datos de Test.")
        return

    # 1. Extrayendo Ground Truth de forma segura en RAM
    print("📥 Extrayendo Ground Truth del Test Set...")
    y_true_dict = {t: [] for t in ordered_tasks}
    snrs_list =[]
    
    # Hacemos una única pasada rápida para extraer las respuestas reales
    for imgs, lbls, snrs in test_ds:
        snrs_list.extend(snrs.numpy())
        for task in ordered_tasks:
            y_true_dict[task].extend(lbls[task].numpy())
            
    snr_array = np.array(snrs_list)
    for t in ordered_tasks: y_true_dict[t] = np.array(y_true_dict[t])

    # 2. Inferencia Ensemble VRAM-Safe
    mc_passes = 20 if 'masses' in tasks else 1
    use_mc = 'masses' in tasks
    
    all_preds_dict = {t:[] for t in ordered_tasks}
    
    print(f"🧠 Iniciando Inferencia Ensemble ({len(fold_weights)} Folds | {mc_passes} MC Passes)...")
    start_time = time.time()
    
    for i, w_path in enumerate(fold_weights):
        print(f"   -> Evaluando Fold {i+1}...")
        model.load_weights(w_path)
        
        for _ in range(mc_passes):
            pass_preds = {t:[] for t in ordered_tasks}
            
            # ITERAMOS DESDE DISCO. NO GUARDAMOS IMÁGENES EN LISTAS. ZERO MEMORY LEAKS.
            for imgs, _, _ in test_ds:
                preds = model(imgs, training=use_mc)
                if len(ordered_tasks) == 1: preds = [preds]
                
                for t_idx, task in enumerate(ordered_tasks):
                    pass_preds[task].extend(preds[t_idx].numpy())
            
            for task in ordered_tasks:
                all_preds_dict[task].append(pass_preds[task])

    total_time = time.time() - start_time
    ms_per_img = (total_time / (len(snr_array) * len(fold_weights) * mc_passes)) * 1000
    print(f"⏱️ Inferencia completada. Latencia: {ms_per_img:.2f} ms/imagen")

    # 3. Consenso de la IA
    final_preds = {}
    uncertainties = {}
    for t in ordered_tasks:
        preds_array = np.array(all_preds_dict[t])
        final_preds[t] = np.mean(preds_array, axis=0)
        uncertainties[t] = np.std(preds_array, axis=0)

    # 4. Evaluación por Rangos SNR
    snr_bounds = {
        "Global": {"min": 0.0, "max": 999.0},
        "Low SNR (8-15)": {"min": 8.0, "max": 15.0},
        "Mid SNR (15-22)": {"min": 15.0, "max": 22.0},
        "High SNR (22-30)": {"min": 22.0, "max": 30.0}
    }
    
    plot_dir = os.path.join(RESULTS_DIR, exp_suffix)
    os.makedirs(plot_dir, exist_ok=True)
    
    if 'detection' in tasks: plt.figure(figsize=(8,6))

    for test_name, bounds in snr_bounds.items():
        print(f"\n--- 📊 Resultados: {test_name} ---")
        
        is_noise = (y_true_dict['detection'] == 0) if 'detection' in tasks else (snr_array < 0)
        mask = is_noise | ((~is_noise) & (snr_array >= bounds['min']) & (snr_array < bounds['max']))
        
        if 'detection' in tasks:
            y_t = y_true_dict['detection'][mask]
            y_p = final_preds['detection'][mask]
            
            if len(np.unique(y_t)) > 1:
                fpr, tpr, _ = roc_curve(y_t, y_p)
                roc_auc = auc(fpr, tpr)
                y_p_class = (y_p > 0.5).astype(int)
                report = classification_report(y_t, y_p_class, output_dict=True, zero_division=0)
                print(f"  [Detección] AUC: {roc_auc:.4f} | Recall: {report.get('1.0', report.get('1', {})).get('recall', 0):.4f}")
                plt.plot(fpr, tpr, lw=2, label=f'{test_name} (AUC = {roc_auc:.3f})')
                
                if test_name == "Global":
                    cm_fig = plt.figure(figsize=(5,4))
                    sns.heatmap(confusion_matrix(y_t, y_p_class), annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusión Detección - {architecture}')
                    plt.savefig(os.path.join(plot_dir, 'cm_detection.png'), bbox_inches='tight')
                    plt.close(cm_fig)
            else:
                print("[Detección] Faltan clases en este rango SNR para calcular AUC.")

        if 'masses' in tasks:
            signal_mask = (~is_noise) & mask
            y_t = y_true_dict['masses'][signal_mask]
            y_p = final_preds['masses'][signal_mask]
            
            if len(y_t) > 0:
                mae_m1 = mean_absolute_error(y_t[:, 0], y_p[:, 0])
                mae_m2 = mean_absolute_error(y_t[:, 1], y_p[:, 1])
                print(f"[Masas] MAE Total: {(mae_m1+mae_m2)/2:.2f} M☉ (M1: {mae_m1:.2f}, M2: {mae_m2:.2f})")
                
                if test_name == "Global":
                    y_std = uncertainties['masses'][signal_mask]
                    sc_fig = plt.figure(figsize=(10,8))
                    idx = np.random.choice(len(y_t), min(200, len(y_t)), replace=False)
                    plt.errorbar(y_t[idx,0], y_p[idx,0], yerr=y_std[idx,0], fmt='o', alpha=0.7, label='M1')
                    plt.errorbar(y_t[idx,1], y_p[idx,1], yerr=y_std[idx,1], fmt='o', alpha=0.7, label='M2')
                    plt.plot([0,50], [0,50], 'k--')
                    plt.title(f'Estimación Bayesiana de Masas - {architecture}')
                    plt.xlabel('Masa Real ($M_\odot$)'); plt.ylabel('Predicción ($M_\odot$)')
                    plt.legend()
                    plt.savefig(os.path.join(plot_dir, 'masses_scatter.png'), bbox_inches='tight')
                    plt.close(sc_fig)

    if 'detection' in tasks:
        plt.plot([0, 1],[0, 1], 'k--', lw=1)
        plt.title(f'Curvas ROC por SNR - {architecture}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, 'roc_curves.png'), bbox_inches='tight')
        plt.close()
        
    print(f"\n✅ Resultados guardados en {plot_dir}")

# ==============================================================================
# ENTRY POINT (CLI)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ALL')
    parser.add_argument('--tasks', type=str, default='ALL')
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    ALL_ARCHS =["CNN-2D", "ViT-2D", "CNN-1D", "ViT-1D"]
    ALL_TASKS = [
        ["detection"], ["masses"],["classification"],
        ["detection", "classification"], ["detection", "masses"], ["classification", "masses"],["detection", "classification", "masses"]
    ]

    target_archs = ALL_ARCHS if args.arch == 'ALL' else [args.arch]
    target_tasks = ALL_TASKS if args.tasks == 'ALL' else [[t.strip() for t in args.tasks.split(',')]]

    for arch in target_archs:
        for t_list in target_tasks:
            evaluate_experiment(arch, t_list)
            
    print("\n🏁 BENCHMARK GLOBAL COMPLETADO 🏁")