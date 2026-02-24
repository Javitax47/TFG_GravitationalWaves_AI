# src/data_processing/preprocess.py (ACTUALIZADO CON SOLUCIÓN DE GPU EN WORKERS)

# ... (todas las importaciones se mantienen igual) ...
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import numpy as np
import h5py
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import gc
from itertools import repeat
import shutil

from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from pycbc.filter import sigmasq
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBCTimeSeries
from pycbc.waveform import get_td_waveform
import tensorflow as tf

# ... (todas las constantes se mantienen igual) ...
SAMPLE_RATE = 4096
DURATION = 2
N_SAMPLES = SAMPLE_RATE * DURATION
DETECTORS = ["L1", "H1", "V1"]
MASS_MIN_BH, MASS_MAX_BH = 5.0, 50.0
MASS_MIN_NS, MASS_MAX_NS = 1.0, 2.5
SPIN_MIN, SPIN_MAX = -0.8, 0.8
SNR_MIN, SNR_MAX = 8.0, 30.0
P_BBH, P_BNS, P_NSBH = 0.70, 0.20, 0.10

NUM_SAMPLES_TO_GENERATE = 15000
CHUNK_SIZE_FOR_OUTPUT_FILES = 3000
MINI_BATCH_SIZE = 256

RESIZE_ENABLED = True
TARGET_SHAPE = (500, 970)

SKIP_TIMESERIES_GENERATION = False
SKIP_QTRANSFORM_GENERATION = False

script_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
QTRANSFORM_CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'qtransform_chunks_resized')

# ... (Todo el PASO A se mantiene idéntico) ...
def load_real_noise_from_network(detectors, sample_rate):
    """Descarga y carga 128s de datos de ruido para una red de detectores desde GWOSC."""
    print(f"Descargando datos para la red de detectores: {detectors}...")
    noise_data = {}
    gps_start = 1251331200
    gps_end = gps_start + 128

    for det in detectors:
        print(f"  - Obteniendo datos para {det}...")
        try:
            ts = TimeSeries.fetch_open_data(det, start=gps_start, end=gps_end, cache=True)
            ts = ts.resample(sample_rate)
            noise_data[det] = ts.value
            print(f"    -> ¡Éxito! {len(ts)} muestras cargadas para {det}.")
        except Exception as e:
            print(f"    -> Falló la descarga para {det}: {e}. Se omitirá este detector.")

    if not noise_data:
        raise RuntimeError("¡Error Crítico! No se pudo descargar datos para NINGÚN detector.")
    return noise_data

def generate_gw_waveform_from_event_type(event_type, sample_rate):
    """Genera una forma de onda (hp, hc) basada en un tipo de evento."""
    params = {'event_type': event_type}
    if event_type == 'BBH':
        approximant = "IMRPhenomPv2"
        mass1, mass2 = np.random.uniform(MASS_MIN_BH, MASS_MAX_BH), np.random.uniform(MASS_MIN_BH, MASS_MAX_BH)
        spin1z, spin2z = np.random.uniform(SPIN_MIN, SPIN_MAX), np.random.uniform(SPIN_MIN, SPIN_MAX)
        params.update({'mass1': mass1, 'mass2': mass2, 'spin1z': spin1z, 'spin2z': spin2z})
        hp, hc = get_td_waveform(approximant=approximant, mass1=mass1, mass2=mass2,
                                 spin1z=spin1z, spin2z=spin2z,
                                 delta_t=1.0/sample_rate, f_lower=20)
    elif event_type == 'BNS':
        approximant = "IMRPhenomD_NRTidalv2"
        mass1, mass2 = np.random.uniform(MASS_MIN_NS, MASS_MAX_NS), np.random.uniform(MASS_MIN_NS, MASS_MAX_NS)
        lambda1, lambda2 = 450 * (1.375 / mass1)**5, 450 * (1.375 / mass2)**5
        params.update({'mass1': mass1, 'mass2': mass2, 'lambda1': lambda1, 'lambda2': lambda2})
        hp, hc = get_td_waveform(approximant=approximant, mass1=mass1, mass2=mass2,
                                 lambda1=lambda1, lambda2=lambda2,
                                 delta_t=1.0/sample_rate, f_lower=20)
    else: # NSBH
        approximant = "IMRPhenomD_NRTidalv2"
        mass1, mass2 = np.random.uniform(MASS_MIN_BH, MASS_MAX_BH), np.random.uniform(MASS_MIN_NS, MASS_MAX_NS)
        lambda1, lambda2 = 0, 450 * (1.375 / mass2)**5
        params.update({'mass1': mass1, 'mass2': mass2, 'lambda1': lambda1, 'lambda2': lambda2})
        hp, hc = get_td_waveform(approximant=approximant, mass1=mass1, mass2=mass2,
                                 lambda1=lambda1, lambda2=lambda2,
                                 delta_t=1.0/sample_rate, f_lower=20)
    return hp, hc, params

def generate_single_sample(label, noise_dict):
    """Función de trabajo para generar una única muestra (ruido o señal inyectada)."""
    available_detectors = list(noise_dict.keys())
    while True:
        try:
            det_name = np.random.choice(available_detectors)
            noise_chunk = noise_dict[det_name]
            start_idx = np.random.randint(0, len(noise_chunk) - N_SAMPLES)
            noise_segment = noise_chunk[start_idx : start_idx + N_SAMPLES]

            if label == 0:
                ts_noise = TimeSeries(noise_segment, sample_rate=SAMPLE_RATE)
                ts_whitened = ts_noise.taper().whiten(fftlength=2, overlap=1)
                return ts_whitened.value.astype(np.float32), 0, {'event_type': 'noise'}
            else:
                event_type = np.random.choice(['BBH', 'BNS', 'NSBH'], p=[P_BBH, P_BNS, P_NSBH])
                hp, hc, params = generate_gw_waveform_from_event_type(event_type, SAMPLE_RATE)
                
                detector = Detector(det_name)
                signal_full = detector.project_wave(hp, hc, np.random.uniform(0, 2*np.pi),
                                                    np.random.uniform(-np.pi/2, np.pi/2),
                                                    np.random.uniform(0, 2*np.pi))
                
                signal_np = np.asarray(signal_full)
                peak_idx = np.argmax(np.abs(signal_np))
                target_peak_idx = int(1.5 * SAMPLE_RATE)
                start_idx_s = peak_idx - target_peak_idx
                end_idx_s = start_idx_s + N_SAMPLES
                
                signal_slice = signal_np[max(0, start_idx_s):end_idx_s]
                signal_trimmed = np.zeros(N_SAMPLES)
                if start_idx_s >= 0:
                    signal_trimmed[0:len(signal_slice)] = signal_slice
                else:
                    start_offset = -start_idx_s
                    if start_offset + len(signal_slice) > N_SAMPLES:
                        signal_slice = signal_slice[:N_SAMPLES - start_offset]
                    signal_trimmed[start_offset : start_offset + len(signal_slice)] = signal_slice
                
                noise_pycbc = PyCBCTimeSeries(noise_segment, delta_t=1.0/SAMPLE_RATE)
                psd = welch(noise_pycbc, seg_len=int(2 * SAMPLE_RATE), seg_stride=int(1 * SAMPLE_RATE))
                signal_pycbc = PyCBCTimeSeries(signal_trimmed, delta_t=1.0/SAMPLE_RATE)
                optimal_snr_sq = sigmasq(signal_pycbc, psd=psd, low_frequency_cutoff=20.0)
                
                if not np.isfinite(optimal_snr_sq) or optimal_snr_sq <= 0: continue
                
                optimal_snr = np.sqrt(optimal_snr_sq)
                target_snr = np.random.uniform(SNR_MIN, SNR_MAX)
                scaled_signal = signal_pycbc * (target_snr / optimal_snr)

                combined_np = noise_segment + np.asarray(scaled_signal)
                ts_combined = TimeSeries(combined_np, sample_rate=SAMPLE_RATE)
                ts_whitened = ts_combined.taper().whiten(fftlength=2, overlap=1)
                
                params.update({'snr': target_snr, 'detector': det_name})
                return ts_whitened.value.astype(np.float32), 1, params
        except Exception:
            continue

def generate_timeseries_dataset(output_path, num_samples):
    """Paso A: Genera y guarda el dataset de series temporales en un archivo HDF5."""
    print(f"\n--- PASO A: Generando dataset de Series Temporales en '{os.path.basename(output_path)}' ---")
    if os.path.exists(output_path):
        print("El dataset de series temporales ya existe. Saltando este paso.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    network_noise = load_real_noise_from_network(DETECTORS, SAMPLE_RATE)
    
    num_signals = num_samples // 2
    labels_to_generate = [1] * num_signals + [0] * (num_samples - num_signals)
    np.random.shuffle(labels_to_generate)
    
    num_cores = 8
    print(f"\nIniciando generación de {num_samples} muestras en {num_cores} núcleos...")
    worker_func = partial(generate_single_sample, noise_dict=network_noise)
    
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, labels_to_generate),
                             total=num_samples, desc="Generando Series Temporales"))
    
    X, y, parameters_list = zip(*results)
    
    print(f"\nGuardando dataset final en {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('X', data=np.array(X, dtype=np.float32))
        f.create_dataset('y', data=np.array(y, dtype=np.int8))
        f.attrs['sample_rate'] = SAMPLE_RATE
        f.attrs['duration'] = DURATION
        f.create_dataset('parameters', data=[json.dumps(p) for p in parameters_list])
        
    print("--- PASO A COMPLETADO ---")

# ==============================================================================
# --- PASO B: Conversión a Q-Transforms (LÓGICA ACTUALIZADA) ---
# ==============================================================================

# ### INICIO DEL CAMBIO: FUNCIÓN DE INICIALIZACIÓN DEL WORKER ###
def worker_initializer():
    """
    Inicializador para los workers del Pool.
    Esto oculta la GPU para cada proceso worker, forzando a TensorFlow
    a usar únicamente la CPU. Es crucial para evitar la contención de la GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ### FIN DEL CAMBIO ###

def worker_q_transform_and_resize(args):
    """
    Worker que procesa una única muestra, calcula la Q-Transform y la redimensiona.
    Ahora se ejecutará garantizadamente en la CPU.
    """
    time_series_data, sample_rate = args
    try:
        ts = TimeSeries(time_series_data, sample_rate=sample_rate)
        q_transform = ts.q_transform(frange=(30, 1000), qrange=(10, 20))
        
        if RESIZE_ENABLED:
            qt_value = q_transform.value
            qt_tensor = tf.convert_to_tensor(qt_value, dtype=tf.float32)
            qt_tensor_with_channel = tf.expand_dims(qt_tensor, axis=-1)
            resized_tensor = tf.image.resize(qt_tensor_with_channel, TARGET_SHAPE)
            return resized_tensor.numpy().astype(np.float32)
        else:
            return np.expand_dims(q_transform.value, axis=-1).astype(np.float32)

    except Exception as e:
        final_shape = TARGET_SHAPE if RESIZE_ENABLED else (1000, 1940)
        return np.zeros(final_shape + (1,), dtype=np.float32)

def generate_qtransform_in_chunks(input_path, output_dir, base_filename, chunk_size, mini_batch_size):
    """
    Paso B: Carga el dataset de series temporales y lo convierte a Q-Transforms
    REDIMENSIONADAS, usando un Pool de workers que solo ven la CPU.
    """
    print(f"\n--- PASO B: Generando dataset de Q-Transforms redimensionadas ---")
    print(f"Directorio de salida: '{output_dir}'")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directorio de salida anterior '{output_dir}' eliminado.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_cores = 8
    print(f"Iniciando conversión con {num_cores} núcleos (solo CPU) y mini-lotes de {mini_batch_size}...")

    try:
        with h5py.File(input_path, 'r') as f_in:
            data_reader = f_in['X']
            y_labels = f_in['y'][:]
            sample_rate = f_in.attrs['sample_rate']
            num_samples = len(y_labels)
            num_chunks = (num_samples + chunk_size - 1) // chunk_size

            qt_shape = TARGET_SHAPE if RESIZE_ENABLED else (1000, 1940)
            print(f"Forma de salida de las Q-Transforms: {qt_shape + (1,)}")

            for i in tqdm(range(num_chunks), desc="Procesando y Guardando Chunks"):
                start_idx_chunk = i * chunk_size
                end_idx_chunk = min((i + 1) * chunk_size, num_samples)
                current_chunk_size = end_idx_chunk - start_idx_chunk
                
                output_filename = os.path.join(output_dir, f"{base_filename}_part_{i+1}_of_{num_chunks}.hdf5")
                
                with h5py.File(output_filename, 'w') as f_out:
                    x_dset = f_out.create_dataset('X', shape=(current_chunk_size,) + qt_shape + (1,), dtype='float32', compression='gzip')
                    y_dset = f_out.create_dataset('y', data=y_labels[start_idx_chunk:end_idx_chunk], compression='gzip')

                    for j in tqdm(range(0, current_chunk_size, mini_batch_size), desc=f"  Procesando Mini-Lotes Chunk {i+1}", leave=False):
                        start_mini_batch = start_idx_chunk + j
                        end_mini_batch = min(start_mini_batch + mini_batch_size, end_idx_chunk)
                        
                        ts_mini_batch = data_reader[start_mini_batch:end_mini_batch]
                        tasks = zip(ts_mini_batch, repeat(sample_rate))

                        # ### INICIO DEL CAMBIO: AÑADIR EL INICIALIZADOR AL POOL ###
                        with mp.Pool(processes=num_cores, initializer=worker_initializer, maxtasksperchild=1) as pool:
                            results = pool.map(worker_q_transform_and_resize, tasks)
                        # ### FIN DEL CAMBIO ###

                        results_array = np.array(results)
                        write_start_index = j
                        write_end_index = j + len(results)
                        x_dset[write_start_index:write_end_index] = results_array
                        
                        del results, results_array, ts_mini_batch
                        gc.collect()

    except Exception as e:
        print(f"\nOcurrió un error crítico durante la generación de Q-Transforms: {e}")
        raise

    print("\n--- PASO B COMPLETADO ---")

# ==============================================================================
# --- Orquestador Principal ---
# ==============================================================================

def main():
    """Orquestador principal del preprocesamiento."""
    
    # Configurar TensorFlow para que no se apropie de toda la memoria de la GPU en el proceso principal.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Crecimiento de memoria de GPU habilitado para el proceso principal.")
        except RuntimeError as e:
            print(f"Error configurando la memoria de la GPU: {e}")

    timeseries_path = os.path.join(PROCESSED_DATA_DIR, f'dataset_{NUM_SAMPLES_TO_GENERATE}_samples.hdf5')
    qtransform_base_name = f'qtransforms_{NUM_SAMPLES_TO_GENERATE}_samples'

    if not SKIP_TIMESERIES_GENERATION:
        generate_timeseries_dataset(timeseries_path, NUM_SAMPLES_TO_GENERATE)
    else:
        print("--- PASO A: Saltando la generación de series temporales (configurado para saltar) ---")
    
    if not SKIP_QTRANSFORM_GENERATION:
        if not os.path.exists(timeseries_path):
            print(f"Error: El archivo de entrada '{timeseries_path}' no existe.")
        else:
            generate_qtransform_in_chunks(
                input_path=timeseries_path,
                output_dir=QTRANSFORM_CHUNKS_DIR,
                base_filename=qtransform_base_name,
                chunk_size=CHUNK_SIZE_FOR_OUTPUT_FILES,
                mini_batch_size=MINI_BATCH_SIZE
            )
    else:
        print("--- PASO B: Saltando la generación de Q-Transforms (configurado para saltar) ---")

    print("\nPreprocesamiento finalizado con éxito.")

if __name__ == '__main__':
    main()