
import os
import sys
import time
import multiprocessing as mp

# Desactivar logs innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_batch_size(batch_size, input_shape=(256, 512, 1)):
    """Ejecuta un test de estrés aislando la GPU en un proceso único."""
    
    # --- SOLUCIÓN AL ERROR DE LIBDEVICE EN EL NUEVO PROCESO ---
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={conda_prefix}'
    
    # Configuraciones de estabilidad
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
    # Importar dentro del proceso para evitar inicialización prematura
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.models.cnn_classifier import build_resnet18_classifier

    print(f"\n--- Probando Batch Size: {batch_size} ---")
    
    try:
        # 1. Crear modelo SOTA (ResNet-18 SOTA Completa)
        model = build_resnet18_classifier(input_shape=input_shape, use_checkpointing=False)
        # Desactivamos XLA aquí también por si acaso
        model.compile(optimizer='adam', loss='binary_crossentropy', jit_compile=False)

        # 2. Generar datos falsos (ruido aleatorio) en la CPU
        dummy_x = tf.random.normal((batch_size, *input_shape))
        dummy_y = tf.random.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.float32)
        
        # 3. Warmup (calentamiento) - No se mide
        print("  Realizando Warmup en GPU...")
        model.train_on_batch(dummy_x, dummy_y)

        # 4. Prueba de rendimiento
        print("  Midiendo rendimiento (10 iteraciones)...")
        start_time = time.time()
        for _ in range(10):
            model.train_on_batch(dummy_x, dummy_y)
        end_time = time.time()
        
        # 5. Extraer métricas
        avg_time_per_batch = (end_time - start_time) / 10.0
        ms_per_image = (avg_time_per_batch / batch_size) * 1000 
        
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        peak_vram_mb = mem_info['peak'] / (1024 ** 2)

        print(f"  [ÉXITO] VRAM Pico: {peak_vram_mb:.2f} MB | Tiempo/Imagen: {ms_per_image:.2f} ms")
        return True, peak_vram_mb, ms_per_image

    except Exception as e:
        # AHORA SÍ imprimimos el error real
        print(f"  [FALLO] El test se detuvo con Batch Size {batch_size}.")
        print(f"  Detalle del error: {str(e)[:200]}...") # Imprime los primeros 200 caracteres del error
        return False, 0.0, 0.0

if __name__ == '__main__':
    # Lista de tamaños de lote a probar
    batch_sizes =[2, 4, 8, 12, 16, 24, 32]
    
    print("="*60)
    print("INICIANDO PROFILER DE HARDWARE SOTA (ResNet-18)")
    print("="*60)
    
    results =[]
    
    # Usamos 'spawn' para garantizar que TF muere y libera la VRAM al 100% en cada loop
    mp.set_start_method('spawn', force=True)
    
    for bs in batch_sizes:
        # Lanzamos la prueba en un proceso aislado
        with mp.Pool(1) as p:
            success, vram, speed = p.apply(test_batch_size, args=(bs,))
            results.append((bs, success, vram, speed))
            
        if not success:
            print("\nLímite físico alcanzado. Deteniendo pruebas superiores.")
            break
            
    # Imprimir Tabla de Diagnóstico
    print("\n\n" + "="*60)
    print("RESULTADOS DEL DIAGNÓSTICO PARA RTX 3060 (6GB)")
    print("="*60)
    print(f"{'Batch Size':<12} | {'Estado':<10} | {'VRAM Pico (MB)':<15} | {'Velocidad (ms/img)':<15}")
    print("-" * 60)
    for bs, success, vram, speed in results:
        estado = "PASA" if success else "FALLA (OOM)"
        vram_str = f"{vram:.1f}" if success else "N/A"
        speed_str = f"{speed:.2f}" if success else "N/A"
        print(f"{bs:<12} | {estado:<10} | {vram_str:<15} | {speed_str:<15}")
    print("="*60)