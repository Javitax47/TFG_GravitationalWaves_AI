import os
import sys
import argparse
from gwpy.timeseries import TimeSeries

def download_gw_data(detector, start_time, end_time, sample_rate=4096, output_dir="data/raw"):
    """
    Descarga datos de ondas gravitacionales de un detector específico para un
    intervalo de tiempo dado y los guarda en un archivo.
    (Esta función no cambia)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    duration = end_time - start_time
    print(f"\nIniciando descarga de {duration} segundos de datos del detector {detector}...")
    print(f"Intervalo de tiempo GPS: {start_time} a {end_time}")
    
    try:
        data = TimeSeries.fetch_open_data(detector, start_time, end_time, sample_rate=sample_rate, verbose=True)
        output_filename = f"{detector}-{start_time}-{duration}.hdf5"
        output_path = os.path.join(output_dir, output_filename)
        data.write(output_path, format='hdf5')
        print(f"\n¡Descarga completada con éxito!")
        print(f"Datos guardados en: {output_path}")
        
    except Exception as e:
        print(f"\nOcurrió un error durante la descarga o el guardado de datos: {e}")

def run_interactive_mode():
    """
    Ejecuta el script en modo interactivo, pidiendo al usuario la información necesaria.
    """
    print("--- Modo Interactivo de Descarga de Datos ---")
    print("Por favor, proporciona la siguiente información:")

    # Pedir detector
    while True:
        detector = input("Introduce el código del detector (ej. 'H1', 'L1'): ").strip().upper()
        if detector:
            break
        print("El detector no puede estar vacío.")

    # Pedir tiempo de inicio
    while True:
        try:
            start_time_str = input("Introduce el tiempo GPS de inicio (ej. 1126259446): ").strip()
            start_time = int(start_time_str)
            break
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número entero para el tiempo.")

    # Pedir tiempo de fin
    while True:
        try:
            end_time_str = input(f"Introduce el tiempo GPS de fin (debe ser mayor que {start_time}): ").strip()
            end_time = int(end_time_str)
            if end_time > start_time:
                break
            else:
                print("Error: El tiempo de fin debe ser estrictamente mayor que el tiempo de inicio.")
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número entero para el tiempo.")

    # Llamar a la función principal con los datos recogidos
    download_gw_data(detector, start_time, end_time)


def main():
    """
    Función principal que decide si ejecutar en modo argumentos o modo interactivo.
    """
    # Si se han pasado argumentos en la línea de comandos (además del nombre del script)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Descarga datos de ondas gravitacionales desde GWOSC."
        )
        parser.add_argument("--detector", type=str, required=True, help="Detector a usar (ej. 'H1', 'L1').")
        parser.add_argument("--start_time", type=int, required=True, help="Tiempo GPS de inicio.")
        parser.add_argument("--end_time", type=int, required=True, help="Tiempo GPS de fin.")
        parser.add_argument("--sample_rate", type=int, default=4096, help="Frecuencia de muestreo (Hz).")
        parser.add_argument("--output_dir", type=str, default="data/raw", help="Directorio para guardar los archivos.")
        args = parser.parse_args()
        
        download_gw_data(
            detector=args.detector,
            start_time=args.start_time,
            end_time=args.end_time,
            sample_rate=args.sample_rate,
            output_dir=args.output_dir
        )
    else:
        # Si no se han pasado argumentos, iniciar el modo interactivo
        run_interactive_mode()

if __name__ == "__main__":
    main()