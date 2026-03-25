Detección y Caracterización de Ondas Gravitacionales mediante Deep Learning: Un Análisis Comparativo de Arquitecturas CNN y Transformer

Autor: Javier Camarena Cuartero
Grado: Grado en Tecnologías Interactivas
Fecha: Junio 2025

1. Resumen del Proyecto

Este proyecto desarrolla y evalúa un pipeline de Deep Learning de nivel profesional para la detección de señales de ondas gravitacionales (OG). Estas señales, procedentes de eventos cósmicos cataclísmicos como la fusión de agujeros negros, se extraen de datos públicos de los observatorios LIGO/Virgo.

El núcleo del trabajo es un análisis comparativo riguroso entre dos paradigmas de la Inteligencia Artificial moderna:

Red Neuronal Convolucional (CNN): Se implementa una arquitectura avanzada ResNet (Red Residual) que opera sobre imágenes 2D. Estas imágenes son Q-Transforms, una representación tiempo-frecuencia de los datos optimizada para señales de "chirp".

Modelo Transformer: Una arquitectura de vanguardia, basada en mecanismos de atención, que se adaptará para la misma tarea de clasificación de espectrogramas.

El objetivo es determinar qué arquitectura ofrece un mejor rendimiento (precisión, sensibilidad, velocidad de inferencia) para esta tarea de clasificación de señales con baja relación señal-ruido. Finalmente, el modelo más eficaz se integrará en un prototipo de aplicación web interactiva que simula la detección en tiempo real y ofrece una estimación de los parámetros astrofísicos de la fuente.

2. Objetivos Principales

OE1: Desarrollar un pipeline robusto para generar datasets de entrenamiento, inyectando formas de onda simuladas en ruido real del Gravitational Wave Open Science Center (GWOSC).

OE2 & OE3: Implementar, entrenar y optimizar los modelos ResNet (CNN) y Transformer. Esto incluye el ajuste automático de hiperparámetros con KerasTuner.

OE4: Realizar un análisis comparativo riguroso del rendimiento de ambos modelos utilizando métricas estándar.

OE5: Desarrollar un modelo secundario para estimar parámetros básicos de la fuente (ej. masas).

OE6 & OE7: Diseñar e integrar los modelos en una aplicación web interactiva para la visualización de resultados.

3. Estructura del Repositorio

El proyecto está organizado siguiendo las mejores prácticas para un proyecto de Machine Learning escalable:

Generated bash
TFG_GravitationalWaves_AI/
│
├── data/ # Contendrá los datasets brutos y procesados
│ └── processed/ # Datasets listos para el entrenamiento (HDF5)
│
├── notebooks/ # Cuadernos Jupyter para exploración y prototipado inicial.
│
├── src/ # Código fuente principal del proyecto
│ ├── data_processing/ # Scripts para la generación de datasets (preprocess.py)
│ ├── models/ # Definición de las arquitecturas de IA (cnn_classifier.py, etc.)
│ ├── training/ # Scripts para entrenar los modelos (train_cnn.py, etc.)
│ ├── evaluation/ # Scripts para evaluar y comparar los modelos
│ └── api/ # Backend (API Flask) para la aplicación web
│
├── frontend/ # Código de la interfaz de usuario web (React/Vue)
├── trained_models/ # Archivos con los pesos de los modelos (.keras) y configuraciones (.json)
├── keras_tuner/ # Directorio generado por KerasTuner para guardar los resultados de la búsqueda.
│
├── requirements.txt # Dependencias de Python del proyecto.
└── README.md # Este archivo.

4. Cómo Empezar
Prerrequisitos

Sistema Operativo:

Linux (Nativo, recomendado)

macOS (Nativo)

Windows 10/11 exclusivamente a través de WSL 2 (Subsistema de Windows para Linux).

¡Atención! Entorno de Ejecución Requerido

Este proyecto está diseñado y optimizado para ejecutarse en un entorno Linux nativo o en Windows a través de WSL 2.

No es compatible con la ejecución nativa en PowerShell o CMD de Windows. Dependencias clave como lalsuite y, más importante aún, el rendimiento óptimo del pipeline de carga de datos para la GPU, solo se garantizan en un sistema de archivos y un entorno tipo Unix.

Si eres usuario de Windows, es obligatorio instalar WSL 2 antes de continuar. Sigue la guía oficial de Microsoft para instalar WSL 2.

Python 3.9+ (gestionado vía Conda)

Git

Conda / Miniconda (Requerido para gestionar el entorno).

(Opcional pero muy recomendado) Una GPU NVIDIA compatible con CUDA para un entrenamiento rápido.

Instalación

Nota: Todos los siguientes comandos deben ser ejecutados dentro de una terminal de Linux o de WSL 2. No funcionarán en PowerShell o CMD.

Clona el repositorio (dentro de WSL, no en /mnt/c):
Es crucial que el proyecto resida en el sistema de archivos nativo de Linux para un rendimiento óptimo.

Generated bash
# Navega a tu directorio de usuario de Linux
cd ~
# Clona el proyecto aquí
git clone [URL-de-tu-repositorio]
cd TFG_GravitationalWaves_AI
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Crea y activa el entorno de Conda:

Generated bash
# Crear un nuevo entorno llamado 'tfg-env' con Python 3.9
conda create -n tfg-env python=3.9 -y
conda activate tfg-env
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Instala todas las dependencias:
El siguiente comando leerá el fichero requirements.txt e instalará todas las librerías con las versiones correctas y compatibles.

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
5. Flujo de Trabajo

El pipeline del proyecto está diseñado para ser ejecutado en secuencia.

Paso 1: Generación de Datasets

Este es el paso más largo. El script preprocess.py se encarga de todo.

Abre el archivo src/data_processing/preprocess.py y configura las constantes NUM_SAMPLES_TO_GENERATE y CHUNK_SIZE_QT según tus necesidades y la capacidad de tu hardware. Se recomienda empezar con 10,000 muestras.

Ejecuta el script desde la raíz del proyecto:

Generated bash
python -m src.data_processing.preprocess
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Este script realizará dos sub-tareas:

Paso A: Creará un archivo dataset_[N]_samples.hdf5 con las series temporales.

Paso B: Creará un archivo qtransforms_[N].hdf5 con las imágenes listas para el entrenamiento. Este es el archivo que usarán los scripts de entrenamiento.

Paso 2: Entrenamiento y Optimización de la CNN (ResNet)

El script train_cnn.py tiene dos modos de operación, controlados por la variable MODE dentro del archivo.

Modo de Sintonización (Opcional pero recomendado):

Edita src/training/train_cnn.py y establece MODE = 'tune'.

Ejecuta: python -m src.training.train_cnn

Este proceso buscará la mejor combinación de hiperparámetros y la guardará en trained_models/best_hyperparameters.json.

Modo de Entrenamiento Final:

Edita src/training/train_cnn.py y establece MODE = 'train'.

Ejecuta: python -m src.training.train_cnn

Este proceso cargará la mejor configuración (si existe) y entrenará el modelo final, guardándolo en trained_models/resnet18_final.keras.

Paso 3: Lanzar la Aplicación Web (Futuro)

Iniciar el backend:

Generated bash
python -m src.api.app
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Iniciar el frontend (desde la carpeta frontend/):

Generated b
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
B
IGNORE_WHEN_COPYING_END