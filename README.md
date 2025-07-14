# Detección y Caracterización de Ondas Gravitacionales mediante Deep Learning: Un Análisis Comparativo de Arquitecturas CNN y Transformer

**Autor:** Javier Camarena Cuartero
**Grado:** Grado en Tecnologías Interactivas
**Fecha:** Junio 2025

---

## 1. Resumen del Proyecto

Este proyecto desarrolla y evalúa sistemas de Deep Learning para la detección de señales de ondas gravitacionales (OG) procedentes de eventos cósmicos como la fusión de agujeros negros. El núcleo del trabajo es un análisis comparativo entre dos arquitecturas de vanguardia: una **Red Neuronal Convolucional 1D (1D-CNN)** y un modelo **Transformer**.

El objetivo es determinar qué arquitectura ofrece un mejor rendimiento (precisión, sensibilidad, velocidad) para identificar las débiles señales "chirp" ocultas en los datos ruidosos de los detectores LIGO/Virgo/KAGRA.

Finalmente, el modelo más eficaz se integrará en un prototipo de aplicación web interactiva que simula la detección en tiempo real y ofrece una estimación de los parámetros astrofísicos de la fuente, haciendo la ciencia de ondas gravitacionales más accesible e intuitiva.

## 2. Objetivos Principales

*   **OE1:** Adquirir y preprocesar datos públicos del Gravitational Wave Open Science Center (GWOSC).
*   **OE2 & OE3:** Implementar, entrenar y evaluar los modelos 1D-CNN y Transformer.
*   **OE4:** Realizar un análisis comparativo riguroso del rendimiento de ambos modelos.
*   **OE5:** Desarrollar un modelo secundario para estimar parámetros básicos de la fuente (ej. masas).
*   **OE6 & OE7:** Diseñar e integrar los modelos en una aplicación web interactiva para la visualización de resultados.

## 3. Estructura del Repositorio

El proyecto está organizado en las siguientes carpetas principales:

```
TFG_GravitationalWaves_AI/
│
├── data/               # Contendrá los datasets brutos y procesados
├── notebooks/          # Cuadernos de Jupyter para exploración y prototipado
├── src/                # Código fuente principal del proyecto
│   ├── data_processing/ # Scripts para descarga y preparación de datos
│   ├── models/          # Definición de las arquitecturas de IA
│   ├── training/        # Scripts para entrenar los modelos
│   ├── evaluation/      # Scripts para evaluar y comparar los modelos
│   └── api/             # Backend (API Flask) para la aplicación web
│
├── frontend/           # Código de la interfaz de usuario web (React/Vue)
├── trained_models/     # Archivos con los pesos de los modelos entrenados
│
├── requirements.txt    # Dependencias de Python
└── README.md           # Este archivo
```

## 4. Cómo Empezar

### Prerrequisitos

*   Python 3.x
*   Git

### Instalación

1.  Clona este repositorio en tu máquina local:
    ```bash
    git clone [URL-de-tu-repositorio]
    cd TFG_GravitationalWaves_AI
    ```

2.  Crea un entorno virtual de Python y actívalo. Esto es muy recomendable para aislar las dependencias del proyecto.
    ```bash
    python -m venv venv
    # En Windows:
    # venv\Scripts\activate
    # En macOS/Linux:
    # source venv/bin/activate
    ```

3.  Instala las dependencias necesarias. El archivo `requirements.txt` se irá actualizando a medida que avance el proyecto.
    ```bash
    pip install -r requirements.txt
    ```

## 5. Uso

El flujo de trabajo del proyecto sigue estos pasos:

1.  **Procesamiento de Datos:** Ejecutar los scripts en `src/data_processing/` para descargar datos de GWOSC y generar los conjuntos de entrenamiento.
2.  **Entrenamiento de Modelos:** Usar los scripts en `src/training/` para entrenar los modelos CNN, Transformer y de estimación de parámetros.
3.  **Evaluación:** Correr el script `src/evaluation/evaluate_models.py` para obtener la comparativa de rendimiento.
4.  **Lanzar la Aplicación Web:**
    *   Iniciar el backend: `python src/api/app.py`
    *   Iniciar el frontend (desde la carpeta `frontend/`): `npm start`

---
*Este documento se actualizará a medida que el proyecto progrese.*