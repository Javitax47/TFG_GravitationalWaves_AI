#!/bin/bash

echo "================================================================="
echo "🌌 GW-AI VISUALIZER: ORQUESTADOR MAESTRO DEL GRID (SOTA) 🌌"
echo "================================================================="

# Función antibalas (Self-Healing)
run_robustly() {
    local mode=$1
    export EXECUTION_MODE=$mode
    
    echo ""
    echo "--------------------------------------------------"
    echo " Empezando fase: ${mode^^} | Arch: $ARCHITECTURE | Tareas: $TASKS"
    echo "--------------------------------------------------"
    
    while true; do
        python -m src.training.train_master
        EXIT_CODE=$?
        
        # 0 = Éxito, 130 = Interrupción manual (Ctrl+C)
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Fase ${mode^^} completada con éxito."
            break
        elif [ $EXIT_CODE -eq 130 ]; then
            echo "🛑 Ejecución pausada por el usuario (Ctrl+C)."
            exit 130
        else
            echo "⚠️ Crasheo detectado (Posible OOM). La VRAM se está purgando..."
            echo "🔄 Reiniciando en 5 segundos desde el último Checkpoint/Trial..."
            sleep 5
        fi
    done
}

# Función para ejecutar un experimento completo (Tune + Train)
run_experiment() {
    export ARCHITECTURE=$1
    export TASKS=$2
    
    # Deducir dimensión automáticamente
    if [[ "$ARCHITECTURE" == *"2D"* ]]; then
        export DIMENSION="2D"
    else
        export DIMENSION="1D"
    fi

    echo ""
    echo "================================================================="
    echo "🚀 NUEVO EXPERIMENTO: $ARCHITECTURE | DIM: $DIMENSION | TAREAS: $TASKS"
    echo "================================================================="

    # 1. Fase de Búsqueda de Hiperparámetros (Tuning)
    run_robustly "tune"
    
    # 2. Fase de Entrenamiento Definitivo (K-Fold)
    run_robustly "train"
}

# ==============================================================================
# MODO DE EJECUCIÓN (HÍBRIDO)
# ==============================================================================

# Si el usuario pasa argumentos, ejecutamos SOLO ese modelo (Ej: ./run_grid.sh "CNN-2D" "detection")
if [ "$#" -eq 2 ]; then
    run_experiment "$1" "$2"
    echo "🎉 Experimento individual finalizado."
    exit 0
fi

# Si no hay argumentos, ejecutamos EL GRID COMPLETO DE 28 MODELOS
echo "⚠️ No se pasaron argumentos. Iniciando el Grid Completo (28 Modelos)..."
echo "⏳ Esto tomará múltiples días/semanas. Pulsa Ctrl+C en cualquier momento para pausar."
sleep 3

# Definición de la matriz
BACKBONES=("CNN-2D" "ViT-2D" "CNN-1D" "ViT-1D")
TASK_COMBINATIONS=(
    "detection" 
    "classification" 
    "masses" 
    "detection,classification" 
    "detection,masses" 
    "classification,masses" 
    "detection,classification,masses"
)

# Bucle anidado que recorre la matriz
for arch in "${BACKBONES[@]}"; do
    for tasks in "${TASK_COMBINATIONS[@]}"; do
        run_experiment "$arch" "$tasks"
    done
done

echo "🏆 ¡CAMPAÑA DEL GRID COMPLETADA MAGISTRALMENTE! 🏆"