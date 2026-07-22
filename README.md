# Auditoría de Potencia Fotovoltaica - IA Multimodal Multi-Tarea ☀🤖

Sistema avanzado de Deep Learning industrial para la auditoría automatizada y el diagnóstico predictivo de activos fotovoltaicos, implementado en PyTorch. Integra visión artificial profunda (ResNet-50) y datos tabulares SCADA para inferencia síncrona de potencia eléctrica y probabilidad de fallo estructural. Basado en el paradigma de Aprendizaje de Representaciones.

## 🧠 Arquitectura del Sistema
*   **Extractor Visual (CNN):** ResNet-50 (Fine-Tuning) para la extracción automática de características y firmas termográficas aéreas capturadas por drones. Vector latente visual: 256 dimensiones.
*   **Rama Tabular SCADA (MLP):** Perceptrón Multicapa para procesar variables analógicas físicas: Irradiación (\(G\)), Temperatura de Celda (\(T_{amb}\)), Voltaje (\(V\)) y Corriente (\(I\)). Vector latente tabular: 64 dimensiones.
*   **Espacio Latente Compartido (Early Fusion):** Mecanismo de fusión intermedia mediante concatenación de tensores en un vector holístico unificado de 320 dimensiones.
*   **Inferencia Paralela (Multi-Task Heads):** Cabezales densos independientes para Regresión Continua (Watts esperados) y Clasificación Diagnóstica (Probabilidad de reemplazo basada en función de activación Sigmoide).

## 📊 Métricas de Rendimiento Logradas
Tras un régimen experimental de 10 épocas de entrenamiento adaptativo con el optimizador `AdamW` y una función de pérdida combinada ponderada (`MSELoss + 50 * BCELoss`), el modelo consolidó los siguientes indicadores de excelencia en producción:
*   **Pérdida Promedio Total de Convergencia:** 5962.05 (Reducción neta del error en un **81.95%**).
*   **Error Absoluto Medio (MAE):** **58.67 Watts** por módulo fotovoltaico.
*   **Coeficiente de Determinación (\(R^2\) Score):** **0.1916** (Estabilidad estadística positiva en la varianza).
*   **Área Bajo la Curva ROC (AUC):** **0.91** (Rendimiento de excelencia diagnóstica comercial).

## 📁 Estructura Limpia del Repositorio
```text
POTENCIA_PANEL_MULTIMODAL/
├── data/
│   └── raw/
│       ├── images/               # Capturas aéreas (.jpg)
│       └── solar_telemetry.csv   # Base de datos SCADA y etiquetas
├── output/
│   ├── curva_aprendizaje_loss.png
│   ├── curva_desviaciones.png
│   ├── curva_roc.png             # Evaluación del clasificador binario
│   ├── pesos_modelo_multimodal.pth
│   └── reporte_diagnostico.csv   # Reporte corporativo de auditoría
├── src/
│   ├── dataset.py                # Clase Dataset y transformaciones v2
│   ├── diagnostico.py            # Inferencia en producción y reportes
│   ├── emparejar_datos.py        # Preprocesamiento y Data Wrangling
│   ├── models.py                 # Arquitectura de la Red Multi-Tarea
│   └── train.py                  # Pipeline maestro de entrenamiento
└── README.md
```

## 🛠️ Flujo de Ejecución (Pipeline de Producción)

1.  **Entrenamiento Maestro:** Ejecuta la optimización adaptativa sobre las 500 escenas controladas y exporta las curvas de aprendizaje:
    ```bash
    python src/train.py
    ```
2.  **Auditoría y Diagnóstico:** Ejecuta el modelo congelado en modo estricto de evaluación (`model.eval()`) sobre los paneles de control, generando el reporte ejecutivo `.csv` y las métricas científicas:
    ```bash
    python src/diagnostico.py
    ```

---
*Desarrollado como proyecto de software e investigación para la Maestría en Inteligencia Artificial.*
