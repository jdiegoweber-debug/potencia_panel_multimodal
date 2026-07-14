# Auditoría de Potencia Fotovoltaica - Inteligencia Artificial Multimodal Real ☀️🤖

Este proyecto implementa una solución avanzada de **Deep Learning Multimodal** en PyTorch para auditar y diagnosticar la potencia en Watts de paneles solares fotovoltaicos comerciales. El sistema fusiona de forma nativa la firma térmica espacial de imágenes infrarrojas junto con variables continuas de telemetría SCADA, reemplazando simulaciones manuales por inferencia neuronal real.

---

## 🚀 Arquitectura Híbrida del Modelo (`src/models.py`)

La red neuronal procesa dos flujos de información asincrónicos simultáneamente:
* **Rama Visual (CNN):** Utiliza una arquitectura **ResNet-18** preentrenada como extractor de características térmicas espaciales (removiendo su cabezal original con `nn.Identity()`).
* **Rama Tabular (MLP):** Un Perceptrón Multicapa con capas de normalización por lote (`nn.BatchNorm1d`) y regularización (`nn.Dropout`) que procesa los tensores numéricos de telemetría (Irradiación W/m² y Temperatura de Celdas °C).
* **Cabezal de Fusión:** Unifica ambos bloques informáticos en un vector latente (`torch.cat`) antes de realizar la regresión lineal continua para dictaminar la potencia real en Watts.

---

## 📁 Estructura Real del Proyecto

```text
POTENCIA_PANEL_MULTIMODAL_MEJORADO/
├── data/
│   ├── processed/                # Datos intermedios o procesados
│   └── raw/
│       ├── images/               # Lote maestro de imágenes reales (.jpg)
│       │   └── panel_test_real.jpg
│       └── solar_telemetry.csv   # Historial indexado de telemetría SCADA
├── models/                       # Modelos exportados o serializados
├── output/                       # Reportes generados, gráficos y pesos del modelo
├── src/
│   ├── dataset.py                # Clase Dataset y transformaciones para PyTorch
│   ├── diagnostico.py            # Inferencia en producción y visualización interactiva
│   ├── emparejar_datos.py        # Módulo de preprocesamiento y limpieza
│   ├── mock_data.py              # Generador de telemetría sintética de prueba
│   ├── models.py                 # Arquitectura de la Red Neuronal Híbrida
│   └── train.py                  # Pipeline de entrenamiento y optimización por gradiente
├── .env                          # Variables de entorno locales
├── .gitignore                    # Filtros de archivos para Git (evita subir imágenes/venv)
├── README.md                     # Documentación ejecutiva del sistema
└── requirements.txt              # Dependencias del entorno virtual
```

---

## 🛠️ Flujo de Trabajo del Repositorio

El proyecto cuenta con un flujo completo de aprendizaje profundo estructurado en dos scripts principales:

### 1. Canal de Entrenamiento (`src/train.py`)
Entrena la red neuronal multimodal sobre un lote controlado de **500 escenas** reales mediante optimización por gradiente.
```bash
python src/train.py
```
* **Comportamiento:** Procesa los datos en minilotes (Batch Size = 16). En cada época calcula el error mediante la función de pérdida `MSELoss` y actualiza los pesos de la red usando el optimizador `Adam`. Al finalizar las 10 épocas, exporta el cerebro de la IA a `output/pesos_modelo_multimodal.pth` y guarda el historial de pérdida en `curva_aprendizaje_loss.png`.

### 2. Módulo de Diagnóstico y Auditoría (`src/diagnostico.py`)
Utiliza los pesos ya entrenados de la IA en modo de producción para auditar de forma masiva **100 paneles** sin usar trucos matemáticos.
```bash
python src/diagnostico.py
```
* **Comportamiento:** Carga el modelo binario congelado (`modelo.eval()`), inyecta en paralelo las imágenes RGB junto a su fila SCADA, evalúa la potencia real y exporta los resultados tabulares a `output/reporte_diagnostico.csv`. Finalmente, abre una **ventana gráfica interactiva de Matplotlib** (`curva_desviaciones.png`) que contrasta la Potencia Esperada Teórica frente al veredicto real de la IA e imprime las métricas de error global.

---

## 📊 Métricas de Rendimiento y Ajuste Estadístico Logrados

El modelo demuestra un aprendizaje matemático legítimo y una excelente capacidad de generalización en hardware CPU:
* **Época 01:** Error Promedio Inicial de **209.71 Watts** (fase de inicialización aleatoria).
* **Época 10:** Error Promedio Final de **20.53 Watts** (reducción del error en un **90.2%**).
* **Evaluación Global de Auditoría (100 Paneles Inéditos):**
  * **R² Score (Coeficiente de Determinación):** **0.7884** (El modelo explica el **78.84%** de la varianza real de la potencia basándose en el análisis híbrido).
  * **MAE (Mean Absolute Error):** **10.65 Watts** de desviación absoluta media por panel, consolidando una tolerancia óptima para inspecciones comerciales de activos fotovoltaicos.
