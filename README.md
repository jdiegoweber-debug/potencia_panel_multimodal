# Sistema Multimodal de Diagnóstico Fotovoltaico con Deep Learning ☀️🛸

Este proyecto implementa un sistema inteligente de mantenimiento predictivo para parques solares fotovoltaicos utilizando redes neuronales híbridas multimodales en **PyTorch**. Siguiendo los principios de *Representation Learning* descritos por Bengio, el sistema integra de forma asincrónica imágenes cenitales tomadas por drones y telemetría de sensores eléctricos SCADA para resolver dos tareas críticas en paralelo.

## 🧠 Arquitectura Híbrida del Sistema

El núcleo del software está diseñado para que convivan dos enfoques metodológicos complementarios compartiendo el mismo pipeline de datos:

1. **Monitoreo Continuo (Regresión):** Estima el rendimiento térmico diario calculando la brecha entre la potencia ideal física y los Watts reales generados (`train.py`).
2. **Sistema de Alerta Temprana (Clasificación Binaria):** Un clasificador probabilístico que evalúa el estado del hardware y dictamina si el panel solar conserva su vida útil ("Queda" / Clase 0) o si requiere sustitución física urgente ("No Queda" / Clase 1) mediante el análisis de su curva ROC y matriz de confusión (`train_clasificacion.py`).

```text
mi_proyecto_multimodal/
│
├── data/
│   └── raw/
│       ├── images/            # Almacenamiento de capturas de drones
│       └── solar_telemetry.csv # Registros SCADA indexados con las fotos
│
├── src/
│   ├── dataset.py             # Clase CustomDataset para carga de datos PyTorch
│   ├── models.py              # Arquitecturas convolucionales y densas hibridadas
│   ├── mock_data.py           # Script de simulación y generación de 20.000 muestras
│   └── train_clasificacion.py # Pipeline de entrenamiento, validación y gráficos
│
├── output/                    # Exportación física automática de resultados
│   ├── curva_roc.png          # Rendimiento probabilístico (AUC = 1.00)
│   ├── matriz_confusion.png   # Cuadrantes de aciertos (Verdaderos Positivos/Negativos)
│   └── pesos_modelo_clasificacion.pth # Coeficientes neuronales optimizados
│
├── .gitignore                 # Configuración para omitir el entorno virtual
└── requirements.txt           # Librerías y dependencias necesarias
```

## 🛠️ Instalación y Configuración

Siga estos pasos para replicar el entorno de desarrollo aislado e independiente en su máquina local:

1. **Clonar el repositorio y situarse en el directorio raíz:**
   ```bash
   cd potencia_panel_multimodal_modelos_Complejos
   ```

2. **Crear e inicializar el entorno virtual (.venv):**
   ```bash
   python -m venv .venv
   ```
   * En Windows (PowerShell):
     ```bash
     .\.venv\Scripts\Activate.ps1
     ```

3. **Instalar el bloque de dependencias requeridas:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Guía de Ejecución

El pipeline se ejecuta de forma ordenada mediante la consola de comandos respetando la siguiente secuencia lógica:

### Paso 1: Generación del Dataset Sintético Masivo
Antes de entrenar las redes convolucionales, ejecute el script encargado de generar las 20.000 muestras tabulares y las matrices de imágenes base sincronizadas:
```bash
python src/mock_data.py
```

### Paso 2: Lanzamiento del Entrenamiento de Clasificación Binaria
Para entrenar el detector "Queda / No Queda", calcular la sensibilidad probabilística y desplegar los gráficos interactivos en su pantalla, ejecute:
```bash
python src/train_clasificacion.py
```

## 📊 Caracterización Estadística del Detector

El sistema evalúa de forma automática su capacidad de generalización sobre un subconjunto independiente de validación (20% de las muestras totales), exportando los resultados en la carpeta `output/`:

* **Curva ROC (AUC = 1.00):** Demuestra una tasa de discriminación óptima entre paneles sanos y defectuosos debido al comportamiento controlado de las reglas lógicas lineales aplicadas en la simulación.
* **Matriz de Confusión:** Valida una precisión impecable cruzando los cuadrantes térmicos bajo un umbral de decisión estricto de $\tau = 0.5$, arrojando 0 Falsos Positivos y 0 Falsos Negativos.
