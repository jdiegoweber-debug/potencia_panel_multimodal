# Estimación de Potencia Fotovoltaica Híbrida Multimodal

Este proyecto implementa una red neuronal profunda híbrida diseñada para estimar la potencia de salida (en Watts) de paneles solares mediante un enfoque multimodal. El sistema combina el análisis de visión computacional de imágenes térmicas/aéreas capturadas por drones con variables físicas operacionales capturadas en tiempo real por sistemas SCADA.

## 📊 Avance Actual y Metodología de Pruebas

Para garantizar un ciclo de desarrollo ágil, optimizar el uso de recursos de cómputo y validar la estabilidad de todo el pipeline matemático, hemos implementado una estrategia de **entrenamiento escalado**. 

Actualmente, el sistema se encuentra en fase de validación utilizando un subconjunto controlado de **500 muestras** (seleccionadas dinámicamente mediante `torch.utils.data.Subset`), reduciendo temporalmente la carga del set completo de telemetría masiva. Esto nos permite ejecutar pruebas de humo ("smoke tests") de extremo a extremo en entornos de CPU en pocos minutos.

---

## 📁 Arquitectura del Proyecto

A continuación, se detalla la estructura modular del repositorio:

```text
potencia_panel_multimodal_mejorado/
├── data/
│   └── raw/
│       ├── solar_telemetry.csv   # Dataset unificado de telemetría y metadatos
│       └── [imágenes_drones].jpg # Repositorio de capturas fotovoltaicas
├── models/
│   └── hybrid_solar_model_500.pth # Pesos exportados del entrenamiento controlado
├── src/
│   ├── dataset.py                # Pipeline asíncrono de carga y transformaciones (Pillow/Pandas)
│   ├── models.py                 # Definición de arquitectura híbrida y función de pérdida compuesta
│   ├── train.py                  # Script principal de entrenamiento con barras de progreso (tqdm)
│   └── diagnostico.py            # Script de inferencia y auditoría con datos inéditos
├── .venv/                        # Entorno virtual aislado de dependencias
├── .gitignore                    # Filtros de exclusión para binarios y entornos virtuales
└── requirements.txt              # Registro unificado de dependencias (PyTorch, Pandas, tqdm)
```

---

## 🚀 Componentes Clave Desarrollados

### 1. Pipeline de Datos Asíncrono (`dataset.py`)
Mapea de forma segura las variables del SCADA y gestiona el aumento de datos visuales en tiempo real sin saturar la memoria RAM.

### 2. Entrenamiento Supervisado y Control del Error (`train.py`)
Configurado para optimizar los 11 millones de parámetros de la red híbrida utilizando el optimizador Adam ($lr = 0.0003$) y una función de pérdida compuesta (`HybridSolarLoss`) regulada por un balance dinámico. Cuenta con integración de barras de progreso interactivas mediante `tqdm` para monitorear el valor de *Loss* lote por lote.

### 3. Script de Auditoría Inédita (`diagnostico.py`)
Para garantizar una evaluación rigurosa y evitar el sobreajuste (*overfitting*), este componente selecciona de forma aleatoria muestras del dataset original **excluyendo estrictamente los índices utilizados en el entrenamiento** (evalúa desde el índice 500 en adelante). 

El script imprime en consola el diagnóstico del sistema desglosando:
* Potencia total estimada en Watts.
* Aporte específico del análisis de la imagen (ResNet18).
* Aporte específico del análisis físico (SCADA).
* Coeficiente dinámico de confianza ($\lambda$), que determina matemáticamente el balance de atención del modelo frente a anomalías visuales o físicas.
