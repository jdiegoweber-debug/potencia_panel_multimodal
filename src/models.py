import torch  # Importa la librería base de PyTorch para el manejo de tensores
import torch.nn as nn  # Importa el módulo de redes neuronales para definir las capas
import torchvision.models as models  # Importa los modelos preentrenados de visión por computadora

# =============================================================================
# 1. MODELO ORIGINAL (REGRESIÓN DE POTENCIA EN WATTS)
# =============================================================================
class RedPanelMultimodal(nn.Module):  # Define la clase del modelo original heredando de nn.Module
    def __init__(self, num_features_scada=2):  # Constructor de la red (configurado para recibir 2 variables SCADA: G y Tamb)
        super(RedPanelMultimodal, self).__init__()  # Inicializa la clase base de PyTorch
        
        # --- Rama Visual ---
        self.extractor_visual = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Carga ResNet18 con pesos preentrenados de ImageNet
        num_filtros_cnn = self.extractor_visual.fc.in_features  # Extrae la cantidad de características de entrada de la última capa (512)
        self.extractor_visual.fc = nn.Identity()  # Reemplaza la capa de clasificación original por una identidad para usarla como extractor puro
        
        self.fc_visual = nn.Sequential(  # Define un bloque secuencial para procesar las características de la imagen
            nn.Linear(num_filtros_cnn, 128),  # Capa lineal que reduce las 512 características visuales a 128
            nn.ReLU(),  # Función de activación ReLU para introducir no linealidad
            nn.Dropout(0.3)  # Dropout del 30% para apagar neuronas aleatoriamente y evitar sobreajuste
        )
        
        # --- Rama Tabular SCADA ---
        self.rama_tabular = nn.Sequential(  # Define un bloque secuencial para procesar los datos numéricos de los sensores
            nn.Linear(num_features_scada, 64),  # Capa lineal que expande las 2 métricas SCADA a 64 dimensiones
            nn.ReLU(),  # Función de activación ReLU para la rama de sensores
            nn.BatchNorm1d(64),  # Normalización por lote para estabilizar y acelerar el entrenamiento de los datos numéricos
            nn.Linear(64, 32),  # Capa lineal que reduce las 64 características a 32
            nn.ReLU(),  # Segunda función de activación ReLU para la rama de sensores
            nn.Dropout(0.2)  # Dropout del 20% para regularizar la rama tabular
        )
        
        # --- Cabezal de Fusión (Regresión Continua) ---
        self.cabezal_regresion = nn.Sequential(  # Bloque final que recibe la combinación de ambas ramas (128 + 32 = 160)
            nn.Linear(160, 64),  # Capa lineal que procesa el vector combinado de 160 dimensiones y lo baja a 64
            nn.ReLU(),  # Función de activación ReLU en el bloque de fusión
            nn.Linear(64, 16),  # Capa lineal que reduce las 64 dimensiones a 16
            nn.ReLU(),  # Función de activación ReLU previa a la salida
            nn.Linear(16, 1)  # Capa de salida con 1 sola neurona lineal que predice la potencia continua en Watts
        )

    def forward(self, x_imagen, x_scada):  # Define el flujo de datos (propagación hacia adelante) de la red
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))  # Pasa la imagen por la ResNet18 y luego por sus capas densas
        f_tabular = self.rama_tabular(x_scada)  # Pasa las métricas de los sensores por la rama tabular
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)  # Concatena horizontalmente ambos vectores en uno solo de 160 elementos
        return self.cabezal_regresion(vector_fusionado)  # Retorna la predicción de Watts calculada por el cabezal de regresión


# =============================================================================
# 2. NUEVO MODELO ADAPTADO (CLASIFICACIÓN BINARIA: QUEDA / NO QUEDA)
# =============================================================================
class RedPanelClasificacion(nn.Module):  # Define la nueva clase para el detector de reemplazo binario que busca tu train
    def __init__(self, num_features_scada=2):  # Constructor de la red de clasificación (recibe las 2 variables de tu dataloader)
        super(RedPanelClasificacion, self).__init__()  # Inicializa la clase base de PyTorch
        
        # --- Rama Visual ---
        self.extractor_visual = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Carga el extractor ResNet18 preentrenado
        num_filtros_cnn = self.extractor_visual.fc.in_features  # Obtiene el tamaño del vector de salida de la convolución (512)
        self.extractor_visual.fc = nn.Identity()  # Desconecta la clasificación original de la red de visión
        
        self.fc_visual = nn.Sequential(  # Red densa para compactar la información visual del panel solar
            nn.Linear(num_filtros_cnn, 128),  # Proyecta las 512 características de la imagen a 128 dimensiones
            nn.ReLU(),  # Aplica la activación ReLU
            nn.Dropout(0.3)  # Dropout para reducir el riesgo de sobreajuste visual
        )
        
        # --- Rama Tabular SCADA ---
        self.rama_tabular = nn.Sequential(  # Bloque de procesamiento para las señales de los sensores eléctricos
            nn.Linear(num_features_scada, 64),  # Mapea las 2 entradas de los sensores a un espacio de 64 dimensiones
            nn.ReLU(),  # Aplica activación ReLU
            nn.BatchNorm1d(64),  # Normaliza las activaciones de los sensores para mejorar la convergencia
            nn.Linear(64, 32),  # Reduce la dimensión intermedia a 32 características combinadas
            nn.ReLU(),  # Aplica activación ReLU
            nn.Dropout(0.2)  # Regularización por dropout en la rama de sensores
        )
        
        # --- Cabezal de Fusión (Clasificación Binaria para la curva ROC y Matriz) ---
        self.cabezal_clasificacion = nn.Sequential(  # Bloque final de decisión multimodal
            nn.Linear(160, 64),  # Toma la fusión de 160 elementos (128 visuales + 32 de sensores) y la lleva a 64
            nn.ReLU(),  # Aplica activación ReLU
            nn.Linear(64, 16),  # Reduce el vector de decisión intermedio a 16 dimensiones
            nn.ReLU(),  # Aplica activación ReLU
            nn.Linear(16, 1),  # Capa de salida con una única neurona que calcula el puntaje bruto del panel
            nn.Sigmoid()  # Función Sigmoide crucial que transforma el puntaje en una probabilidad entre 0 y 1 (necesaria para ROC)
        )

    def forward(self, x_imagen, x_scada):  # Define el flujo de datos para la tarea de clasificación
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))  # Extrae y procesa el vector de características de la imagen
        f_tabular = self.rama_tabular(x_scada)  # Procesa las señales de los sensores analógicos SCADA
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)  # Une la vista del dron con los datos del sensor en un solo vector
        return self.cabezal_clasificacion(vector_fusionado)  # Retorna la probabilidad final de si el panel requiere reemplazo o no


# =============================================================================
# 3. MODELO COMPLEJO SUGERIDO (COMPLEJIDAD EXTRA REQUERIDA POR EL DOCENTE)
# =============================================================================
class RedPanelComplejaResNet50(nn.Module):  # Red de alta capacidad aprovechando tu gran volumen de imágenes
    def __init__(self, num_features_scada=2):  # CORREGIDO: Se agregaron los 4 espacios de sangría reglamentarios de Python a toda la clase
        super(RedPanelComplejaResNet50, self).__init__()  # Inicializa la clase base de PyTorch de forma correcta
        
        # --- Rama Visual Pesada ---
        self.extractor_visual = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Carga la ResNet50 preentrenada (más profunda)
        num_filtros_cnn = self.extractor_visual.fc.in_features  # ResNet50 entrega un vector de salida mayor (2048 características)
        self.extractor_visual.fc = nn.Identity()  # Remueve el clasificador de 1000 clases nativo de ImageNet
        
        self.fc_visual = nn.Sequential(  # Red densa ampliada para procesar el gran volumen de datos visuales
            nn.Linear(num_filtros_cnn, 256),  # Comprime la salida masiva de 2048 a 256 características clave
            nn.ReLU(),  # Aplica activación ReLU
            nn.BatchNorm1d(256),  # Agrega normalización aquí para manejar el flujo del extractor pesado
            nn.Dropout(0.4)  # Eleva el dropout al 40% porque un extractor más grande puede memorizar de más (overfitting)
        )
        
        # --- Rama Tabular SCADA Ampliada ---
        self.rama_tabular = nn.Sequential(  # Aumenta la capacidad de procesamiento del bloque numérico
            nn.Linear(num_features_scada, 128),  # Expande los 2 sensores a 128 neuronas para buscar relaciones complejas
            nn.ReLU(),  # Aplica activación ReLU
            nn.BatchNorm1d(128),  # Paréntesis cerrado de forma correcta
              nn.Linear(32, 1),  # Capa de salida final de una neurona
            nn.Sigmoid()  # Transforma el resultado en probabilidad continua (Crucial para la curva ROC)
        )  # <--- ESTE PARENTESIS CIERRA EL nn.Sequential QUE TE DABA ERROR EN LA 124

    def forward(self, x_imagen, x_scada):  # Define la propagación de datos para este modelo complejo
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))  # Procesa la imagen del dron
        f_tabular = self.rama_tabular(x_scada)  # Procesa las señales SCADA eléctricas
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)  # Fusiona ambos bloques en un vector
        return self.cabezal_clasificacion(vector_fusionado)  # Calcula la probabilidad final
