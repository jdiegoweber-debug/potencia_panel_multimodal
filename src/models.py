import torch  # Importa la librería base de PyTorch para el manejo de tensores
import torch.nn as nn  # Importa el módulo de redes neuronales para definir las capas
import torchvision.models as models  # Importa los modelos preentrenados de visión por computadora

# =============================================================================
# 1. MODELO ORIGINAL (REGRESIÓN DE POTENCIA EN WATTS)
# =============================================================================
class RedPanelMultimodal(nn.Module):
    def __init__(self, num_features_scada=2):
        super(RedPanelMultimodal, self).__init__()
        
        # --- Rama Visual ---
        self.extractor_visual = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_filtros_cnn = self.extractor_visual.fc.in_features  # 512
        self.extractor_visual.fc = nn.Identity()
        
        self.fc_visual = nn.Sequential(
            nn.Linear(num_filtros_cnn, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # --- Rama Tabular SCADA ---
        self.rama_tabular = nn.Sequential(
            nn.Linear(num_features_scada, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # --- Cabezal de Fusión (Regresión Continua) ---
        self.cabezal_regresion = nn.Sequential(
            nn.Linear(128 + 32, 64),  # 160 -> 64
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Predicción continua de Watts
        )

    def forward(self, x_imagen, x_scada):
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))
        f_tabular = self.rama_tabular(x_scada)
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)
        return self.cabezal_regresion(vector_fusionado)


# =============================================================================
# 2. MODELO ADAPTADO (CLASIFICACIÓN BINARIA: QUEDA / REEMPLAZO)
# =============================================================================
class RedPanelClasificacion(nn.Module):
    def __init__(self, num_features_scada=2):
        super(RedPanelClasificacion, self).__init__()
        
        # --- Rama Visual ---
        self.extractor_visual = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_filtros_cnn = self.extractor_visual.fc.in_features  # 512
        self.extractor_visual.fc = nn.Identity()
        
        self.fc_visual = nn.Sequential(
            nn.Linear(num_filtros_cnn, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # --- Rama Tabular SCADA ---
        self.rama_tabular = nn.Sequential(
            nn.Linear(num_features_scada, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # --- Cabezal de Fusión (Clasificación Binaria) ---
        self.cabezal_clasificacion = nn.Sequential(
            nn.Linear(128 + 32, 64),  # 160 -> 64
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Transforma el puntaje en probabilidad (0 a 1) para Curva ROC
        )

    def forward(self, x_imagen, x_scada):
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))
        f_tabular = self.rama_tabular(x_scada)
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)
        return self.cabezal_clasificacion(vector_fusionado)


# =============================================================================
# 3. MODELO MULTI-TAREA (PROPUESTA ACADÉMICA IDEAL)
# =============================================================================
class RedPanelMultiTarea(nn.Module):
    """
    Este modelo optimiza ambas tareas al mismo tiempo usando una arquitectura avanzada.
    Aprovecha el extractor ResNet50 solicitado para justificar complejidad de tesis.
    """
    def __init__(self, num_features_scada=2):
        super(RedPanelMultiTarea, self).__init__()
        
        # --- Rama Visual Avanzada (ResNet50) ---
        self.extractor_visual = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_filtros_cnn = self.extractor_visual.fc.in_features  # 2048
        self.extractor_visual.fc = nn.Identity()
        
        self.fc_visual = nn.Sequential(
            nn.Linear(num_filtros_cnn, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4)
        )
        
        # --- Rama Tabular SCADA Ampliada ---
        self.rama_tabular = nn.Sequential(
            nn.Linear(num_features_scada, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # El vector fusionado total tendrá un tamaño de 256 + 64 = 320 dimensiones
        
        # --- Cabezal Salida 1: Regresión de Potencia (Watts) ---
        self.cabezal_regresion = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # --- Cabezal Salida 2: Clasificación Estado (Vida útil / Reemplazo) ---
        self.cabezal_clasificacion = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_imagen, x_scada):
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))
        f_tabular = self.rama_tabular(x_scada)
        
        # Fusión Latente Multimodal
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)
        
        # Inferencia paralela Multi-tarea
        potencia_watts = self.cabezal_regresion(vector_fusionado)
        prob_reemplazo = self.cabezal_clasificacion(vector_fusionado)
        
        return potencia_watts, prob_reemplazo
