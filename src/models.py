# Guardar este bloque exacto en src/models.py
import torch
import torch.nn as nn
import torchvision.models as models

class RedPanelMultimodal(nn.Module):
    def __init__(self, num_features_scada=2):
        super(RedPanelMultimodal, self).__init__()
        # Extractor visual robusto (ResNet18)
        self.extractor_visual = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_filtros_cnn = self.extractor_visual.fc.in_features
        self.extractor_visual.fc = nn.Identity()
        
        self.fc_visual = nn.Sequential(
            nn.Linear(num_filtros_cnn, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Procesador tabular SCADA
        self.rama_tabular = nn.Sequential(
            nn.Linear(num_features_scada, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Conexión densa de fusión multimodal (128 + 32 = 160)
        self.cabezal_regresion = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Salida continua en Watts
        )

    def forward(self, x_imagen, x_scada):
        f_visual = self.fc_visual(self.extractor_visual(x_imagen))
        f_tabular = self.rama_tabular(x_scada)
        vector_fusionado = torch.cat((f_visual, f_tabular), dim=1)
        return self.cabezal_regresion(vector_fusionado)
