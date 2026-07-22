import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2  # Migrado a la API v2 oficial
import torchvision.models as models_pretrained
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# =============================================================================
# ARQUITECTURA VISUAL PURA
# =============================================================================
class RedPanelSoloImagenes(nn.Module):
    def __init__(self):
        super(RedPanelSoloImagenes, self).__init__()
        # Cargar ResNet18 preentrenada
        self.extractor_visual = models_pretrained.resnet18(weights=models_pretrained.ResNet18_Weights.DEFAULT)
        num_filtros_cnn = self.extractor_visual.fc.in_features
        self.extractor_visual.fc = nn.Identity()  # Extractor puro
        
        # Cabezal de clasificación binaria
        self.cabezal_clasificacion = nn.Sequential(
            nn.Linear(num_filtros_cnn, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # Control estricto de overfitting visual
            nn.Linear(64, 1),
            nn.Sigmoid()      # Probabilidad matemática de 0 a 1
        )

    def forward(self, x_imagen):
        caracteristicas_visuales = self.extractor_visual(x_imagen)
        return self.cabezal_clasificacion(caracteristicas_visuales)

# =============================================================================
# DATASET CON TRANSFORMACIONES REALISTAS (V2)
# =============================================================================
class DatasetSoloImagenes(Dataset):
    def __init__(self, df, ruta_imagenes, transform=None):
        self.df = df.reset_index(drop=True)
        self.ruta_imagenes = ruta_imagenes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        registro = self.df.iloc[idx]
        
        # Encontrar dinámicamente el nombre del archivo
        nombre_archivo = os.path.basename(str(registro.iloc[2])) # Apunta al índice de la imagen en tu fila SCADA
        ruta_completa_img = os.path.join(self.ruta_imagenes, nombre_archivo)
        
        try:
            imagen = Image.open(ruta_completa_img).convert("RGB")
        except Exception:
            imagen = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
        if self.transform:
            tensor_imagen = self.transform(imagen)
            
        # Target binario
        es_defectuoso = float(registro['target_binario'])
        tensor_target = torch.tensor([es_defectuoso], dtype=torch.float32)
        
        return tensor_imagen, tensor_target

# =============================================================================
# PIPELINE PRINCIPAL DE ENTRENAMIENTO
# =============================================================================
def ejecutar_entrenamiento_imagenes_puras():
    print("[MONOMODAL] Inicializando Pipeline Visual Puro (ROC Realista)...")
    
    ruta_script = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    ruta_raiz = os.path.dirname(ruta_script)
    ruta_base = os.path.join(ruta_raiz, "data", "raw")
    ruta_images = os.path.join(ruta_base, "images")
    ruta_csv = os.path.join(ruta_base, "solar_telemetry.csv")
    
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] Archivo de telemetría no hallado en: {ruta_csv}")
        return
        
    df_maestro = pd.read_csv(ruta_csv).head(500)
    
    # --- CREACIÓN DEL TARGET CON ERROR HUMANO / RUIDO REALISTA ---
    # Ubicamos la severidad visual (Columna índice 8 en tu CSV mapeado)
    severidad = df_maestro.iloc[:, 8].values
    prob_defecto = 1 / (1 + np.exp(-10 * (severidad - 0.5)))  # Curva logística suave
    df_maestro['target_binario'] = (np.random.rand(len(df_maestro)) < prob_defecto).astype(float)

    # Transformaciones v2 rápidas para entrenamiento
    transform_train = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # División estratificada segura
    df_train, df_val = train_test_split(
        df_maestro, test_size=0.2, random_state=42, stratify=df_maestro['target_binario']
    )

    dataset_train = DatasetSoloImagenes(df_train, ruta_images, transform=transform_train)
    dataset_val = DatasetSoloImagenes(df_val, ruta_images, transform=transform_val)

    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)

    modelo = RedPanelSoloImagenes()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)

    criterio_perdida = nn.BCELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=1e-4)

    print(f"[PROCESO] Entrenando sobre {len(dataset_train)} muestras en: {dispositivo.type.upper()}")

    # Bucle de entrenamiento corto (5 Épocas de comparación rápida)
    for epoca in range(1, 6):
        modelo.train()
        progreso_batch = tqdm(dataloader_train, desc=f"Época {epoca:02d}/05", unit="batch")
        for batch_imagenes, batch_targets in progreso_batch:
            batch_imagenes, batch_targets = batch_imagenes.to(dispositivo), batch_targets.to(dispositivo)
            
            optimizador.zero_grad()
            predicciones = modelo(batch_imagenes)
            loss = criterio_perdida(predicciones, batch_targets)
            loss.backward()
            optimizador.step()
            
            progreso_batch.set_postfix(Loss=f"{loss.item():.4f}")

    # Evaluación y guardado de métricas para la Curva ROC
    modelo.eval()
    etiquetas_reales, probabilidades_predichas = [], []

    print("\n[EVALUACIÓN] Calculando probabilidades para la Curva ROC...")
    with torch.no_grad():
        for batch_imagenes, batch_targets in dataloader_val:
            batch_imagenes = batch_imagenes.to(dispositivo)
            outputs = modelo(batch_imagenes)
            etiquetas_reales.extend(batch_targets.cpu().numpy().flatten())
            probabilidades_predichas.extend(outputs.cpu().numpy().flatten())

    etiquetas_reales = np.array(etiquetas_reales)
    probabilidades_predichas = np.array(probabilidades_predichas)

    # Cálculo final del AUC ROC
    fpr, tpr, _ = roc_curve(etiquetas_reales, probabilidades_predichas)
    area_bajo_curva = auc(fpr, tpr)

    # Generación y guardado del gráfico interactivo
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {area_bajo_curva:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Rendimiento Monomodal: Curva ROC Realista')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    ruta_salida = os.path.join(ruta_raiz, "output")
    os.makedirs(ruta_salida, exist_ok=True)
    plt.savefig(os.path.join(ruta_salida, "curva_roc_realista.png"), dpi=150)
    plt.close()
    
    # Exportar los pesos binarios aprendidos en disco
    ruta_pesos_clas = os.path.join(ruta_salida, "pesos_modelo_clasificacion.pth")
    torch.save(modelo.state_dict(), ruta_pesos_clas)
    print(f"[ÉXITO] Curva ROC guardada en output. Pesos de clasificación salvados en: {ruta_pesos_clas}")

if __name__ == "__main__":
    ejecutar_entrenamiento_imagenes_puras()
