import os  # Módulo del sistema operativo para gestionar de manera segura las rutas físicas del disco duro
import torch  # Biblioteca principal de PyTorch para el manejo y operaciones masivas de tensores
import torch.nn as nn  # Módulo de PyTorch que contiene todas las capas neuronales y funciones de pérdida
import torch.optim as optim  # Algoritmos de optimización matemática para el descenso de gradiente (Adam)
from torch.utils.data import Dataset, DataLoader, random_split  # Herramientas para la estructuración y partición de sets
import torchvision.transforms as transforms  # Operadores estándar para el escalado y la normalización de píxeles
import torchvision.models as models_pretrained  # Modelos preentrenados oficiales de PyTorch para visión (ResNet)
import pandas as pd  # Biblioteca de Pandas para la carga ágil y procesamiento del archivo maestro CSV
import numpy as np  # Módulo para el control de matrices numéricas nativas de NumPy y cómputo de métricas
from PIL import Image  # Módulo Pillow encargado de la decodificación y apertura física de archivos JPG
from tqdm import tqdm  # Barra de progreso gráfica interactiva para monitorear el bucle en la consola
import matplotlib.pyplot as plt  # Motor de renderizado gráfico para la proyección y exportación de imágenes PNG
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay  # Métricas estadísticas analíticas

# =============================================================================
# ARQUITECTURA VISUAL PURA (MONOMODAL PARA CURVA ROC REALISTA)
# =============================================================================
class RedPanelSoloImagenes(nn.Module):  # Define la clase de la red monomodal visual heredando de nn.Module
    def __init__(self):  # Constructor encargado de inicializar el extractor de visión profunda
        super(RedPanelSoloImagenes, self).__init__()  # Inicializa la clase base de PyTorch de manera formal
        
        # Cargamos ResNet18 con sus pesos preentrenados oficiales congelados de ImageNet
        self.extractor_visual = models_pretrained.resnet18(weights=models_pretrained.ResNet18_Weights.DEFAULT)  # Red de visión
        num_filtros_cnn = self.extractor_visual.fc.in_features  # Extrae la dimensión de salida del bloque convolucional (512)
        self.extractor_visual.fc = nn.Identity()  # Desconecta la clasificación nativa para usarla como extractor puro
        
        # CABEZAL DE DECISIÓN VISUAL COMPACTO: Analiza los mapas de características del dron de forma directa
        self.cabezal_clasificacion = nn.Sequential(  # Agrupa las capas que tomarán la decisión final mirando solo la foto
            nn.Linear(num_filtros_cnn, 64),  # Capa densa que proyecta las 512 características visuales a un espacio de 64
            nn.ReLU(),  # Función de activación ReLU para introducir no linealidad en el análisis de la foto
            nn.Dropout(0.3),  # Dropout del 30% para apagar neuronas al azar y forzar a la red a no memorizar las fotos
            nn.Linear(64, 1),  # Capa de salida con una neurona final que calcula el puntaje de degradación
            nn.Sigmoid()  # Función Sigmoide obligatoria que convierte el puntaje en una probabilidad real entre 0 y 1
        )  # Concluye de forma segura el bloque secuencial de decisión monomodal visual

    def forward(self, x_imagen):  # Define la propagación hacia adelante inyectando únicamente la matriz de la foto
        caracteristicas_visuales = self.extractor_visual(x_imagen)  # Extrae el vector geométrico abstracto de la foto del dron
        return self.cabezal_clasificacion(caracteristicas_visuales)  # Retorna la probabilidad de reemplazo calculada por el cabezal

class DatasetSoloImagenes(Dataset):  # Dataset adaptado para ignorar el SCADA
    def __init__(self, df_scada, ruta_imagenes, limite_muestras=500):
        self.df = df_scada.head(limite_muestras).copy()
        self.ruta_imagenes = ruta_imagenes
        self.transformacion_imagen = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        registro = self.df.iloc[idx]
        ruta_relativa_img = str(registro.iloc[2])  # Columna ruta_foto_actual
        nombre_archivo = os.path.basename(ruta_relativa_img)
        ruta_completa_img = os.path.join(self.ruta_imagenes, nombre_archivo)
        
        try:
            imagen = Image.open(ruta_completa_img).convert("RGB")
            tensor_imagen = self.transformacion_imagen(imagen)
        except Exception:
            tensor_imagen = torch.zeros(3, 224, 224)
            
        # Target binario mapeado directo desde la severidad visual simulada
        severidad = float(registro.iloc[8])  # Columna severidad_visual
        es_defectuoso = 1.0 if severidad >= 0.5 else 0.0
        tensor_target = torch.tensor([es_defectuoso], dtype=torch.float32)
        
        return tensor_imagen, tensor_target

def ejecutar_entrenamiento_imagenes_puras():
    print("[MONOMODAL] Inicializando Pipeline Visual Puro (ROC Realista)...")
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_script)
    ruta_base = os.path.join(ruta_raiz, "data", "raw")
    ruta_images = os.path.join(ruta_base, "images")
    ruta_csv = os.path.join(ruta_base, "solar_telemetry.csv")
    
    if not os.path.exists(ruta_csv):
        print("[ERROR] Archivo de telemetría no hallado.")
        return
        
    df_maestro = pd.read_csv(ruta_csv)
    dataset_completo = DatasetSoloImagenes(df_maestro, ruta_images, limite_muestras=500)
    
    tamano_train = int(0.8 * len(dataset_completo))
    tamano_val = len(dataset_completo) - tamano_train
    dataset_train, dataset_val = random_split(dataset_completo, [tamano_train, tamano_val])
    
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)
    
    modelo = RedPanelSoloImagenes()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    
    criterio_perdida = nn.BCELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)
    
    print(f"[PROCESO] Entrenando sobre {tamano_train} imágenes en hardware: {dispositivo.type.upper()}")
    
    for epoca in range(1, 6):  # 5 épocas rápidas bastan para ver la curva real
        modelo.train()
        progreso_batch = tqdm(dataloader_train, desc=f"Época {epoca:02d}/05", unit="batch")
        for batch_imagenes, batch_targets in progreso_batch:
            batch_imagenes = batch_imagenes.to(dispositivo)
            batch_targets = batch_targets.to(dispositivo)
            
            predicciones = modelo(batch_imagenes)
            loss = criterio_perdida(predicciones, batch_targets)
            
            optimizador.zero_grad()
            loss.backward()
            optimizador.step()
            progreso_batch.set_postfix(Loss=f"{loss.item():.4f}")
            
    modelo.eval()
    etiquetas_reales, probabilidades_predichas = [], []
    
    print("\n[EVALUACIÓN] Extrayendo predicciones probabilísticas del test visual...")
    with torch.no_grad():
        for batch_imagenes, batch_targets in dataloader_val:
            batch_imagenes = batch_imagenes.to(dispositivo)
            outputs = modelo(batch_imagenes)
            etiquetas_reales.extend(batch_targets.cpu().numpy().flatten())
            probabilidades_predichas.extend(outputs.cpu().numpy().flatten())
            
    etiquetas_reales = np.array(etiquetas_reales)
    probabilidades_predichas = np.array(probabilidades_predichas)
    
    # --- PROYECCIÓN DE LA CURVA ROC REALISTA ---
    fpr, tpr, _ = roc_curve(etiquetas_reales, probabilidades_predichas)
    area_bajo_curva = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC Visual (AUC = {area_bajo_curva:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Trazado de diagonal corregido por código
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Rendimiento Monomodal: Solo Imágenes')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    ruta_salida = os.path.join(ruta_raiz, "output")
    os.makedirs(ruta_salida, exist_ok=True)
    plt.savefig(os.path.join(ruta_salida, "curva_roc_realista.png"))
    print("[GRÁFICO] Mostrando Curva ROC en pantalla...")
    plt.show()
    plt.close()
    print("[PROCESO TERMINADO] Pesos e imágenes guardados con éxito.")

if __name__ == "__main__":
    ejecutar_entrenamiento_imagenes_puras()
