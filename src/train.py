import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Importamos la arquitectura real que guardaste en models.py
from models import RedPanelMultimodal

class DatasetMultimodalPaneles(Dataset):
    """
    Dataset personalizado en PyTorch para cargar de forma sincrónica 
    las matrices de imágenes reales junto a sus correspondientes métricas SCADA.
    """
    def __init__(self, df_scada, ruta_imagenes, limite_muestras=500):
        # Tomamos una muestra controlada de forma estricta (500 escenas)
        self.df = df_scada.head(limite_muestras).copy()
        self.ruta_imagenes = ruta_imagenes
        
        # Transformaciones estándar exigidas por arquitecturas de visión de PyTorch (ResNet)
        self.transformacion_imagen = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Extraer la ruta de la imagen desde la primera columna del dataframe
        registro = self.df.iloc[idx]
        ruta_relativa_img = str(registro.iloc[0]) # ej: "images\0.jpg"
        nombre_archivo = os.path.basename(ruta_relativa_img)
        ruta_completa_img = os.path.join(self.ruta_imagenes, nombre_archivo)
        
        # Carga robusta de la matriz de píxeles en RGB
        try:
            imagen = Image.open(ruta_completa_img).convert("RGB")
            tensor_imagen = self.transformacion_imagen(imagen)
        except Exception as e:
            # En caso de archivos corruptos o faltantes, generamos una matriz neutra estructurada
            tensor_imagen = torch.zeros(3, 224, 224)
            
        # 2. Extracción y Normalización rápida de métricas SCADA (Columnas 1 y 2)
        irradiacion = float(registro.iloc[1])
        temperatura = float(registro.iloc[2])
        tensor_scada = torch.tensor([irradiacion / 1000.0, temperatura / 100.0], dtype=torch.float32)
        
        # 3. Target de aprendizaje: Calcular los Watts esperados usando física real
        area_panel = 1.6
        eficiencia_nominal = 0.17
        watts_reales = irradiacion * area_panel * eficiencia_nominal
        tensor_target = torch.tensor([watts_reales], dtype=torch.float32)
        
        return tensor_imagen, tensor_scada, tensor_target

def ejecutar_entrenamiento_real():
    print("[ENTRENAMIENTO REAL] Inicializando Pipeline de Deep Learning en PyTorch...")
    
    # Rutas absolutas a los datos indexados
    ruta_base = os.path.join("E:\\", "Fundamentos IA", "potencia_panel_multimodal_mejorado", "data", "raw")
    ruta_images = os.path.join(ruta_base, "images")
    ruta_csv = os.path.join(ruta_base, "solar_telemetry.csv")
    
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encuentra el archivo de telemetría base en: {ruta_csv}")
        return
        
    # Carga de dataframe maestro usando Pandas
    df_maestro = pd.read_csv(ruta_csv)
    
    # Instanciamos el pipeline del dataset configurado con tus 500 escenas
    CANTIDAD_ESCENAS = 500
    dataset_prueba = DatasetMultimodalPaneles(df_maestro, ruta_images, limite_muestras=CANTIDAD_ESCENAS)
    
    # El DataLoader agrupa la muestra en minilotes y habilita el paralelismo del procesador (Batch Size de 16)
    dataloader_prueba = DataLoader(dataset_prueba, batch_size=16, shuffle=True)
    
    # Instanciar el modelo híbrido multimodal indicando que recibirá 2 inputs del SCADA
    modelo = RedPanelMultimodal(num_features_scada=2)
    
    # Selección dinámica de hardware (Aprovecha tu tarjeta de video NVIDIA si CUDA está configurado)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    print(f"[HARDWARE] Ejecutando iteraciones de tensores sobre: {dispositivo.type.upper()}")
    
    # Definición de hiperparámetros óptimos de optimización matemática
    criterio_perdida = nn.MSELoss()  # Error Cuadrático Medio para regresión numérica continua
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)  # Algoritmo de descenso de gradiente adaptativo
    
    # Ciclo de entrenamiento (Épocas de aprendizaje profundo)
    EPOCAS = 10
    print(f"[PROCESO] Iniciando ciclo de optimización adaptativa sobre las {CANTIDAD_ESCENAS} escenas...")
    
    modelo.train()  # Encendemos las capas de regularización Dropout y BatchNorm
    for epoca in range(1, EPOCAS + 1):
        perdida_acumulada_epoca = 0.0
        
        # BARRA DE PROGRESO INTERACTIVA: tqdm envuelve al dataloader en cada época para ver el avance en tiempo real
        progreso_batch = tqdm(dataloader_prueba, desc=f"Época {epoca:02d}/{EPOCAS:02d}", unit="batch", leave=True)
        
        for batch_imagenes, batch_scada, batch_targets in progreso_batch:
            # Transferencia de lotes de tensores al hardware asignado (CPU o GPU)
            batch_imagenes = batch_imagenes.to(dispositivo)
            batch_scada = batch_scada.to(dispositivo)
            batch_targets = batch_targets.to(dispositivo)
            
            # 1. Forward Pass: Inyección multimodal paralela en la red
            predicciones = modelo(batch_imagenes, batch_scada)
            
            # 2. Cálculo de la brecha de error
            loss = criterio_perdida(predicciones, batch_targets)
            
            # 3. Backward Pass: Backpropagation matemática real (Cálculo de gradientes)
            optimizador.zero_grad()  # Limpiar históricos de memoria de gradientes anteriores
            loss.backward()          # Calcular derivadas parciales automáticas del error
            optimizador.step()       # Actualizar matrices de pesos internos neuronales
            
            perdida_acumulada_epoca += loss.item() * batch_imagenes.size(0)
            
            # Actualizar métricas dinámicas en la misma barra de progreso interactiva
            error_actual_watts = np.sqrt(loss.item())
            progreso_batch.set_postfix(Loss_W=f"{error_actual_watts:.2f}")
            
        # Calcular la raíz del error cuadrático medio al finalizar cada época completa
        error_medio_watts = np.sqrt(perdida_acumulada_epoca / CANTIDAD_ESCENAS)
        print(f" [RESULTADO] Época {epoca:02d} completada. Error Promedio Total: {error_medio_watts:.2f} Watts\n")
        
    print("\n[PROCESO TERMINADO] El ajuste por descenso de gradiente ha concluido con éxito.")
    
    # Crear directorio y salvar el estado de la memoria de la IA aprendida
    ruta_salida_pesos = os.path.join("E:\\", "Fundamentos IA", "potencia_panel_multimodal_mejorado", "output")
    os.makedirs(ruta_salida_pesos, exist_ok=True)
    ruta_archivo_pesos = os.path.join(ruta_salida_pesos, "pesos_modelo_multimodal.pth")
    
    torch.save(modelo.state_dict(), ruta_archivo_pesos)
    print(f"[ÉXITO] Pesos reales de la IA exportados de forma íntegra en: {ruta_archivo_pesos}")

if __name__ == "__main__":
    ejecutar_entrenamiento_real()
