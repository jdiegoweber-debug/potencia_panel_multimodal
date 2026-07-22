import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importamos las herramientas limpias que ya pusimos a punto en los pasos previos
from dataset import AsynchronousSolarDataset, get_drone_transforms
from models import RedPanelMultiTarea  # Importamos la propuesta Multi-tarea (ResNet50)

def ejecutar_entrenamiento_real():
    print("[ENTRENAMIENTO MULTI-TAREA] Inicializando Pipeline de Deep Learning...")

    # 1. Definición estricta de rutas del proyecto
    ruta_base = os.path.dirname(os.path.abspath(__file__)) # Detecta dinámicamente la carpeta src
    ruta_proyecto = os.path.dirname(ruta_base)             # Carpeta raíz del proyecto
    
    ruta_images = os.path.join(ruta_proyecto, "data", "raw", "images")
    ruta_csv = os.path.join(ruta_proyecto, "data", "raw", "solar_telemetry.csv")
    ruta_output = os.path.join(ruta_proyecto, "output")
    os.makedirs(ruta_output, exist_ok=True)

    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encuentra el archivo de telemetría en: {ruta_csv}")
        return

    # 2. Inicialización del Dataset asincrónico real (Muestra controlada de 500 escenas)
    # Nota: Tu dataset.py ya se encarga de parsear las características y targets reales del CSV
    dataset_entrenamiento = AsynchronousSolarDataset(
        csv_file=ruta_csv,
        img_dir=ruta_images,
        transform=get_drone_transforms(train=True) # Aumentación de datos activa
    )
    
    # Acotamos estrictamente a 500 muestras como pide el requerimiento de control
    CANTIDAD_ESCENAS = 500
    dataset_entrenamiento.data_frame = dataset_entrenamiento.data_frame.head(CANTIDAD_ESCENAS)

    # El DataLoader agrupa las muestras en minilotes de 16
    dataloader = DataLoader(dataset_entrenamiento, batch_size=16, shuffle=True, drop_last=True)

    # 3. Instanciar el modelo con 4 variables SCADA (G, Tamb, V, I) según tu dataset.py
    modelo = RedPanelMultiTarea(num_features_scada=4)

    # Selección dinámica de hardware (Aprovecha CUDA si está disponible)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    print(f"[HARDWARE] Ejecutando operaciones de tensores sobre: {dispositivo.type.upper()}")

    # 4. Funciones de pérdida para Multi-tarea y Optimizador
    criterio_regresion = nn.MSELoss()          # Error Cuadrático Medio para los Watts
    criterio_clasificacion = nn.BCELoss()      # Entropía Cruzada Binaria para clasificación ("Queda/Reemplazo")
    optimizador = optim.AdamW(modelo.parameters(), lr=0.0005, weight_decay=1e-4) # Optimizador avanzado para evitar Overfitting

    # Historial para graficar las curvas de pérdida
    historial_loss = []

    # 5. Ciclo de optimización (10 Épocas)
    EPOCAS = 10
    print(f"[PROCESO] Iniciando optimización adaptativa sobre {CANTIDAD_ESCENAS} escenas...")
    modelo.train()

    for epoca in range(1, EPOCAS + 1):
        perdida_acumulada = 0.0
        progreso_batch = tqdm(dataloader, desc=f"Época {epoca:02d}/{EPOCAS:02d}", unit="batch", leave=True)

        for batch_imagenes, batch_scada, batch_targets_w, _, batch_targets_cls in progreso_batch:
            # Transferencia de lotes al hardware asignado
            batch_imagenes = batch_imagenes.to(dispositivo)
            batch_scada = batch_scada.to(dispositivo)
            batch_targets_w = batch_targets_w.to(dispositivo)
            batch_targets_cls = batch_targets_cls.to(dispositivo)

            # Reajuste de dimensiones para evitar advertencias de PyTorch
            batch_targets_w = batch_targets_w.view(-1, 1)
            batch_targets_cls = batch_targets_cls.view(-1, 1)

            # Reiniciar gradientes del optimizador
            optimizador.zero_grad()

            # Forward Pass: Salidas en paralelo desde la red unificada
            pred_watts, pred_reemplazo = modelo(batch_imagenes, batch_scada)

            # Cálculo de pérdidas individuales
            loss_w = criterio_regresion(pred_watts, batch_targets_w)
            loss_cls = criterio_clasificacion(pred_reemplazo, batch_targets_cls)

            # Fusión dinámica de pérdidas (Balanceo de tareas)
            # Escalamos loss_cls para equilibrar magnitudes frente a MSE
            loss_total = loss_w + (loss_cls * 50.0) 

            # Backward Pass y actualización de pesos
            loss_total.backward()
            optimizador.step()

            perdida_acumulada += loss_total.item() * batch_imagenes.size(0)
            
            # Métricas dinámicas en barra interactiva
            error_watts = np.sqrt(loss_w.item())
            progreso_batch.set_postfix(Loss_Tot=f"{loss_total.item():.2f}", RMSE_W=f"{error_watts:.2f}")

        loss_medio_epoca = perdida_acumulada / CANTIDAD_ESCENAS
        historial_loss.append(loss_medio_epoca)
        print(f" [RESULTADO] Época {epoca:02d} completada. Pérdida Promedio Total: {loss_medio_epoca:.4f}\n")

    print("[PROCESO TERMINADO] La optimización multi-tarea ha concluido con éxito.")

    # 6. Salvar los pesos del cerebro de la IA aprendida
    ruta_archivo_pesos = os.path.join(ruta_output, "pesos_modelo_multimodal.pth")
    torch.save(modelo.state_dict(), ruta_archivo_pesos)
    print(f"[ÉXITO] Pesos multi-tarea exportados de forma íntegra en: {ruta_archivo_pesos}")

    # 7. Generar automáticamente el gráfico de la curva de aprendizaje requerido
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCAS + 1), historial_loss, marker='o', color='crimson', lw=2, label='Pérdida Total Fucionada')
    plt.title('Curva de Aprendizaje - Red Panel Multi-Tarea')
    plt.xlabel('Época de Entrenamiento')
    plt.ylabel('Valor de la Función de Pérdida (Loss)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    ruta_grafico = os.path.join(ruta_output, "curva_aprendizaje_loss.png")
    plt.savefig(ruta_grafico, dpi=150)
    plt.close()
    print(f"[ÉXITO] Curva de aprendizaje guardada en visualizaciones: {ruta_grafico}")

if __name__ == "__main__":
    ejecutar_entrenamiento_real()
