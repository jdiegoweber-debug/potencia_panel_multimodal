import os  # Manejo de rutas de archivos de forma segura
import sys  # Interacción con variables de entorno del sistema
import random  # Selección aleatoria de muestras de prueba

# Asegura que la carpeta src/ esté en el path de búsqueda de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print("[DIAGNÓSTICO] Cargando módulos y componentes del sistema...")

import torch  # Biblioteca principal de PyTorch
from dataset import AsynchronousSolarDataset, get_drone_transforms  # Dataset fotovoltaico
from models import HybridSolarPredictor  # Arquitectura multimodal híbrida

def main():
    # 1. --- CONFIGURACIÓN DE RUTAS Y HARDWARE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(root_dir, "data", "raw", "solar_telemetry.csv")
    img_dir = os.path.join(root_dir, "data", "raw")
    checkpoint_path = os.path.join(root_dir, "models", "hybrid_solar_model_500.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] No se encontraron los pesos en {checkpoint_path}. ¡Espera a que termine el entrenamiento!")
        return

    # 2. --- CARGA DE DATOS ORIGINALES ---
    # Usamos las transformaciones en modo evaluación (sin aumentaciones aleatorias)
    transform = get_drone_transforms(train=False)
    full_dataset = AsynchronousSolarDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
    
    # 3. --- RECONSTRUCCIÓN DEL MODELO ---
    model = HybridSolarPredictor(num_operational_features=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Modo estático de evaluación
    print("[ÉXITO] Red Neuronal cargada y lista para diagnosticar.\n")

    # =========================================================================
    # CAMBIO CLAVE: EXCLUSIÓN DE LAS 500 IMÁGENES DE ENTRENAMIENTO
    # =========================================================================
    # Creamos un rango que empieza en el índice 500 hasta el final real del archivo
    rango_imagenes_no_vistas = range(500, len(full_dataset))
    
    cant_muestras = 2
    indices_aleatorios = random.sample(rango_imagenes_no_vistas, cant_muestras)
    print(f"--- INICIANDO EXAMEN DE {cant_muestras} PANELES NUEVOS (NO VISTOS EN ENTRENAMIENTO) ---")
    # =========================================================================

    with torch.no_grad():  # Apaga gradientes para ahorrar RAM
        for i, idx in enumerate(indices_aleatorios):
            # Extrae la información cruda del dataset
            image, op_data, p_real, visual_label = full_dataset[idx]
            
            # Formatea los tensores para simular un lote de tamaño 1 (Batch size = 1)
            img_input = image.unsqueeze(0).to(device)
            op_input = op_data.unsqueeze(0).to(device)
            
            # El modelo procesa la información híbrida (Foto + SCADA)
            p_final, lmbda, p_img, p_op = model(img_input, op_input)
            
            # Conversión de tensores a valores numéricos legibles de Python
            pred_watts = p_final.item()
            real_watts = p_real.item() if isinstance(p_real, torch.Tensor) else p_real
            val_lambda = lmbda.item()
            
            print(f"\n================ MUESTRA INÉDITA #{i+1} [Índice CSV: {idx}] ================")
            print(f"Telemetry SCADA (4 Atributos Físicos): {op_data.numpy().tolist()}")
            print(f"Potencia Real Registrada en Campo : {real_watts:.2f} Watts")
            print("----------------------------------------------------------------")
            print(">>> DIAGNÓSTICO DEL SISTEMA MULTIMODAL <<<")
            print(f"-> Potencia Total Estimada por la Red  : {pred_watts:.2f} Watts")
            print(f"-> Aporte del Análisis Visual (Imagen) : {p_img.item():.2f} Watts")
            print(f"-> Aporte del Análisis Físico (SCADA)  : {p_op.item():.2f} Watts")
            print(f"-> Balance Dinámico de Confianza (λ)   : {val_lambda:.4f}")
            print(f"   (Un λ cercano a 1 prioriza la imagen; cercano a 0 prioriza la física)")

            
            # Error absoluto de esta muestra en específico
            error_individual = abs(pred_watts - real_watts)
            print(f"-> Desviación de Precisión Puntual     : {error_individual:.2f} Watts")

if __name__ == "__main__":
    main()
