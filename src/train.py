import os  # Importa la librería del sistema operativo para gestionar rutas de archivos de forma segura
import sys  # Importa la librería del sistema para interactuar con variables de entorno del intérprete Python

# Extrae la ruta absoluta de la carpeta donde reside este script de entrenamiento (la carpeta src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:  # Evalúa si esa ruta ya está registrada en los caminos de búsqueda globales de Python
    sys.path.append(current_dir)  # Si no está, la inserta para permitir importaciones directas de archivos vecinos sin errores

print("[CONTROL] Script iniciado correctamente...")

import torch  # Importa la librería PyTorch para el manejo y optimización matemática de tensores
from torch.utils.data import DataLoader, Subset  # Importa el cargador para crear lotes aleatorios organizados y la herramienta para subconjuntos
from tqdm import tqdm  # Importa la librería para visualizar barras de progreso interactivas en la terminal de comandos
from dataset import AsynchronousSolarDataset, get_drone_transforms  # Importa nuestra clase de carga de datos y las transformaciones de imágenes
from models import HybridSolarPredictor, HybridSolarLoss  # Importa la arquitectura de red neuronal y nuestra función de pérdida personalizada


def main():
    print("[CONTROL] Disparando el bloque de entrenamiento directo...")

    # 1. --- CONFIGURACIÓN DE HIPERPARÁMETROS ESCALADOS ---
    epochs = 5  # Ajustamos a 5 épocas para ver una convergencia rápida y controlar el tiempo con la muestra reducida
    batch_size = 16  # Cambiamos el tamaño del lote a 16 para adaptarnos a las 500 imágenes y estabilizar los gradientes
    learning_rate = 0.0003  # Tasa de aprendizaje pequeña y controlada para evitar oscilaciones salvajes en la optimización
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Selecciona GPU NVIDIA si existe en la máquina, si no usa CPU
    print(f"Utilizando dispositivo: {device}")  # Muestra en pantalla el componente seleccionado para procesar el modelo

    # 2. --- PIPELINE DE DATOS ---
    root_dir = os.path.dirname(current_dir)  # Sube un nivel de carpetas hacia la raíz del proyecto fotovoltaico
    csv_path = os.path.join(root_dir, "data", "raw", "solar_telemetry.csv")  # Construye la ruta absoluta hacia el nuevo CSV de 2,000 muestras
    img_dir = os.path.join(root_dir, "data", "raw")  # Construye la ruta absoluta hacia las imágenes de los drones
    train_transform = get_drone_transforms(train=True)  # Inicializa el pipeline de aumentación y redimensionamiento de las imágenes
    
    # Instancia el cargador de datos unificado completo
    full_dataset = AsynchronousSolarDataset(csv_file=csv_path, img_dir=img_dir, transform=train_transform)

    # ==========================================
    # SELECCIÓN RESTRINGIDA A 500 IMÁGENES
    # ==========================================
    num_imagenes_prueba = 500  # Define el número máximo de muestras a utilizar en esta prueba corta
    indices_prueba = list(range(min(num_imagenes_prueba, len(full_dataset))))  # Genera una lista secuencial de índices hasta 500
    dataset = Subset(full_dataset, indices_prueba)  # Empaqueta el conjunto para que actúe solo sobre la muestra acotada
    # ==========================================

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Crea el iterador de lotes barajados para el entrenamiento masivo
    print(f"[CONTROL] Dataset configurado con {len(dataset)} muestras.")  # Muestra la confirmación de la carga reducida en pantalla

    # 3. --- INICIALIZACIÓN DE COMPONENTES ---
    model = HybridSolarPredictor(num_operational_features=4).to(device)  # Instancia la red híbrida y pasa sus millones de pesos a la CPU/GPU
    criterion = HybridSolarLoss(alpha=1.0, beta=1.0, gamma=0.01)  # Pérdida personalizada reduciendo la penalización de lambda para darle libertad
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Instancia el optimizador Adam pasándole todos los pesos de la red

    # 4. --- BUCLE PRINCIPAL DE ENTRENAMIENTO MULTIMODAL ---
    print("Iniciando entrenamiento del modelo Híbrido Multimodal (Subconjunto)...")
    for epoch in range(epochs):  # Recorre el ciclo de optimización a lo largo de las 5 épocas completas
        model.train()  # Coloca al modelo en modo de entrenamiento (activa capas dinámicas de Dropout y BatchNorm)
        running_loss = 0.0  # Resetea a cero el acumulador del error total sufrido en la época actual
        
        # Envolvemos el cargador con tqdm para inyectar la barra visual dinámica en la consola
        pbar = tqdm(train_loader, desc=f"Época [{epoch+1}/{epochs}]")
        
        for images, op_data, p_real, visual_labels in pbar:  # Itera lote por lote sobre las 500 muestras distribuidas de a 16
            images = images.to(device)  # Envía las matrices de píxeles del lote al hardware seleccionado (CPU/GPU)
            op_data = op_data.to(device)  # Envía las variables físicas del SCADA al hardware seleccionado
            p_real = p_real.to(device)  # Envía las etiquetas de potencia reales al hardware seleccionado
            visual_labels = visual_labels.to(device)  # Envía las etiquetas de severidad visual al hardware seleccionado

            p_final, lmbda, p_img, p_op = model(images, op_data)  # Corre el Forward Pass: calcula potencias parciales y el lambda dinámico
            loss = criterion(p_final, p_img, p_op, lmbda, p_real, visual_labels)  # Evalúa el desvío global y físico mediante la pérdida compuesta

            optimizer.zero_grad()  # Borra los residuos de gradientes del lote previo para limpiar la memoria matemática
            loss.backward()  # Corre el Backward Pass: calcula derivadas y propaga el error por los 11 millones de parámetros
            optimizer.step()  # Modifica los pesos de las conexiones neuronales de toda la red híbrida en simultáneo

            running_loss += loss.item() * images.size(0)  # Acumula la pérdida ponderada multiplicada por el tamaño real del lote actual
            
            # Actualiza el indicador de pérdida instantánea en el extremo derecho de la barra de progreso
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataset)  # Saca el promedio real de error dividiendo la acumulada por las 500 muestras totales
        print(f"-> Fin Época [{epoch+1}/{epochs}] - Pérdida Promedio: {epoch_loss:.4f}\n")  # Imprime la evolución del aprendizaje en la terminal

    # 5. --- EVALUACIÓN DE PERFORMANCE FINAL ---
    print("\n[MÉTRICAS] Calculando métricas de evaluación final...")
    model.eval()  # Coloca al modelo en modo estático desactivando aleatoriedades para realizar la auditoría
    absolute_errors = []  # Inicializa la lista para los desvíos en Watts de cada muestra
    all_preds = []  # Inicializa la lista para almacenar todas las estimaciones hechas por el modelo
    all_real = []  # Inicializa la lista para recopilar todos los valores de potencia reales históricos

    with torch.no_grad():  # Apaga el motor de gradientes de PyTorch para liberar memoria RAM y acelerar la inferencia
        for images, op_data, p_real, _ in train_loader:  # Recorre de forma pasiva todo el cargador de datos lote por lote
            images = images.to(device)  # Pasa el lote de imágenes al hardware activo
            op_data = op_data.to(device)  # Pasa el lote numérico al hardware activo
            p_final, _, _, _ = model(images, op_data)  # Obtiene la estimación de potencia final combinada del sistema
            
            absolute_errors.extend(torch.abs(p_final - p_real.to(device)).cpu().numpy().flatten())  # Acumula el error absoluto de predicción
            all_preds.extend(p_final.cpu().numpy().flatten())  # Guarda la predicción lineal en la lista de control
            all_real.extend(p_real.numpy().flatten())  # Guarda la potencia real en la lista de control

    mae = sum(absolute_errors) / len(absolute_errors)  # Saca el promedio aritmético del Error Absoluto Medio en Watts reales
    
    # Bloque de determinación del coeficiente R² Score basado en variabilidad explicada
    mean_real = sum(all_real) / len(all_real)  # Halla el centro o promedio global de las potencias registradas en el campo
    ss_res = sum((r - p) ** 2 for r, p in zip(all_real, all_preds))  # Suma el cuadrado de las distancias entre predicción y realidad (residuos)
    ss_tot = sum((r - mean_real) ** 2 for r in all_real)  # Suma el cuadrado de las variaciones naturales de los datos respecto a su media
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Divide la varianza residual por la varianza total para extraer el R² Score final

    print(f"-> Error Absoluto Medio (MAE): {mae:.2f} Watts")  # Muestra la precisión promedio final del modelo híbrido multimodal
    print(f"-> Coeficiente de Determinación (R² Score): {r2:.4f}")  # Muestra qué tanto porcentaje de la física del panel logró modelar el sistema

    # 6. --- GUARDADO AUTOMÁTICO DE PESOS ---
    models_dir = os.path.join(root_dir, "models")  # Define la ruta para resguardar la matriz de conocimiento de la red entrenada
    os.makedirs(models_dir, exist_ok=True)  # Crea el directorio models/ si no se encuentra en el almacenamiento físico
    checkpoint_path = os.path.join(models_dir, "hybrid_solar_model_500.pth")  # Define la ruta e identificación del archivo de pesos unificado para 500 muestras
    torch.save(model.state_dict(), checkpoint_path)  # Exporta y graba la matriz de pesos completa directamente en el disco duro
    print(f"\n[ÉXITO] Pesos del modelo guardados correctamente en: {checkpoint_path}")  # Mensaje de cierre de la ejecución


if __name__ == "__main__":
    main()
