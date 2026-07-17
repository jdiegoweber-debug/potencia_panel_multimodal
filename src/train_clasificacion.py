import os  # Módulo del sistema operativo para construir y validar rutas físicas de archivos
import torch  # Biblioteca principal de PyTorch para administrar tensores y computación en GPU
import torch.nn as nn  # Módulo de redes neuronales de PyTorch para capas lineales y de pérdida
import torch.optim as optim  # Algoritmos de optimización matemática para el descenso de gradiente
from torch.utils.data import Dataset, DataLoader, random_split  # Herramientas para procesar y dividir sets de datos
import torchvision.transforms as transforms  # Operadores para reescalar y normalizar imágenes de entrada
import pandas as pd  # Librería de Pandas para leer y manipular estructuras de datos tabulares CSV
import numpy as np  # Módulo matemático para realizar operaciones vectoriales y manejar matrices
from PIL import Image  # Biblioteca Pillow para la apertura y decodificación física de archivos JPG
from tqdm import tqdm  # Generador de barras de progreso interactivas para la terminal de comandos
import matplotlib.pyplot as plt  # Motor de renderizado gráfico para dibujar y exportar las curvas estadísticas
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay  # Métricas oficiales de Scikit-Learn

# Importamos la nueva arquitectura de clasificación binaria que guardamos en tu archivo models.py
from models import RedPanelClasificacion  # Carga el modelo multimodal estructurado con salida Sigmoide

# =============================================================================
# DATASET PERSONALIZADO ADAPTADO A CLASIFICACIÓN BINARIA
# =============================================================================
class DatasetClasificacionPaneles(Dataset):  # Define la clase encargada de preparar las muestras multimodales
    def __init__(self, df_scada, ruta_imagenes, limite_muestras=500):  # Constructor que indexa el CSV y las fotos
        self.df = df_scada.head(limite_muestras).copy()  # Extrae una cuota controlada de 500 registros para el test
        self.ruta_imagenes = ruta_imagenes  # Almacena el directorio del disco duro donde se alojan las imágenes
        self.transformacion_imagen = transforms.Compose([  # Agrupa secuencialmente el procesamiento visual requerido
            transforms.Resize((224, 224)),  # Escala la foto al tamaño exacto de entrada compatible con la ResNet
            transforms.ToTensor(),  # Transforma los píxeles enteros en un tensor flotante de PyTorch de 0.0 a 1.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Aplica medias de ImageNet
        ])  # Concluye la definición del pipeline de transformación de visión artificial

    def __len__(self):  # Método obligatorio que le indica a PyTorch la extensión de los datos de entrenamiento
        return len(self.df)  # Retorna el conteo exacto de filas presentes en la porción del dataframe copiado

    def __getitem__(self, idx):  # Método obligatorio para despachar muestras individuales usando su índice numérico
        registro = self.df.iloc[idx]  # Extrae la fila correspondiente del dataframe basándose en el puntero recibido
        ruta_relativa_img = str(registro.iloc[2])  # Recupera la ruta relativa desde la columna 2 del CSV de manera segura
        nombre_archivo = os.path.basename(ruta_relativa_img)  # Limpia caracteres para quedarse con el nombre puro de la foto
        ruta_completa_img = os.path.join(self.ruta_imagenes, nombre_archivo)  # Crea la ruta absoluta para buscar en el disco
        
        try:  # Bloque de seguridad contra fallas de lectura en el almacenamiento
            imagen = Image.open(ruta_completa_img).convert("RGB")  # Abre la fotografía y fuerza la decodificación a RGB
            tensor_imagen = self.transformacion_imagen(imagen)  # Aplica los filtros de normalización sobre los píxeles
        except Exception as e:  # Captura la excepción si la imagen física no existe o está corrupta en el disco duro
            tensor_imagen = torch.zeros(3, 224, 224)  # Genera una matriz neutra de ceros para no truncar el entrenamiento
            
        irradiacion = float(registro.iloc[3])  # Recupera el valor analógico de la columna de irradiancia (índice 3 G)
        temperatura = float(registro.iloc[4])  # Recupera la temperatura registrada desde la columna de sensores (índice 4 Tamb)
        tensor_scada = torch.tensor([irradiacion / 1000.0, temperatura / 100.0], dtype=torch.float32)  # Normaliza variables SCADA
        
        # SINCRONIZACIÓN PERFECTA: Leemos directo el cero o uno de tu columna 9 (requiere_reemplazo)
        es_defectuoso = float(registro.iloc[9])  # Extrae el indicador binario que modela la condición de reemplazo físico
        tensor_target = torch.tensor([es_defectuoso], dtype=torch.float32)  # Empaqueta el valor binario en un tensor flotante
        
        return tensor_imagen, tensor_scada, tensor_target  # Despacha la tupla multimodal al cargador automático de lotes
        irradiacion = float(registro.iloc[3])  # Recupera el valor analógico de la columna de irradiancia (índice 3 G)
        temperatura = float(registro.iloc[4])  # Recupera la temperatura registrada desde la columna de sensores (índice 4 Tamb)
        tensor_scada = torch.tensor([irradiacion / 1000.0, temperatura / 100.0], dtype=torch.float32)  # Normaliza variables SCADA
        
        # SINCRONIZACIÓN PERFECTA: Leemos directo el cero o uno de tu columna 9 (requiere_reemplazo)
        es_defectuoso = float(registro.iloc[9])  # Extrae el indicador binario que modela la condición de reemplazo físico
        tensor_target = torch.tensor([es_defectuoso], dtype=torch.float32)  # Empaqueta el valor binario en un tensor flotante
        
        return tensor_imagen, tensor_scada, tensor_target  # Despacha la tupla multimodal al cargador de lotes
# =============================================================================
# PIPELINE DE ENTRENAMIENTO Y CARACTERIZACIÓN DEL DETECTOR
# =============================================================================
def ejecutar_entrenamiento_clasificacion():  # Función principal encargada de coordinar el entrenamiento binario
    print("[CLASIFICACIÓN] Inicializando Pipeline para Detección de Reemplazo...")  # Notifica el arranque del script
    
    ruta_script = os.path.dirname(os.path.abspath(__file__))  # Obtiene el directorio exacto donde se localiza este archivo
    ruta_raiz = os.path.dirname(ruta_script)  # Sube un escalón jerárquico de carpetas para pararse en la raíz del proyecto
    ruta_base = os.path.join(ruta_raiz, "data", "raw")  # Construye de manera dinámica la ruta hacia la subcarpeta data
    ruta_images = os.path.join(ruta_base, "images")  # Localiza de forma automática el directorio de las fotos de drones
    ruta_csv = os.path.join(ruta_base, "solar_telemetry.csv")  # Construye el camino adaptativo al archivo CSV de entrada
    
    print(f"[RUTAS_CHECK] Buscando archivo CSV en: {ruta_csv}")  # Reporta en pantalla la ubicación activa analizada
    
    if not os.path.exists(ruta_csv):  # Evalúa la presencia física del archivo de datos antes de proceder
        print(f"[ERROR CRÍTICO] El script se detuvo porque NO existe el archivo CSV en la ruta indicada.")  # Alerta visual
        return  # Interrumpe la ejecución del script de manera segura
        
    df_maestro = pd.read_csv(ruta_csv)  # Lee y monta el archivo CSV completo en memoria usando un DataFrame de Pandas
    CANTIDAD_ESCENAS = 500  # Fija la cuota estricta de muestras para mantener simetría con tu experimento previo
    
    dataset_completo = DatasetClasificacionPaneles(df_maestro, ruta_images, limite_muestras=CANTIDAD_ESCENAS)  # Instancia el dataset
    
    # DIVISIÓN DEL DATASET: Usamos 80% para entrenar (400) y 20% para validar métricas ROC (100) sin mezclar datos
    tamano_train = int(0.8 * len(dataset_completo))  # Calcula la porción correspondiente al entrenamiento matemático
    tamano_val = len(dataset_completo) - tamano_train  # Determina el remanente exacto para la fase de validación
    dataset_train, dataset_val = random_split(dataset_completo, [tamano_train, tamano_val])  # Divide el dataset al azar
    
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)  # Crea el cargador de lotes para entrenamiento
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)  # Crea el cargador de validación de manera estable sin mezclar
    
    modelo = RedPanelClasificacion(num_features_scada=2)  # Instancia la red neuronal híbrida de clasificación
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Elige tarjeta de video GPU o procesador CPU
    modelo = modelo.to(dispositivo)  # Mueve todos los parámetros internos de la red al hardware seleccionado
    print(f"[HARDWARE] Ejecutando clasificación binaria sobre: {dispositivo.type.upper()}")  # Muestra el motor de ejecución
    
    # HIPERPARÁMETROS DE CLASIFICACIÓN
    criterio_perdida = nn.BCELoss()  # Binary Cross Entropy: Función de pérdida obligatoria para salidas probabilísticas Sigmoide
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)  # Ajustador de pesos optimizado con gradiente adaptativo
    
    EPOCAS = 10  # Define el número total de pasadas completas sobre el set de datos
    print(f"[PROCESO] Entrenando sobre {tamano_train} muestras y validando sobre {tamano_val} muestras...")  # Estado actual
    
    for epoca in range(1, EPOCAS + 1):  # Comienza el bucle principal de entrenamiento por épocas
        modelo.train()  # Activa los módulos dinámicos de regularización como Dropout y Normalización de Lotes
        perdida_acumulada = 0.0  # Inicializa el acumulador para promediar la pérdida de la época en curso
        
        progreso_batch = tqdm(dataloader_train, desc=f"Época {epoca:02d}/{EPOCAS:02d}", unit="batch", leave=True)  # Barra gráfica
        for batch_imagenes, batch_scada, batch_targets in progreso_batch:  # Itera sobre cada minilote del set de entrenamiento
            batch_imagenes = batch_imagenes.to(dispositivo)  # Despacha el lote de imágenes al procesador de destino
            batch_scada = batch_scada.to(dispositivo)  # Despacha las muestras tabulares SCADA al hardware de destino
            batch_targets = batch_targets.to(dispositivo)  # Despacha las etiquetas verdaderas de reemplazo a la GPU/CPU
            
            predicciones = modelo(batch_imagenes, batch_scada)  # Inyecta las variables en la red para obtener probabilidades
            loss = criterio_perdida(predicciones, batch_targets)  # Mide el desvío de la predicción usando entropía cruzada
            
            optimizador.zero_grad()  # Borra el historial de gradientes del paso anterior para evitar acumulaciones erróneas
            loss.backward()  # Ejecuta la retropropagación calculando los gradientes de error para cada neurona
            optimizador.step()  # Desplaza los pesos del modelo en la dirección óptima dictada por el gradiente
            
            perdida_acumulada += loss.item() * batch_imagenes.size(0)  # Magnifica la pérdida por el tamaño real del lote
            progreso_batch.set_postfix(Loss=f"{loss.item():.4f}")  # Imprime la pérdida instantánea al costado de la barra
            
        print(f" [RESULTADO] Época {epoca:02d} completada. Pérdida Promedio Train: {perdida_acumulada / tamano_train:.4f}")  # Reporte
        
    # =============================================================================
    # EVALUACIÓN FINAL: RECOLECCIÓN DE PREDICCIONES PARA ROC Y MATRIZ
    # =============================================================================
    modelo.eval()  # Apaga el Dropout y congela las capas de normalización para evaluar con estabilidad
    etiquetas_reales = []  # Estructura de lista vacía para almacenar los objetivos reales del subconjunto de validación
    probabilidades_predichas = []  # Estructura de lista para acumular los porcentajes de certeza devueltos por la Sigmoide
    
    print("\n[EVALUACIÓN] Caracterizando el detector sobre el conjunto de validación...")  # Notifica el inicio del test
    with torch.no_grad():  # Desactiva el cálculo de gradientes para ahorrar memoria RAM y acelerar la velocidad del test
        for batch_imagenes, batch_scada, batch_targets in dataloader_val:  # Recorre de forma secuencial el set de validación
            batch_imagenes = batch_imagenes.to(dispositivo)  # Envía las imágenes de test al hardware activo
            batch_scada = batch_scada.to(dispositivo)  # Envía los vectores de sensores de test al hardware activo
            
            outputs = modelo(batch_imagenes, batch_scada)  # Extrae las probabilidades de salida de la red neuronal
            
            etiquetas_reales.extend(batch_targets.cpu().numpy().flatten())  # Mueve las etiquetas reales a la CPU y las aplana
            probabilidades_predichas.extend(outputs.cpu().numpy().flatten())  # Almacena las predicciones continuas en la CPU
            
    etiquetas_reales = np.array(etiquetas_reales)  # Convierte la colección de objetivos en una matriz nativa de NumPy
    probabilidades_predichas = np.array(probabilidades_predichas)  # Convierte las certezas en una matriz nativa de NumPy
    
    ruta_salida_graficos = os.path.join(ruta_raiz, "output")  # Establece de manera dinámica la ruta hacia la carpeta output
    os.makedirs(ruta_salida_graficos, exist_ok=True)  # Crea la carpeta output de forma física si no existe en el almacenamiento
    
    # --- GRÁFICO 1: CURVA ROC ---
    fpr, tpr, umbrales = roc_curve(etiquetas_reales, probabilidades_predichas)  # Calcula tasas de falsos y verdaderos positivos
    area_bajo_curva = auc(fpr, tpr)  # Computa el índice numérico AUC que define la calidad de discriminación del modelo
    
    plt.figure(figsize=(6, 5))  # Inicializa un lienzo cuadrado para dibujar la curva de rendimiento del detector
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {area_bajo_curva:.2f})')  # Traza la línea ROC
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # CORREGIDO: Coordenadas fijadas para la diagonal de referencia
    plt.xlim([0.0, 1.0])  # Fija el rango del eje horizontal desde cero hasta el cien por ciento de falsos positivos
    plt.ylim([0.0, 1.05])  # Fija el rango del eje vertical permitiendo un margen superior para visualización
    plt.xlabel('Tasa de Falsos Positivos (FPR)')  # Rotula el eje X explicándole la métrica al lector
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')  # Rotula el eje Y detallando la sensibilidad del detector
    plt.title('Caracterización del Detector: Curva ROC')  # Define el título principal de la ventana del gráfico
    plt.legend(loc="lower right")  # Posiciona la leyenda de la curva en el extremo inferior derecho para no tapar datos
    plt.grid(True)  # Añade una cuadrícula de fondo para facilitar la lectura de coordenadas por el profesor
    
    ruta_roc = os.path.join(ruta_salida_graficos, "curva_roc.png")  # Combina la carpeta de salida con el nombre de la imagen ROC
    plt.savefig(ruta_roc)  # Guarda la figura actual en el almacenamiento físico en formato de imagen PNG
    print(f"[GRÁFICO] Mostrando Curva ROC en pantalla...")  # Notifica al usuario la apertura visual inminente
    plt.show()  # MUESTRA EL GRÁFICO DE LA CURVA ROC EN UNA VENTANA INTERACTIVA EN PANTALLA ADEMÁS DE GUARDARLO
    plt.close()  # Libera la memoria del lienzo gráfico de Matplotlib para no sobrecargar el sistema operativo
    print(f"[GRÁFICO] Curva ROC guardada con éxito en: {ruta_roc}")  # Informa la exportación exitosa de la curva
    
    # --- GRÁFICO 2: MATRIZ DE CONFUSIÓN ---
    decisiones_binarias = (probabilidades_predichas >= 0.5).astype(int)  # Aplica un umbral del 50% para clasificar 0 o 1
    matriz = confusion_matrix(etiquetas_reales, decisiones_binarias)  # Cruza los aciertos y fallos reales contra el umbral
    
    fig, ax = plt.subplots(figsize=(5, 5))  # Genera una nueva ventana gráfica exclusiva para la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=['Queda (Sano)', 'No Queda (Reemplazo)'])  # Configura la vista
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')  # Dibuja los cuadrantes usando tonalidades de azul y números enteros
    plt.title('Matriz de Confusión')  # Agrega el título en la parte superior del mapa térmico de aciertos
    
    ruta_matriz = os.path.join(ruta_salida_graficos, "matriz_confusion.png")  # Construye la ruta de destino de la matriz
    plt.savefig(ruta_matriz)  # Exporta la matriz de confusión a un archivo físico PNG para integrarla en tu reporte PDF
    print("[GRÁFICO] Mostrando Matriz de Confusión en pantalla...")  # Alerta sobre la proyección de la matriz térmica
    plt.show()  # MUESTRA LA MATRIZ DE CONFUSIÓN EN UNA VENTANA INTERACTIVA EN PANTALLA ADEMÁS DE GUARDARLO
    plt.close()  # Clausura el objeto gráfico limpiando los recursos del procesador
    print(f"[GRÁFICO] Matriz de Confusión guardada con éxito en: {ruta_matriz}")  # Confirma la grabación final de los datos
    
    # SALVAR PESOS EXCLUSIVOS DEL CLASIFICADOR
    ruta_archivo_pesos = os.path.join(ruta_salida_graficos, "pesos_modelo_clasificacion.pth")  # Define el nombre para los nuevos pesos
    torch.save(modelo.state_dict(), ruta_archivo_pesos)  # Guarda los coeficientes neuronales entrenados de la red de clasificación
    print(f"[ÉXITO] Pesos del clasificador exportados de forma íntegra en: {ruta_archivo_pesos}")  # Mensaje final de éxito

if __name__ == "__main__":  # Punto de entrada estándar de Python para verificar ejecuciones directas por terminal
    ejecutar_entrenamiento_clasificacion()  # Invoca la función principal para comenzar el flujo completo del clasificador
