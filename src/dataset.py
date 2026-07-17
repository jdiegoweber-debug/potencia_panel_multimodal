import os  # Importa la librería del sistema operativo para gestionar rutas de archivos y carpetas
import pandas as pd  # Importa pandas para leer y manipular estructuras de datos tabulares desde archivos CSV
from PIL import Image  # Importa Pillow para abrir y convertir matrices físicas de imágenes de drones en disco
import torch  # Importa la biblioteca de PyTorch para estructurar los datos en tensores matemáticos
from torch.utils.data import Dataset  # Importa la clase base Dataset de PyTorch para crear cargadores personalizados
from torchvision.transforms import v2  # Importa las herramientas avanzadas v2 de Torchvision para transformaciones de imágenes

class AsynchronousSolarDataset(Dataset):  # Define la clase del cargador multimodal heredando directamente de Dataset
    def __init__(self, csv_file, img_dir, transform=None):  # Constructor encargado de inicializar las rutas y configuraciones bases
        """
        csv_file: Ruta al archivo CSV con telemetría SCADA e índice de fotos.
        img_dir: Ruta a la carpeta que contiene las imágenes de los drones.
        transform: Pipeline de transformaciones/aumentación de imágenes.
        """
        self.data_frame = pd.read_csv(csv_file)  # Lee el archivo CSV de telemetría y lo monta en memoria en un DataFrame de Pandas
        self.img_dir = img_dir  # Almacena internamente la ruta de la carpeta física que aloja las fotos de los drones
        self.transform = transform  # Guarda el pipeline de transformaciones de visión para aplicarlo en el flujo

    def __len__(self):  # Método obligatorio de PyTorch para conocer las dimensiones del lote de datos
        return len(self.data_frame)  # Devuelve la cantidad exacta de filas o registros disponibles en el archivo de telemetría

    def __getitem__(self, idx):  # Método obligatorio para extraer muestras individuales indexadas durante el entrenamiento
        if torch.is_tensor(idx):  # Verifica si el índice recibido es un tensor de PyTorch en lugar de un entero
            idx = idx.tolist()  # Convierte el tensor de índice en una lista estándar de Python para poder operar con Pandas

        # 1. Extraer variables operacionales (Columnas: G, Tamb, V, I) que corresponden a los índices del 3 al 6 del CSV
        op_features = self.data_frame.iloc[idx, 3:7].values.astype('float32')  # Extrae los 4 sensores eléctricos en coma flotante
        op_tensor = torch.tensor(op_features)  # Empaqueta las características SCADA en un tensor nativo de PyTorch de una dimensión

        # 2. Cargar la imagen correspondiente de la telemetría asincrónica
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 2])  # Une la carpeta base con el nombre de la foto de la columna dos
        
        if not os.path.exists(img_name):  # Evalúa si la foto del panel falta o no se generó correctamente en el almacenamiento
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # Resuelve la falta creando una matriz de píxeles gris neutra
        else:  # Si la imagen existe físicamente en la ruta indicada del disco duro
            image = Image.open(img_name).convert('RGB')  # Abre el archivo del dron y fuerza la decodificación a canales de color RGB

        if self.transform:  # Comprueba si se definió un pipeline de transformaciones o aumentaciones de visión artificial
            image = self.transform(image)  # Aplica los filtros de reescalado y aumentación directamente sobre los píxeles cargados

        # 3. Target Regresión: Potencia Real Generada (Columna 7 del CSV) para tu modelo original
        p_real = torch.tensor(self.data_frame.iloc[idx, 7], dtype=torch.float32).unsqueeze(0)  # Convierte la potencia a tensor con dimensión extra

        # 4. Target Opcional: Severidad visual anotada (Columna 8 del CSV) para análisis complementarios
        visual_label = torch.tensor(self.data_frame.iloc[idx, 8], dtype=torch.float32).unsqueeze(0)  # Estructura la severidad en un tensor

        # 5. Target Clasificación Binaria ("Queda / No queda"): Requiere reemplazo lee directamente la columna nueva número 9
        requiere_reemplazo_valor = float(self.data_frame.iloc[idx, 9])  # Extrae de forma directa el cero o uno lógico del archivo CSV
        requiere_reemplazo = torch.tensor([requiere_reemplazo_valor], dtype=torch.float32)  # Convierte la bandera en tensor para la pérdida BCELoss

        # Retorna el lote completo conteniendo las entradas y los objetivos de ambos mundos para lograr convivencia perfecta
        return image, op_tensor, p_real, visual_label, requiere_reemplazo  # Envía las cinco variables ordenadas al bucle de entrenamiento

def get_drone_transforms(train=True):  # Función auxiliar para inicializar las transformaciones optimizadas del parque solar
    """Pipeline de aumentación optimizado para paneles solares."""
    if train:  # Configura las transformaciones con aumentación de datos si estamos en la fase de entrenamiento adaptativo
        return v2.Compose([  # Agrupa de forma secuencial los operadores de transformación de visión artificial
            v2.ToImage(),  # Transforma el archivo cargado de Pillow en un objeto de tipo imagen nativo de PyTorch
            v2.Resize((256, 256)),  # Redimensiona la altura y el ancho de la matriz de la foto a 256 píxeles
            v2.RandomCrop((224, 224)),  # Realiza un recorte aleatorio interno para simular variaciones de distancias del dron
            v2.RandomHorizontalFlip(p=0.5),  # Invierte la imagen de izquierda a derecha al azar con un cincuenta por ciento de probabilidad
            v2.RandomVerticalFlip(p=0.5),  # Invierte la imagen de arriba hacia abajo simulando vuelos inversos del dron
            v2.RandomRotation(degrees=15),  # Gira la imagen aleatoriamente hasta quince grados para emular desvíos de cámara
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0),  # Altera el brillo y contraste simulando nubes
            v2.ToDtype(torch.float32, scale=True),  # Cambia el tipo de datos a coma flotante escalando los píxeles de 0 a 1
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza usando las medias globales de ImageNet
        ])  # Concluye el bloque de aumentación para entrenamiento
    else:  # Si estamos en la fase estricta de validación o evaluación de métricas frente al profesor
        return v2.Compose([  # Crea la secuencia limpia de procesamiento sin distorsiones ni giros aleatorios
            v2.ToImage(),  # Convierte la imagen física de validación en estructura de datos interna de PyTorch
            v2.Resize((224, 224)),  # Escala directamente el tamaño al estándar de entrada sin hacer recortes intermedios
            v2.ToDtype(torch.float32, scale=True),  # Lleva los enteros de los píxeles a floats decimales normalizados
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Aplica los mismos factores de normalización estables
        ])  # Concluye el bloque de transformación para validación limpia
