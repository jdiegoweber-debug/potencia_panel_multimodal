import os  # Importa la librería del sistema operativo para gestionar rutas de archivos y carpetas
import pandas as pd  # Importa pandas para leer y manipular estructuras de datos tabulares desde archivos CSV
from PIL import Image  # Importa Pillow para abrir y convertir matrices físicas de imágenes de drones en disco
import torch  # Importa la biblioteca de PyTorch para estructurar los datos en tensores matemáticos
from torch.utils.data import Dataset  # Importa la clase base Dataset de PyTorch para crear cargadores personalizados
from torchvision.transforms import v2  # Importa las herramientas avanzadas v2 de Torchvision para transformaciones de imágenes

class AsynchronousSolarDataset(Dataset):  # Define la clase del cargador multimodal heredando directamente de Dataset
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file: Ruta al archivo CSV con telemetría SCADA e índice de fotos.
        img_dir: Ruta a la carpeta que contiene las imágenes de los drones.
        transform: Pipeline de transformaciones/aumentación de imágenes.
        """
        self.data_frame = pd.read_csv(csv_file)  # Lee el archivo CSV de telemetría y lo monta en memoria
        self.img_dir = img_dir  # Almacena internamente la ruta de la carpeta física que aloja las fotos
        self.transform = transform  # Guarda el pipeline de transformaciones de visión

    def __len__(self):
        return len(self.data_frame)  # Devuelve la cantidad exacta de filas o registros disponibles

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Convierte el tensor de índice en una lista estándar de Python

        # 1. Extraer variables operacionales (Columnas: G, Tamb, V, I) -> 4 características SCADA
        op_features = self.data_frame.iloc[idx, 3:7].values.astype('float32')
        op_tensor = torch.tensor(op_features)  # Tensor de dimensión [4]

        # 2. Cargar la imagen correspondiente de la telemetría asincrónica
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx, 2]))
        
        if not os.path.exists(img_name):
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # Imagen gris por defecto si falta
        else:
            image = Image.open(img_name).convert('RGB')  # Abre la imagen en modo RGB

        if self.transform:
            image = self.transform(image)  # Aplica transformaciones v2

        # 3. Target Regresión: Potencia Real Generada (Columna 7)
        p_real = torch.tensor(self.data_frame.iloc[idx, 7], dtype=torch.float32).unsqueeze(0)  # [1]

        # 4. Target Opcional: Severidad visual anotada (Columna 8)
        visual_label = torch.tensor(self.data_frame.iloc[idx, 8], dtype=torch.float32).unsqueeze(0)  # [1]

        # 5. Target Clasificación Binaria ("Queda = 0 / Reemplazo = 1") (Columna 9)
        requiere_reemplazo_valor = float(self.data_frame.iloc[idx, 9])
        requiere_reemplazo = torch.tensor([requiere_reemplazo_valor], dtype=torch.float32)  # [1]

        # Retorna todo el lote multimodal empaquetado para el bucle adaptativo
        return image, op_tensor, p_real, visual_label, requiere_reemplazo

def get_drone_transforms(train=True):
    """Pipeline de aumentación optimizado con las nuevas utilidades v2 de Torchvision."""
    if train:
        return v2.Compose([
            v2.ToImage(),  # Reemplaza a ToTensor() en v2
            v2.Resize((256, 256)),
            v2.RandomCrop((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0),
            v2.ToDtype(torch.float32, scale=True),  # Escala automáticamente a [0, 1]
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
