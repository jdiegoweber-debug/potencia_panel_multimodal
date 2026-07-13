import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class AsynchronousSolarDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file: Ruta al archivo CSV con telemetría SCADA e índice de fotos.
        img_dir: Ruta a la carpeta que contiene las imágenes de los drones.
        transform: Pipeline de transformaciones/aumentación de imágenes.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Extraer variables operacionales (Columnas: G, Tamb, V, I)
        op_features = self.data_frame.iloc[idx, 3:7].values.astype('float32')
        op_tensor = torch.tensor(op_features)

        # 2. Cargar la imagen correspondiente de la telemetría asincrónica
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 2])
        
        # Si no tienes imágenes reales creadas aún, generamos un tensor aleatorio de prueba
        if not os.path.exists(img_name):
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 3. Target: Potencia Real Generada (Columna 7)
        p_real = torch.tensor(self.data_frame.iloc[idx, 7], dtype=torch.float32).unsqueeze(0)
        
        # 4. Target Opcional: Severidad visual anotada (Columna 8)
        visual_label = torch.tensor(self.data_frame.iloc[idx, 8], dtype=torch.float32).unsqueeze(0)

        return image, op_tensor, p_real, visual_label


def get_drone_transforms(train=True):
    """Pipeline de aumentación optimizado para paneles solares."""
    if train:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256)),
            v2.RandomCrop((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
