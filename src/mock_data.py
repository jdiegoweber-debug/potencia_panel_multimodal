import os  # Importa la librería del sistema operativo para gestionar carpetas de forma segura
from pathlib import Path  # Importa pathlib para manejar rutas absolutas nativas en Windows automáticamente
import numpy as np  # Importa numpy para realizar operaciones matemáticas y generar números aleatorios
import pandas as pd  # Importa pandas para administrar la telemetría en tablas y exportar archivos CSV
from PIL import Image  # Importa Pillow para la creación y manipulación de archivos físicos de imágenes

def generate_mock_dataset():  # Declara la función principal que generará todo nuestro set de datos masivo
    current_file = Path(__file__).resolve()  # Obtiene la ubicación exacta en el disco de este archivo de simulación
    root_dir = current_file.parent.parent  # Sube un nivel de carpetas para pararse en la raíz del proyecto
    raw_dir = root_dir / "data" / "raw"  # Define la ruta absoluta hacia la carpeta destino de los datos crudos
    img_dir = raw_dir / "images"  # Define la ruta absoluta hacia la subcarpeta donde se guardarán las fotos
    img_dir.mkdir(parents=True, exist_ok=True)  # Crea toda la estructura de carpetas en el disco si es que no existe
    csv_path = raw_dir / "solar_telemetry.csv"  # Define el camino y nombre final del archivo de telemetría CSV
    
    # CONFIGURACIÓN MASIVA: Fijo el set en 20.000 muestras para alimentar con potencia real a tus redes convolucionales
    num_samples = 20000  # Capacidad máxima real para un entrenamiento profundo multimodal robusto
    print(f"Generando {num_samples} muestras masivas para entrenamiento profundo...")  # Muestra el aviso en la terminal
    
    img_names = ["dron_sano.jpg", "dron_leve.jpg", "dron_critico.jpg"]  # Define los nombres de los 3 estados físicos base
    Image.new("RGB", (300, 300), color=(50, 100, 150)).save(img_dir / img_names[0])  # Genera y guarda la foto del panel óptimo (azul limpio)
    Image.new("RGB", (300, 300), color=(70, 90, 130)).save(img_dir / img_names[1])  # Genera y guarda la foto del panel con suciedad/degradación menor
    Image.new("RGB", (300, 300), color=(180, 50, 50)).save(img_dir / img_names[2])  # Genera y guarda la foto del panel con hotspot crítico (rojo de calor)
    
    data = []  # Inicializa una lista vacía para acumular fila por fila los datos del SCADA
    np.random.seed(42)  # Fija una semilla aleatoria para garantizar que los datos sean reproducibles en cualquier PC
    
    for i in range(num_samples):  # Inicia el bucle principal que iterará las 20,000 veces para construir los registros
        timestamp = f"2026-07-13 15:{i//60:02d}:{i%60:02d}"  # Crea una estampa de tiempo secuencial en minutos y segundos
        panel_id = f"PANEL_{np.random.randint(1, 10)}"  # Asigna el registro de manera aleatoria entre 9 paneles solares distintos
        
        # Distribución estadística de fallas: 75% sanos, 15% defectos leves, 10% fallas severas catastróficas
        rand_state = np.random.rand()  # Genera un número decimal aleatorio flotante entre 0 y 1
        if rand_state < 0.75:  # Si cae en el primer 75%, el panel está estructuralmente sano
            img_index = 0  # Elige el índice de la foto limpia
            severidad_visual = 0.0  # La degradación física es nula
            requiere_reemplazo = 0  # CLASIFICACIÓN BINARIA: No requiere reemplazo (0 = Queda)
        elif rand_state < 0.90:  # Si cae en el siguiente 15%, el panel presenta una anomalía leve
            img_index = 1  # Elige el índice de la foto con daño menor
            severidad_visual = 0.25  # Registra un 25% de degradación en la superficie
            requiere_reemplazo = 0  # CLASIFICACIÓN BINARIA: Aún no requiere reemplazo inmediato (0 = Queda)
        else:  # En el 10% restante, el panel está sufriendo un daño crítico
            img_index = 2  # Elige el índice de la foto roja de falla
            severidad_visual = 0.85  # Registra un 85% de severidad destructiva en la celda
            requiere_reemplazo = 1  # CLASIFICACIÓN BINARIA: Alerta crítica de mantenimiento activo (1 = No Queda / Reemplazo)
            
        img_relative_path = os.path.join("images", img_names[img_index])  # Construye la ruta interna indexada que usará el cargador
        G = np.random.uniform(300.0, 1050.0)  # Simula la Irradiancia solar instantánea en Watts por metro cuadrado
        Tamb = np.random.uniform(12.0, 42.0)  # Simula la Temperatura Ambiente del parque fotovoltaico en grados Celsius
        V = np.random.uniform(32.0, 48.0)  # Simula la Tensión eléctrica medida en los bornes del panel en Volts
        I = np.random.uniform(3.0, 10.0)  # Simula la Corriente instantánea que circula por el circuito en Amperes
        
        p_ideal = (V * I) * (G / 1000.0)  # Aplica la ley física de potencia corregida por el factor de irradiancia estándar
        # Aplica la penalización destructiva: la potencia real cae drásticamente si la severidad visual es alta
        p_real = p_ideal * (1.0 - (severidad_visual * 0.8)) + np.random.normal(0, 0.3)
        p_real = max(0.0, p_real)  # Restringe matemáticamente para evitar potencias negativas imposibles
        
        data.append([  # Inyecta toda la fila simulada dentro de la lista contenedora de memoria incluyendo el nuevo target binario
            timestamp, panel_id, img_relative_path, G, Tamb, V, I, p_real, severidad_visual, requiere_reemplazo
        ])  # Termina el empaquetado de la lista interna del lote
        
    # Agrego 'requiere_reemplazo' como la décima columna (índice 9) de nuestro mapa para que lo lea el dataset
    columns = ['timestamp', 'panel_id', 'ruta_foto_actual', 'G', 'Tamb', 'V', 'I', 'P_real', 'severidad_visual', 'requiere_reemplazo']
    df = pd.DataFrame(data, columns=columns)  # Convierte la lista en una estructura de matriz de datos tabulares de Pandas
    df.to_csv(csv_path, index=False)  # Graba la matriz completa directamente en el archivo físico del disco
    print(f"¡Éxito! Dataset masivo guardado correctamente en:\n{csv_path}")  # Confirma la finalización exitosa del proceso

if __name__ == "__main__":  # Condicional de ejecución directa desde consola
    generate_mock_dataset()  # Lanza la función de simulación masiva descrita arriba
