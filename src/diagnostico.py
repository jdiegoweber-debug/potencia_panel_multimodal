import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from dotenv import load_dotenv

# Importamos la arquitectura real desde tu archivo de modelos
from models import RedPanelMultimodal

load_dotenv()

def cargar_telemetria_scada(ruta_carpeta_raw):
    posibles_archivos = [
        ("solar_telemetry.csv", pd.read_csv),
        ("solar_telemetry.xlsx", pd.read_excel),
        ("solar_telemetry.xls", pd.read_excel)
    ]
    for nombre, lector in posibles_archivos:
        ruta_archivo = os.path.join(ruta_carpeta_raw, nombre)
        if os.path.exists(ruta_archivo):
            try:
                df_scada = lector(ruta_archivo)
                print(f"[SCADA] Archivo '{nombre}' cargado con éxito. Registros: {len(df_scada)}")
                return df_scada
            except Exception as e:
                print(f"[AVISO] Error al leer {nombre}: {str(e)}")
    print("[AVISO] No se encontró 'solar_telemetry'. Se usarán datos por defecto.")
    return None

def generar_y_mostrar_grafico(df, r2, mae, ruta_salida):
    """
    Genera, guarda y muestra en pantalla la gráfica estadística con las métricas de precisión de la IA.
    """
    print("[INFO] Generando gráfico estadístico de desviaciones...")
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    indices = np.arange(len(df))
    
    # Graficar curvas reales del comportamiento del sistema
    plt.plot(indices, df["Potencia_Esperada_W"], color='#2ecc71', linewidth=2.5, label='Potencia Esperada Teórica (Campo W)', alpha=0.9)
    plt.plot(indices, df["Potencia_Diag_W"], color='#e74c3c', linewidth=2, linestyle='--', label='Potencia Diagnosticada por Red IA (W)', alpha=0.9)
    
    # Barras de pérdidas / brechas reales identificadas por PyTorch
    plt.bar(indices, df["Desviacion_Real_W"], color='#f39c12', alpha=0.5, label='Desviación Detectada por la IA (Watts)', width=0.6)
    
    # Añadimos el R² Score y el MAE destacados en el título
    plt.title(f"Auditoría con Red Neuronal Multimodal | R²: {r2:.4f} | MAE: {mae:.2f} W (Muestra: {len(df)} Paneles)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Índice Secuencial del Panel", fontsize=12, labelpad=10)
    plt.ylabel("Potencia (Watts)", fontsize=12, labelpad=10)
    
    if len(df) > 20:
        paso = max(1, len(df) // 10)
        plt.xticks(indices[::paso], df["Archivo"].iloc[::paso], rotation=30, fontsize=9)
    else:
        plt.xticks(indices, df["Archivo"], rotation=45, fontsize=9)
        
    plt.legend(loc="upper right", frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()
    
    ruta_grafico = os.path.join(ruta_salida, "curva_desviaciones.png")
    plt.savefig(ruta_grafico, dpi=300)
    print(f"[ÉXITO] Gráfico estadístico guardado en: {ruta_grafico}")
    
    print("[SISTEMA] Abriendo ventana interactiva de Matplotlib...")
    plt.show()
    plt.close()

def procesar_diagnostico_multimodal():
    print("[DIAGNÓSTICO DINÁMICO] Cargando componentes de Inteligencia Artificial (PyTorch)...")
    
    ruta_base = os.path.join("E:\\", "Fundamentos IA", "potencia_panel_multimodal_mejorado", "data", "raw")
    ruta_images = os.path.join(ruta_base, "images")
    ruta_pesos = os.path.join("E:\\", "Fundamentos IA", "potencia_panel_multimodal_mejorado", "output", "pesos_modelo_multimodal.pth")
    
    if not os.path.exists(ruta_images):
        print(f"[ERROR] La carpeta de imágenes no existe: {ruta_images}")
        return

    if not os.path.exists(ruta_pesos):
        print(f"[ERROR] No se encontraron los pesos de la red en: {ruta_pesos}")
        return

    df_scada = cargar_telemetria_scada(ruta_base)

    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    try:
        todos_los_archivos = os.listdir(ruta_images)
        lista_imagenes = [
            os.path.join(ruta_images, f) for f in todos_los_archivos 
            if f.lower().endswith(extensiones_validas)
        ]
    except Exception as e:
        print(f"[ERROR] No se pudo leer el directorio de imágenes: {str(e)}")
        return
    
    total_disponibles = len(lista_imagenes)
    print(f"[INFO] Se detectaron {total_disponibles} imágenes en la carpeta 'data/raw/images'.")

    # Instanciación de la Red Neuronal real
    modelo = RedPanelMultimodal(num_features_scada=2)
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    modelo.eval()
    
    transformacion_imagen = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    LIMITE_PRUEBA = 100
    imagenes_a_procesar = lista_imagenes[:LIMITE_PRUEBA]
    print(f"[MUESTRA] Iniciando auditoría con {len(imagenes_a_procesar)} imágenes evaluadas por la IA.\n")

    historial_resultados = []

    for idx, ruta_img in enumerate(tqdm(imagenes_a_procesar, desc="Procesando paneles", unit="img")):
        nombre_archivo = os.path.basename(ruta_img)
        try:
            imagen_pil = Image.open(ruta_img).convert("RGB")
            tensor_imagen = transformacion_imagen(imagen_pil).unsqueeze(0).to(dispositivo)
            
            irradiacion_cruda = 800.0
            temperatura_cruda = 25.0
            
            if df_scada is not None:
                ruta_busqueda = f"images\\{nombre_archivo}"
                mascara = df_scada.astype(str).eq(ruta_busqueda).any(axis=1)
                registro = df_scada[mascara]
                if not registro.empty:
                    irradiacion_cruda = float(registro.iloc[0, 1])
                    temperatura_cruda = float(registro.iloc[0, 2])
            
            area_panel = 1.6
            eficiencia_nominal = 0.17
            potencia_esperada_campo_w = irradiacion_cruda * area_panel * eficiencia_nominal
            
            tensor_scada = torch.tensor([[irradiacion_cruda / 1000.0, temperatura_cruda / 100.0]], dtype=torch.float32).to(dispositivo)
            
            with torch.no_grad():
                prediccion_tensor = modelo(tensor_imagen, tensor_scada)
                potencia_total_diagnosticada = float(prediccion_tensor.cpu().item())
            
            desviacion_sistema = abs(potencia_total_diagnosticada - potencia_esperada_campo_w)
            
            historial_resultados.append({
                "Archivo": nombre_archivo,
                "Potencia_Esperada_W": potencia_esperada_campo_w,
                "Potencia_Diag_W": potencia_total_diagnosticada,
                "Desviacion_Real_W": desviacion_sistema
            })
            
        except Exception as e:
            print(f"\n[ERROR] No se pudo procesar {nombre_archivo}: {str(e)}")
            continue

    if historial_resultados:
        df_final = pd.DataFrame(historial_resultados)
        
        # ---------------------------------------------------------------------
        # CÁLCULOS ESTADÍSTICOS CIENTÍFICOS (Métricas globales de la IA)
        # ---------------------------------------------------------------------
        y_real = df_final["Potencia_Esperada_W"].to_numpy()
        y_pred = df_final["Potencia_Diag_W"].to_numpy()
        
        # 1. R² Score
        ss_res = np.sum((y_real - y_pred) ** 2)          
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2) 
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # 2. MAE (Error Absoluto Medio)
        mae_score = np.mean(np.abs(y_real - y_pred))
        
        # Redondear para la exportación limpia del CSV
        df_final["Potencia_Esperada_W"] = df_final["Potencia_Esperada_W"].round(2)
        df_final["Potencia_Diag_W"] = df_final["Potencia_Diag_W"].round(2)
        df_final["Desviacion_Real_W"] = df_final["Desviacion_Real_W"].round(2)
        
        ruta_salida = os.path.join("E:\\", "Fundamentos IA", "potencia_panel_multimodal_mejorado", "output")
        os.makedirs(ruta_salida, exist_ok=True)
        archivo_csv = os.path.join(ruta_salida, "reporte_diagnostico.csv")
        df_final.to_csv(archivo_csv, index=False)
        
        print("\n================ RESUMEN DE PROCESAMIENTO RED NEURONAL MULTIMODAL ================")
        print(df_final.head(10).to_string(index=False))
        print("... (Truncado para legibilidad en consola) ...")
        print(f"\n[MÉTRICA GLOBAL] Coeficiente de Determinación R² Score : {r2_score:.4f}")
        print(f"[MÉTRICA GLOBAL] Error Absoluto Medio (MAE)            : {mae_score:.2f} Watts")
        print(f"[ÉXITO] Archivos tabulares guardados íntegramente en: {archivo_csv}")
        
        generar_y_mostrar_grafico(df_final, r2_score, mae_score, ruta_salida)

if __name__ == "__main__":
    procesar_diagnostico_multimodal()
