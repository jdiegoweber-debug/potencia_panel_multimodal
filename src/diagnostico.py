import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import v2
from sklearn.metrics import roc_curve, auc

from models import RedPanelMultiTarea

def cargar_telemetria_scada(ruta_carpeta_raw):
    posibles_archivos = [
        ("solar_telemetry.csv", pd.read_csv),
        ("solar_telemetry.xlsx", pd.read_excel)
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
    print("[INFO] Generando gráfico estadístico de desviaciones...")
    plt.figure(figsize=(14, 7))
    indices = np.arange(len(df))
    
    plt.plot(indices, df["Potencia_Esperada_W"], color='#2ecc71', linewidth=2.5, label='Potencia Esperada Teórica (Campo W)', alpha=0.9)
    plt.plot(indices, df["Potencia_Diag_W"], color='#e74c3c', linewidth=2, linestyle='--', label='Potencia Diagnosticada por Red IA (W)', alpha=0.9)
    plt.bar(indices, df["Desviacion_Real_W"], color='#f39c12', alpha=0.5, label='Desviación Detectada por la IA (Watts)', width=0.6)
    
    plt.title(f"Auditoría Multitarea con Red Neuronal | R²: {r2:.4f} | MAE: {mae:.2f} W (Muestra: {len(df)} Paneles)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Índice Secuencial del Panel", fontsize=12, labelpad=10)
    plt.ylabel("Potencia (Watts)", fontsize=12, labelpad=10)
    
    if len(df) > 20:
        paso = max(1, len(df) // 10)
        plt.xticks(indices[::paso], df["Archivo"].iloc[::paso], rotation=30, fontsize=9)
    else:
        plt.xticks(indices, df["Archivo"], rotation=45, fontsize=9)
        
    plt.legend(loc="upper right", frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, "curva_desviaciones.png"), dpi=150)
    plt.close()

def generar_curva_roc(etiquetas_reales, probabilidades_predichas, ruta_salida):
    print("[INFO] Generando Curva ROC para el análisis de clasificación...")
    fpr, tpr, _ = roc_curve(etiquetas_reales, probabilidades_predichas)
    area_bajo_curva = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC Multitarea (AUC = {area_bajo_curva:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Rendimiento del Clasificador: Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    ruta_roc = os.path.join(ruta_salida, "curva_roc.png")
    plt.savefig(ruta_roc, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ÉXITO] Curva ROC guardada en: {ruta_roc}")

def procesar_diagnostico_multimodal():
    print("[DIAGNÓSTICO DINÁMICO] Cargando componentes de IA Multi-tarea (PyTorch)...")
    
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_script)
    
    ruta_base = os.path.join(ruta_proyecto, "data", "raw")
    ruta_images = os.path.join(ruta_base, "images")
    ruta_pesos = os.path.join(ruta_proyecto, "output", "pesos_modelo_multimodal.pth")
    ruta_salida = os.path.join(ruta_proyecto, "output")

    if not os.path.exists(ruta_images):
        print(f"[ERROR] La carpeta de imágenes no existe: {ruta_images}")
        return
    if not os.path.exists(ruta_pesos):
        print(f"[ERROR] No se encontraron los pesos de la red en: {ruta_pesos}")
        return

    df_scada = cargar_telemetria_scada(ruta_base)

    modelo = RedPanelMultiTarea(num_features_scada=4)
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    modelo.eval()

    transformacion_imagen = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Tomamos directamente los primeros 100 registros del DataFrame SCADA para asegurar consistencia e indexación real
    LIMITE_PRUEBA = 100
    df_prueba = df_scada.head(LIMITE_PRUEBA).copy()
    
    print(f"[MUESTRA] Iniciando auditoría masiva sobre {len(df_prueba)} paneles indexados reales.\n")
    
    historial_resultados = []
    etiquetas_reales_list = []
    probabilidades_predichas_list = []

    for idx in tqdm(range(len(df_prueba)), desc="Evaluando paneles", unit="img"):
        registro = df_prueba.iloc[idx]
        
        # Extracción exacta siguiendo la estructura limpia de tu dataset.py
        # Columna 2: Nombre o ruta de la foto actual
        nombre_archivo = os.path.basename(str(registro.iloc[2]))
        ruta_completa_img = os.path.join(ruta_images, nombre_archivo)

        try:
            if not os.path.exists(ruta_completa_img):
                imagen_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                imagen_pil = Image.open(ruta_completa_img).convert("RGB")
                
            tensor_imagen = transformacion_imagen(imagen_pil).unsqueeze(0).to(dispositivo)

            # Extraemos las 4 variables SCADA tal cual se inyectaron en el train.py
            # Columnas del índice 3 al 6 (G, Tamb, V, I)
            op_features = registro.iloc[3:7].values.astype('float32')
            tensor_scada = torch.tensor([op_features], dtype=torch.float32).to(dispositivo)

            # Extraemos los targets legítimos del CSV (Columna 7: Watts, Columna 9: Reemplazo)
            potencia_esperada_campo_w = float(registro.iloc[7])
            target_reemplazo_real = float(registro.iloc[9])

            with torch.no_grad():
                prediccion_watts, prediccion_reemplazo = modelo(tensor_imagen, tensor_scada)
                
            potencia_total_diagnosticada = float(prediccion_watts.cpu().item())
            prob_reemplazo = float(prediccion_reemplazo.cpu().item())
            desviacion_sistema = abs(potencia_total_diagnosticada - potencia_esperada_campo_w)

            historial_resultados.append({
                "Archivo": nombre_archivo,
                "Potencia_Esperada_W": potencia_esperada_campo_w,
                "Potencia_Diag_W": potencia_total_diagnosticada,
                "Desviacion_Real_W": desviacion_sistema,
                "Probabilidad_Reemplazo": prob_reemplazo,
                "Dictamen_IA": "REEMPLAZAR" if prob_reemplazo >= 0.5 else "SANO"
            })
            
            etiquetas_reales_list.append(target_reemplazo_real)
            probabilidades_predichas_list.append(1.0 - prob_reemplazo)

        except Exception as e:
            print(f"\n[ERROR] No se pudo procesar el índice {idx} ({nombre_archivo}): {str(e)}")
            continue

    if historial_resultados:
        df_final = pd.DataFrame(historial_resultados)

        y_real = df_final["Potencia_Esperada_W"].to_numpy()
        y_pred = df_final["Potencia_Diag_W"].to_numpy()

        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        mae_score = np.mean(np.abs(y_real - y_pred))

        df_final["Potencia_Esperada_W"] = df_final["Potencia_Esperada_W"].round(2)
        df_final["Potencia_Diag_W"] = df_final["Potencia_Diag_W"].round(2)
        df_final["Desviacion_Real_W"] = df_final["Desviacion_Real_W"].round(2)
        df_final["Probabilidad_Reemplazo"] = (df_final["Probabilidad_Reemplazo"] * 100).round(2).astype(str) + "%"

        archivo_csv = os.path.join(ruta_salida, "reporte_diagnostico.csv")
        df_final.to_csv(archivo_csv, index=False)

        print("\n================ RESUMEN DE PROCESAMIENTO MULTI-TAREA DE LA IA ================")
        print(df_final.head(10).to_string(index=False))
        print(f"\n[MÉTRICA CONTINUA] Coeficiente de Determinación R² Score : {r2_score:.4f}")
        print(f"[MÉTRICA CONTINUA] Error Absoluto Medio (MAE) : {mae_score:.2f} Watts")

        generar_y_mostrar_grafico(df_final, r2_score, mae_score, ruta_salida)
        generar_curva_roc(np.array(etiquetas_reales_list), np.array(probabilidades_predichas_list), ruta_salida)

if __name__ == "__main__":
    procesar_diagnostico_multimodal()
