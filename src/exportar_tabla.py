import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exportar_y_mostrar_reporte_grafico():
    print("[REPORTE MULTIMODAL] Generando matriz integrada con letra micro-escalada...")
    
    # 1. Definición de rutas relativas dinámicas
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_script)
    ruta_csv_raw = os.path.join(ruta_proyecto, "data", "raw", "solar_telemetry.csv")
    ruta_csv_out = os.path.join(ruta_proyecto, "output", "reporte_diagnostico.csv")
    ruta_output = os.path.join(ruta_proyecto, "output")

    if not os.path.exists(ruta_csv_raw) or not os.path.exists(ruta_csv_out):
        print("[ERROR] Faltan los archivos CSV bases. Asegúrate de haber ejecutado diagnostico.py")
        return

    # 2. Cargar y combinar la telemetría SCADA real con las predicciones de la IA
    df_raw = pd.read_csv(ruta_csv_raw).head(10)
    df_out = pd.read_csv(ruta_csv_out).head(10)

    # Extraemos y renombramos las 4 variables SCADA del CSV original
    telemetria_scada = df_raw.iloc[:, 3:7].copy()
    telemetria_scada.columns = ["G_W_m2", "Tamb_C", "Voltaje_V", "Corriente_A"]

    telemetria_scada["G_W_m2"] = telemetria_scada["G_W_m2"].round(1)
    telemetria_scada["Tamb_C"] = telemetria_scada["Tamb_C"].round(1)
    telemetria_scada["Voltaje_V"] = telemetria_scada["Voltaje_V"].round(1)
    telemetria_scada["Corriente_A"] = telemetria_scada["Corriente_A"].round(2)

    # Creamos la matriz consolidada final
    df_consolidado = pd.DataFrame({
        "Archivo": df_out["Archivo"],
        "Irrad. (G)": telemetria_scada["G_W_m2"],
        "Temp. (T)": telemetria_scada["Tamb_C"],
        "Volt. (V)": telemetria_scada["Voltaje_V"],
        "Corr. (I)": telemetria_scada["Corriente_A"],
        "P. Real": df_out["Potencia_Esperada_W"],
        "P. IA": df_out["Potencia_Diag_W"],
        "Desvío": df_out["Desviacion_Real_W"],
        "Prob. Fallo": df_out["Probabilidad_Reemplazo"],
        "Dictamen": df_out["Dictamen_IA"]
    })

    # 3. Configuración del lienzo ultra-gigante (30 pulgadas) para contener la fuente del sistema
    fig, ax = plt.subplots(figsize=(30, 8), dpi=300)
    ax.axis('off')

    # Títulos ultra-cortos e independientes
    columnas_header = [
        "Archivo", "Irrad.(W/m²)", "Temp.(°C)", "Volt.(V)", "Corr.(A)",
        "P.Campo(W)", "P.IA(W)", "Desvío(W)", "Prob.Fallo", "Dictamen IA"
    ]

    # Anchos manuales súper holgados por columna
    anchos_columnas = [0.14, 0.09, 0.08, 0.08, 0.08, 0.10, 0.11, 0.09, 0.10, 0.13]

    # 4. Construcción de la tabla
    tabla_grafica = ax.table(
        cellText=df_consolidado.values, 
        colLabels=columnas_header, 
        loc='upper center', 
        cellLoc='center',
        colWidths=anchos_columnas
    )

    # 5. CONTROL ESTRICTO: Bloqueamos cualquier auto-ajuste automático de Matplotlib
    tabla_grafica.auto_set_font_size(False)
    
    # Escalado de celda alto para dar un colchón vertical generoso
    tabla_grafica.scale(1.0, 3.0) 

    # Forzamos micro-fuente celda por celda para ganarle al reescalado de Windows/DPIs
    for (fila, columna), celda in tabla_grafica.get_celld().items():
        celda.set_text_props(fontsize=3.5) # Letra micro-controlada
        
        if fila == 0:
            # Estilo del Header ejecutivo
            celda.set_text_props(color='white', weight='bold', fontsize=3.5)
            celda.set_facecolor('#1a365d')
            celda.set_edgecolor('#2b6cb0')
        else:
            # Alternancia Cebra en las filas
            if fila % 2 == 0:
                celda.set_facecolor('#f8fafc')
            else:
                celda.set_facecolor('white')
            celda.set_edgecolor('#edf2f7')
            
            # Sección SCADA
            if 1 <= columna <= 4:
                celda.set_facecolor('#f1f5f9' if fila % 2 == 0 else '#f8fafc')
                celda.get_text().set_color('#475569')
            
            # Dictamen final IA
            if columna == 9:
                texto_dictamen = celda.get_text().get_text()
                if texto_dictamen == "REEMPLAZAR":
                    celda.get_text().set_color('#e53e3e')
                    celda.get_text().set_weight('bold')
                else:
                    celda.get_text().set_color('#38a169')
                    celda.get_text().set_weight('bold')

    # Título principal con espacio aéreo de sobra
    plt.title("REPORTE INTEGRADO MULTIMODAL: TELEMETRÍA SCADA E INFERENCIA NEURONAL MULTI-TAREA", 
              fontsize=10, fontweight='bold', color='#1a365d', pad=40)

    # 6. Guardar la nueva imagen impecable en el output
    ruta_imagen_tabla = os.path.join(ruta_output, "tabla_reporte_auditoria.png")
    plt.savefig(ruta_imagen_tabla, bbox_inches='tight', dpi=300)
    print(f"[ÉXITO] Matriz micro-controlada exportada correctamente en: {ruta_imagen_tabla}")

    # 7. Desplegar ventana emergente interactiva
    print("[SISTEMA] Desplegando reporte optimizado en pantalla...")
    plt.show()
    plt.close()

if __name__ == "__main__":
    exportar_y_mostrar_reporte_grafico()
