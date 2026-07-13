import torch                              # Importa la librería principal de PyTorch para el manejo de tensores
import torch.nn as nn                     # Importa el módulo de Redes Neuronales de PyTorch (capas, activaciones)
import torchvision.models as models       # Importa modelos de visión por computadora preentrenados de PyTorch

class HybridSolarPredictor(nn.Module):    # Define la clase principal de nuestro modelo de regresión híbrido
    def __init__(self, num_operational_features=4): # Constructor; recibe las 4 variables físicas del SCADA (G, Tamb, V, I)
        super(HybridSolarPredictor, self).__init__() # Inicializa la clase padre para habilitar los métodos de PyTorch
        
        # --- RAMA 1: EXTRACTOR VISUAL (CNN) ---
        # Inicializa la arquitectura ResNet18 sin descargar pesos de internet para evitar bloqueos SSL
        self.cnn_backbone = models.resnet18(weights=None)
        
        # Recorre todos los parámetros para asegurar que toda la red aprenda de cero en el nuevo disco
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True     # Libera los pesos convolucionales para ajustarse al dataset solar
            
        cnn_out_features = self.cnn_backbone.fc.in_features # Obtiene el número de características de la capa final
        self.cnn_backbone.fc = nn.Identity() # Transforma la última capa en una identidad para extraer el vector descriptivo

        # Red lineal secuencial profunda para procesar la información visual extraída
        self.img_potencia_head = nn.Sequential(
            nn.Linear(cnn_out_features, 128), # Capa lineal que reduce de 512 características a 128 neuronas
            nn.ReLU(),                        # Activación no lineal ReLU para aprender patrones complejos
            nn.Dropout(0.2),                  # Apaga el 20% de las neuronas al azar para mitigar el sobreajuste
            nn.Linear(128, 32),               # Capa lineal intermedia que reduce de 128 a 32 neuronas
            nn.ReLU(),                        # Activación no lineal ReLU
            nn.Linear(32, 1)                  # Capa final que entrega un único número: Potencia estimada por imagen
        )

        # --- RAMA 2: PREDICTOR ELÉCTRICO (MLP) ---
        # Red neuronal densa con mayor capacidad matemática para modelar las variables del SCADA
        self.mlp_branch = nn.Sequential(
            nn.BatchNorm1d(num_operational_features), # Normaliza las variables numéricas al vuelo (media=0, varianza=1)
            nn.Linear(num_operational_features, 128), # Capa lineal de entrada que proyecta a 128 neuronas ocultas
            nn.ReLU(),                        # Activación no lineal ReLU
            nn.Dropout(0.1),                  # Apaga el 10% de las neuronas para regularizar el proceso eléctrico
            nn.Linear(128, 64),               # Reducción lineal intermedia de 128 a 64 neuronas
            nn.ReLU(),                        # Activación no lineal ReLU
            nn.Linear(64, 32),                # Reducción lineal intermedia de 64 a 32 neuronas
            nn.ReLU(),                        # Activación no lineal ReLU
            nn.Linear(32, 1)                  # Capa final que entrega un único número: Potencia estimada por operación
        )

        # --- RAMA 3: COMPUERTA DINÁMICA DE ATENCIÓN (GATING) ---
        # Sub-red que calcula el factor de confianza uniendo el contexto de la imagen y de la telemetría
        self.lambda_gating = nn.Sequential(
            nn.Linear(cnn_out_features + num_operational_features, 64), # Capa lineal que une las dos fuentes en 64 neuronas
            nn.ReLU(),                        # Activación no lineal ReLU
            nn.Dropout(0.1),                  # Capa de regularización Dropout
            nn.Linear(64, 1),                 # Reduce las características a una única salida continua
            nn.Sigmoid()                      # Comprime el valor estrictamente en el rango (0, 1) para el parámetro lambda
        )

    def forward(self, image, operational_data): # Define el flujo de propagación hacia adelante del modelo
        img_features = self.cnn_backbone(image) # Extrae el vector plano descriptivo de la imagen
        p_imagen = self.img_potencia_head(img_features) # Estima la potencia basándose puramente en la imagen
        p_operacional = self.mlp_branch(operational_data) # Estima la potencia basándose puramente en el SCADA

        gating_input = torch.cat((img_features, operational_data), dim=1) # Concatena ambos vectores en un tensor unificado
        res_lambda = self.lambda_gating(gating_input) # Pasa los datos unidos por la compuerta para hallar el lambda dinámico

        # Aplica la combinación lineal convexa teórica final (Fusión Tardía)
        p_final = (res_lambda * p_imagen) + ((1.0 - res_lambda) * p_operacional)
        return p_final, res_lambda, p_imagen, p_operacional # Devuelve la predicción final y los análisis intermedios


class HybridSolarLoss(nn.Module):         # Define nuestra función de pérdida personalizada guiada por la física
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.05): # Constructor con los pesos de sintonización de errores
        super(HybridSolarLoss, self).__init__() # Inicializa los componentes base de PyTorch para pérdidas
        self.mse = nn.MSELoss()           # Instancia la función estándar de Error Cuadrático Medio como base
        self.alpha = alpha                # Guarda la ponderación asignada al error eléctrico
        self.beta = beta                  # Guarda la ponderación asignada al error de la compuerta de atención
        self.gamma = gamma                # Guarda la ponderación asignada a la regularización L2 sobre lambda

    def forward(self, p_final, p_img, p_op, lmbda, p_real, visual_damage_label=None): # Ejecuta el cálculo matemático del error
        loss_final = self.mse(p_final, p_real) # Calcula el error de diferencias al cuadrado de la predicción combinada final
        loss_op = torch.mean((1.0 - lmbda) * (p_op - p_real) ** 2) # Evalúa el desvío eléctrico centrándose en paneles sanos
        
        if visual_damage_label is not None: # Si existen anotaciones físicas de daños hechas por inspectores
            loss_gating = self.mse(lmbda, visual_damage_label) # Compara el parámetro lambda contra la verdad de terreno
        else:                             # Si opera de forma autónoma sin etiquetas de daño directo
            loss_gating = torch.mean(lmbda * p_real / (torch.max(p_real) + 1e-6)) # Castiga al sistema si lambda sube en paneles óptimos

        loss_reg = torch.mean(lmbda ** 2) # Regularización L2 para evitar oscilaciones inestables o extremas en la compuerta

        # Retorna la suma ponderada final de todos los componentes de error evaluados
        return loss_final + (self.alpha * loss_op) + (self.beta * loss_gating) + (self.gamma * loss_reg)
