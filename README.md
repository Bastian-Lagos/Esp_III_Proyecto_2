🧠 Verificador de Identidad Facial

Este proyecto implementa un sistema de verificación facial utilizando embeddings generados con FaceNet y un clasificador entrenado con scikit-learn. Se expone una API REST para evaluar imágenes y determinar si corresponden a una identidad específica.

🚀 Instalación
- Clona este repositorio:
git clone https://github.com/Bastian-Lagos/Esp_III_Proyecto_2
cd ruta_del_repositorio
- Instala las dependencias:
pip install -r requirements.txt



⚙️ Variables de entorno

Antes de ejecutar los scripts, asegúrate de definir las siguientes variables de entorno (puedes usar un archivo .env):
| Variable | Descripcion | Valor por defecto | 
| -------- | :---------: | ----------------:|
| MODEL_PATH | Ruta al modelo entrenado .joblib | models/model.joblib | 
| SCALER_PATH | Ruta al escalador .joblib | models/scaler.joblib | 
| THRESHOLD | Umbral de decision para clasificacion | 0.75 | 
| PORT | Puerto para FLASK | 5000 | 
| MAX_MB | Tamaño maximo en MB para las imagenes | 5 | 
| MODEL_VERSION | Nombre del modelo | me-verifier-v1 | 



📁 Organización de datos

Coloca tus imágenes en las siguientes carpetas:
data/
├── me/        # Imágenes de la persona a verificar (clase 1)
└── not_me/    # Imágenes de otras personas (clase 0)



🧪 Ejecución paso a paso

Ejecuta los siguientes scripts en orden:
- Recorte de rostros:
python .\scripts\crop_faces.py
- Generación de embeddings:
python .\scripts\embeddings.py
- Entrenamiento del modelo:
python train.py
- Evaluación y selección de umbral óptimo:
python evaluate.py
- Inicio del servidor Flask:
python .\api\api.py



📡 Endpoint disponible

Una vez iniciado el servidor, puedes acceder atraves de postman a:
- GET /healthz → Verifica que el modelo esté cargado.
- POST /verify → Envía una imagen para verificación facial