import joblib
import os
import gridfs
import numpy as np
import pandas as pd
from pymongo import MongoClient
from django.conf import settings

# Configuración Hardcoded o desde variables de entorno (Recomendado .env en producción)
# Asegúrate de usar la misma URI que en el script de subida
MONGO_URI = "mongodb+srv://mendoza30josue_db_user:FtkUaI5cinhT2fRI@clustersigmoid.mvjjbfl.mongodb.net/?appName=ClusterSigmoid"
DB_NAME = "Cluster0"
MODEL_FILENAME = 'random_forest_model.pkl'

class MLManager:
    def __init__(self):
        self.model = None
        self.features = []
        self.model_metadata = {}

    def load_model_from_mongo(self):
        if self.model: return self.model, self.features

        try:
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            fs = gridfs.GridFS(db)
            grid_out = fs.find_one({"filename": MODEL_FILENAME})
            
            if not grid_out: return None, []

            if hasattr(grid_out, 'metadata') and grid_out.metadata:
                self.features = grid_out.metadata.get('features', [])
                self.model_metadata = grid_out.metadata # Guardamos todos los metadatos
            
            self.model = joblib.load(grid_out)
            return self.model, self.features
        except Exception as e:
            print(f"Error Mongo: {e}")
            return None, []

    def get_sigmoid_data(self, shift_x):
        # Generamos datos para la curva
        x_vals = np.linspace(-10, 10, 200)
        # Ecuación Sigmoide desplazada
        y_vals = 1 / (1 + np.exp(-(x_vals - float(shift_x))))
        return list(x_vals), list(y_vals)

    def predict_dummy(self):
        self.load_model_from_mongo()
        return {
            # Aquí recuperamos el accuracy guardado en el Paso 1
            'f1_score': round(self.model_metadata.get('f1_score', 0), 4),
            'accuracy': round(self.model_metadata.get('accuracy', 0), 4), 
            'features': self.features,
            'status': 'loaded_from_cloud'
        }
    
    def get_preview_data_light(self):
        sample_path = os.path.join(settings.BASE_DIR, 'sample_head.csv')
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
        return pd.DataFrame({'Info': ['Sube sample_head.csv al repo']})