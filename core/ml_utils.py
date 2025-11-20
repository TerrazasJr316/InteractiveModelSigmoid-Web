import joblib
import os
import gridfs
import numpy as np
import pandas as pd
from pymongo import MongoClient
from django.conf import settings

# Configuración Hardcoded o desde variables de entorno (Recomendado .env en producción)
# Asegúrate de usar la misma URI que en el script de subida
MONGO_URI = "mongodb+srv://<USUARIO>:<PASSWORD>@<CLUSTER>.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "Cluster0"
MODEL_FILENAME = 'random_forest_model.pkl'

class MLManager:
    def __init__(self):
        self.model = None
        self.features = []
        self.model_metadata = {}

    def load_model_from_mongo(self):
        """Descarga el modelo desde Mongo GridFS si no está en memoria"""
        if self.model:
            return self.model, self.features

        try:
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            fs = gridfs.GridFS(db)
            
            # Buscar el archivo
            grid_out = fs.find_one({"filename": MODEL_FILENAME})
            
            if not grid_out:
                print("Error: No se encontró el modelo en Mongo Atlas.")
                return None, []

            # Leer metadatos guardados en el script de subida
            if hasattr(grid_out, 'metadata') and grid_out.metadata:
                self.features = grid_out.metadata.get('features', [])
                self.model_metadata = grid_out.metadata
            
            # Cargar el modelo directamente desde el stream de bytes
            # joblib puede leer file-like objects
            print("Descargando modelo desde Atlas...")
            self.model = joblib.load(grid_out)
            print("Modelo cargado en memoria RAM.")
            
            return self.model, self.features

        except Exception as e:
            print(f"Error crítico conectando a Mongo: {e}")
            return None, []

    def get_sigmoid_data(self, shift_x):
        """
        Genera la data para la sigmoide visual
        """
        x_vals = np.linspace(-10, 10, 100)
        # Ecuación Sigmoide: 1 / (1 + e^-(x - bias))
        y_vals = 1 / (1 + np.exp(-(x_vals - float(shift_x))))
        return list(x_vals), list(y_vals)

    def predict_dummy(self):
        # Función auxiliar para obtener métricas guardadas sin re-entrenar
        # Esto sirve para mostrarlas en el frontend
        self.load_model_from_mongo()
        return {
            'f1_score': self.model_metadata.get('f1_score', 0),
            'features': self.features,
            'status': 'loaded_from_cloud'
        }
    
    # NOTA: Se elimina load_and_select_features completo para no cargar el CSV en Render
    # Si necesitas mostrar la tabla "preview", sugiero guardar un "head.json" pequeño 
    # en Mongo o en el repo, NO cargar el CSV de 600mb.
    
    def get_preview_data_light(self):
        """
        Para evitar cargar el CSV gigante, devolvemos datos dummy o 
        leemos un csv recortado 'sample.csv' que debes crear.
        """
        # Opción A: Leer un sample.csv pequeño que subas al repo (Recomendado)
        sample_path = os.path.join(settings.BASE_DIR, 'sample_head.csv')
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            return df
        
        # Opción B: Datos dummy si no hay archivo
        return pd.DataFrame({'Error': ['Sube un archivo sample_head.csv al repo']})