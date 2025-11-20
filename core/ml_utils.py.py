import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from django.conf import settings

# Ruta al dataset y modelo
DATASET_PATH = os.path.join(settings.BASE_DIR, 'TotalFeatures-ISCXFlowMeter.csv')
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models_backup')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class MLManager:
    def __init__(self):
        self.df = None
        self.best_features = []
    
    def load_and_select_features(self):
        # Cargar solo una muestra si es muy pesado para dev, o completo para prod
        # Para optimizar memoria, definimos tipos de datos si es posible
        if self.df is None:
            # Cargamos el CSV
            self.df = pd.read_csv(DATASET_PATH)
            
            # Limpieza básica
            self.df = self.df.dropna()
            # Asumiendo que la última columna es la etiqueta (Label)
            target_col = self.df.columns[-1]
            
            # Buscamos correlación
            # Excluimos columnas no numéricas para la correlación
            numeric_df = self.df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr().abs()
            
            # Lógica para encontrar las 2 variables con mayor correlación ENTRE ELLAS
            # (que no sea la diagonal 1.0)
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Encontrar el par con máxima correlación
            max_corr = upper.stack().idxmax()
            self.best_features = list(max_corr)
            
            print(f"Variables seleccionadas por alta correlación: {self.best_features}")
            
        return self.df, self.best_features

    def train_model(self, training_percentage):
        df, features = self.load_and_select_features()
        target_col = df.columns[-1] # Asumimos target al final
        
        # Variables X e y
        X = df[features]
        y = df[target_col]
        
        # Convertir target a numérico si es texto
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        # Split
        test_size = 1.0 - (training_percentage / 100.0)
        # Guardapolvos: Mínimo dejar 10% para test
        if test_size < 0.1: test_size = 0.1 
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # MODELO: RANDOM FOREST (No Lineal)
        # Limitamos max_depth para evitar F1 = 1.00 (Overfitting)
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predicciones
        y_pred = clf.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='weighted') # Weighted por si hay desbalance
        acc = accuracy_score(y_test, y_pred)
        
        # Guardar binario .pkl
        joblib.dump(clf, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
        
        return {
            'f1_score': round(f1, 4),
            'accuracy': round(acc, 4),
            'features': features,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

    def get_sigmoid_data(self, shift_x):
        """
        Genera datos para la visualización de la sigmoide desplazada.
        La sigmoide matemática es: 1 / (1 + e^-(x - shift))
        """
        # Usamos un rango genérico basado en los datos normalizados o reales
        x_vals = np.linspace(-10, 10, 100) # Rango para dibujar la curva
        
        # Función sigmoide manual con desplazamiento en X
        # sigmoid = 1 / (1 + exp(-(x - bias)))
        y_vals = 1 / (1 + np.exp(-(x_vals - float(shift_x))))
        
        return list(x_vals), list(y_vals)