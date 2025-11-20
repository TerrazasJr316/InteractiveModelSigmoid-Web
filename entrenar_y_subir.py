import pandas as pd
import numpy as np
import joblib
import os
import gridfs
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# --- CONFIGURACIÓN ---
MONGO_URI = "mongodb+srv://mendoza30josue_db_user:FtkUaI5cinhT2fRI@clustersigmoid.mvjjbfl.mongodb.net/?appName=ClusterSigmoid"
DB_NAME = "Cluster0"  # El nombre de tu base de datos en Atlas
CSV_PATH = 'data/TotalFeatures-ISCXFlowMeter.csv'
MODEL_FILENAME = 'random_forest_model.pkl'

def train_and_upload():
    print(f"1. Cargando dataset: {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    
    # Limpieza rápida
    df = df.dropna()
    
    # Selección de características (Lógica de correlación del chat anterior)
    print("2. Seleccionando características...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    max_corr_pair = upper.stack().idxmax()
    best_features = list(max_corr_pair)
    print(f"   Variables seleccionadas: {best_features}")

    # Preparar X e y
    target_col = df.columns[-1]
    X = df[best_features]
    y = df[target_col]
    
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    # Split 80/20 para asegurar validación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Entrenando Random Forest (Puede tardar)...")
    # Limitamos profundidad para evitar F1=1.00 y reducir tamaño del archivo
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    # Métricas
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    print(f"   Resultado -> F1: {f1:.4f} | Accuracy: {acc:.4f}")

    if f1 >= 0.999:
        print("ADVERTENCIA: F1 es muy alto, ajusta max_depth.")

    # Guardar localmente
    print("4. Guardando .pkl localmente...")
    joblib.dump(clf, MODEL_FILENAME)

    # SUBIR A MONGO ATLAS (GridFS)
    print("5. Conectando a Mongo Atlas y subiendo archivo...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)

    # Eliminar modelo anterior si existe para no llenar la DB
    old_file = fs.find_one({"filename": MODEL_FILENAME})
    if old_file:
        fs.delete(old_file._id)
        print("   Versión anterior eliminada.")

    with open(MODEL_FILENAME, 'rb') as f:
        fs.put(f, filename=MODEL_FILENAME, 
               metadata={
                   "f1_score": f1, 
                   "features": best_features,
                   "description": "Modelo RF para API Sigmoide"
               })
    
    print("¡ÉXITO! Modelo subido a la nube correctamente.")

if __name__ == "__main__":
    train_and_upload()