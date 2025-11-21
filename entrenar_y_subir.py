import pandas as pd
import numpy as np
import joblib
import os
import gridfs
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# --- TUS DATOS DE MONGO ---
# ¡PON TU URI REAL AQUÍ!
MONGO_URI = "mongodb+srv://mendoza30josue_db_user:FtkUaI5cinhT2fRI@clustersigmoid.mvjjbfl.mongodb.net/?appName=ClusterSigmoid"
DB_NAME = "Cluster0"
CSV_PATH = 'data/TotalFeatures-ISCXFlowMeter.csv'
MODEL_FILENAME = 'random_forest_model.pkl'

def train_and_upload():
    print("1. Cargando dataset...")
    df = pd.read_csv(CSV_PATH).dropna()
    
    # Seleccionando Features
    print("2. Seleccionando features...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    best_features = list(upper.stack().idxmax())
    print(f"   Features: {best_features}")

    X = df[best_features]
    y = df[df.columns[-1]]
    if y.dtype == 'object': y = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Entrenando...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred) # CÁLCULO DEL ACCURACY
    
    print(f"   F1: {f1:.4f} | Accuracy: {acc:.4f}")

    print("4. Guardando y subiendo a Mongo...")
    joblib.dump(clf, MODEL_FILENAME)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)

    # Borrar anterior
    old = fs.find_one({"filename": MODEL_FILENAME})
    if old: fs.delete(old._id)

    # SUBIR CON METADATOS CORRECTOS
    with open(MODEL_FILENAME, 'rb') as f:
        fs.put(f, filename=MODEL_FILENAME, 
               metadata={
                   "f1_score": f1, 
                   "accuracy": acc,  # <--- ESTO FALTABA
                   "features": best_features
               })
    print("¡LISTO! Modelo actualizado en la nube.")

if __name__ == "__main__":
    train_and_upload()