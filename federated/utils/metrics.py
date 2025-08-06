import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import PROCESSED_DATA_DIR, FEDERATED_CONFIG  
from federated.models.base_model import BaseModel
import time

def compute_metrics_global(model_type: str):
    """Entrena y evalúa un modelo centralizado para obtener métricas globales"""

    try:
        # Cargar todos los datos federados
        all_data = []
        for i in range(1, 1 +  FEDERATED_CONFIG["num_clients"]):
            path = os.path.join(PROCESSED_DATA_DIR, f"banco{i}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_data.append(df)

        if not all_data:
            print("No se encontraron datos para evaluación global")
            return {}

        full_df = pd.concat(all_data, ignore_index=True)
        X = full_df.drop("Score", axis=1).values
        y = full_df["Score"].values

        model = BaseModel(model_type)

        # Entrenar modelo
        t0 = time.time()
        model.fit(X, y)
        training_time = time.time() - t0

        # Inferencia
        t0 = time.time()
        y_pred = model.predict(X)
        inference_time = (time.time() - t0) / len(y)

        # Calcular métricas
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return {
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "avg_training_time": training_time,
            "avg_inference_time": inference_time,
            "total_samples": len(y)
        }

    except Exception as e:
        print(f"Error en evaluación centralizada: {e}")
        return {}
