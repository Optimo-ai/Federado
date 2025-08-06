"""
Cliente para aprendizaje federado
"""
import flwr as fl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.models.base_model import BaseModel
from federated.privacy.differential_privacy import DifferentialPrivacy
from config import PROCESSED_DATA_DIR

class CreditScoringClient(fl.client.NumPyClient):
    """Cliente de aprendizaje federado para predicción de score crediticio"""

    def __init__(self, client_id: int, model_type: str = 'ridge', privacy_technique: str = 'none'):
        print(f"[CREANDO CLIENTE {client_id}] modelo={model_type}, privacidad={privacy_technique}", flush=True)
        self.client_id = client_id
        self.model_type = model_type
        self.privacy_technique = privacy_technique

        try:
            self.model = BaseModel(model_type)
            self.privacy = DifferentialPrivacy(privacy_technique)
            self.X_train, self.y_train, self.X_test, self.y_test = self._load_client_data()
            self.metrics_history = []
        except Exception as e:
            print(f"[ERROR] Cliente {client_id} no pudo inicializarse: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise e

    def _load_client_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cargar datos del cliente específico"""
        try:
            filename = f"banco{self.client_id}.csv"
            filepath = os.path.abspath(os.path.join(PROCESSED_DATA_DIR, filename))

            print(f"[DEBUG] Intentando cargar archivo: {filepath}", flush=True)
            print(f"[DEBUG] Archivos en {PROCESSED_DATA_DIR}: {os.listdir(PROCESSED_DATA_DIR)}", flush=True)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

            df = pd.read_csv(filepath)

            # Separar características y etiquetas
            X = df.drop('Score', axis=1).values
            y = df['Score'].values

            # Dividir en entrenamiento y prueba
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print(f"[CLIENTE {self.client_id}] {len(X_train)} muestras de entrenamiento, {len(X_test)} de prueba", flush=True)
            return X_train, y_train, X_test, y_test

        except Exception as e:
            print(f"[ERROR CRÍTICO] No se pudieron cargar los datos del cliente {self.client_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise e

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        try:
            parameters = self.model.get_parameters()
            return self.privacy.apply_privacy(parameters)
        except Exception as e:
            print(f"[ERROR] get_parameters cliente {self.client_id}: {e}", flush=True)
            return [np.array([1.0])]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        try:
            self.model.set_parameters(parameters)
        except Exception as e:
            print(f"[ERROR] set_parameters cliente {self.client_id}: {e}", flush=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        try:
            print(f"[CLIENTE {self.client_id}] Iniciando entrenamiento...", flush=True)
            self.set_parameters(parameters)
            self.model.fit(self.X_train, self.y_train)

            train_metrics = self.model.evaluate(self.X_train, self.y_train)
            test_metrics = self.model.evaluate(self.X_test, self.y_test)

            metrics = {
                f"train_{k}": v for k, v in train_metrics.items()
            }
            metrics.update(
                {
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "client_id": self.client_id,
                    "num_samples": len(self.X_train)
                }
            )

            self.metrics_history.append(metrics)
            return self.get_parameters(config), len(self.X_train), metrics

        except Exception as e:
            print(f"[ERROR] fit cliente {self.client_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return [np.array([1.0])], 1, {"error": str(e)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        try:
            self.set_parameters(parameters)
            y_pred = self.model.predict(self.X_test)
            y_true = self.y_test

            if self.model_type in ["logistic", "neural"]:
                y_pred = (y_pred >= 0.5).astype(int)
                y_true = y_true.astype(int)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "loss": mean_squared_error(y_true, y_pred),
                "inference": 0.001
            }

            return metrics["loss"], len(self.X_test), metrics

        except Exception as e:
            print(f"[ERROR] evaluate cliente {self.client_id}: {e}", flush=True)
            return 1000.0, 1, {"error": str(e)}

# ---------------------------------------------

def create_client_fn(model_type: str, privacy_technique: str):
    def client_fn(cid: str) -> CreditScoringClient:
        try:
            print(f"[create_client_fn] Recibido cid: {cid}", flush=True)
            import re
            match = re.search(r"\d+", cid)
            if not match:
                raise ValueError(f"CID inválido: {cid}")
            cid_int = int(match.group())
            return CreditScoringClient(cid_int, model_type, privacy_technique)
        except Exception as e:
            print(f"[ERROR] No se pudo crear cliente {cid}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise e
    return client_fn
