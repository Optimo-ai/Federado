import os
import json
from config import RESULTS_DIR

def save_federated_metrics(model_type, strategy, privacy, metrics):
    """Guardar métricas de un experimento federado"""
    filename = f"metrics_{model_type}_{strategy}_{privacy}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
