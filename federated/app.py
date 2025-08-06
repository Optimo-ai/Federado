import flwr as fl
import os

from config import FEDERATED_CONFIG
from federated.client import create_client_fn
from federated.server import create_strategy

def start() -> None:
    # Leer parámetros desde variables de entorno
    model_type = os.environ.get("MODEL_TYPE", "ridge")
    aggregation = os.environ.get("AGGREGATION_STRATEGY", "fedavg")
    privacy = os.environ.get("PRIVACY_TECHNIQUE", "none")

    # Crear estrategia federada y función para instanciar clientes
    strategy = create_strategy(aggregation)
    client_fn = create_client_fn(
        model_type=model_type,
        privacy_technique=privacy
    )

    # Iniciar simulación federada
    fl.simulation.run_simulation(
        client_fn=client_fn,
        num_clients=FEDERATED_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FEDERATED_CONFIG["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
