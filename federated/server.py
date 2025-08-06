"""Servidor para aprendizaje federado"""
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.common import Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated.aggregation.strategies import AggregationStrategy
from config import FEDERATED_CONFIG


class FlowerStrategy(fl.server.strategy.Strategy):
    """Estrategia personalizada para el servidor de aprendizaje federado"""

    def __init__(self, aggregation_strategy: str = 'fedavg'):
        self.aggregation = AggregationStrategy(aggregation_strategy)
        self.round_metrics = []

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Inicializar parámetros globales"""
        # Parámetros iniciales dummy
        initial_params = [np.random.randn(10), np.array([0.0])]
        return fl.common.ndarrays_to_parameters(initial_params)

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager) -> List[Tuple[ClientProxy, Dict]]:
        """Configurar ronda de entrenamiento"""
        # Seleccionar todos los clientes disponibles
        clients = client_manager.all()

        # Configuración para cada cliente
        config = {
            'server_round': server_round,
            'local_epochs': 1,
        }

        return [(client, config) for client in clients]

    def aggregate_fit(
            self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict]:
        """Agregar resultados de entrenamiento"""
        if not results:
            return None, {}

        # Extraer parámetros y número de muestras
        parameters_list = []
        num_samples_list = []
        metrics_list = []

        for client_proxy, fit_res in results:
            parameters_list.append(
                fl.common.parameters_to_ndarrays(fit_res.parameters))
            num_samples_list.append(fit_res.num_examples)
            metrics_list.append(fit_res.metrics)

        # Agregar parámetros
        try:
            aggregated_params = self.aggregation.aggregate(
                parameters_list, num_samples_list)
            aggregated_parameters = fl.common.ndarrays_to_parameters(
                aggregated_params)
        except Exception as e:
            print(f"Error en agregación: {e}")
            # Usar primer conjunto de parámetros como fallback
            aggregated_parameters = fl.common.ndarrays_to_parameters(
                parameters_list[0])

        # Agregar métricas
        aggregated_metrics = self._aggregate_metrics(metrics_list,
                                                     num_samples_list)
        aggregated_metrics['round'] = server_round

        # Guardar métricas de la ronda
        self.round_metrics.append(aggregated_metrics)

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager) -> List[Tuple[ClientProxy, Dict]]:
        """Configurar ronda de evaluación"""
        clients = client_manager.all()
        config = {'server_round': server_round}
        return [(client, config) for client in clients]

    def aggregate_evaluate(
            self, server_round: int, results: List[Tuple[ClientProxy,
                                                         EvaluateRes]],
            failures: List[BaseException]) -> Tuple[Optional[float], Dict]:
        """Agregar resultados de evaluación"""
        if not results:
            return None, {}

        # Extraer métricas
        losses = []
        num_samples_list = []
        metrics_list = []

        for client_proxy, evaluate_res in results:
            losses.append(evaluate_res.loss)
            num_samples_list.append(evaluate_res.num_examples)
            metrics_list.append(evaluate_res.metrics)

        # Calcular pérdida promedio ponderada
        total_samples = sum(num_samples_list)
        weighted_loss = sum(loss * num_samples for loss, num_samples in zip(
            losses, num_samples_list)) / total_samples

        # Agregar métricas
        aggregated_metrics = self._aggregate_metrics(metrics_list,
                                                     num_samples_list)
        aggregated_metrics['round'] = server_round
        aggregated_metrics['aggregated_loss'] = weighted_loss

        return weighted_loss, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict]]:
       return None


    def _aggregate_metrics(self, metrics_list: List[Dict],
                           num_samples_list: List[int]) -> Dict:
        """Agregar métricas de múltiples clientes"""
        if not metrics_list:
            return {}

        total_samples = sum(num_samples_list)
        aggregated = {}

        # Métricas numéricas para promediar
        numeric_metrics = [
            'train_mae', 'train_mse', 'train_r2', 'test_mae', 'test_mse',
            'test_r2', 'training_time', 'inference_time'
        ]

        for metric in numeric_metrics:
            values = []
            weights = []

            for i, client_metrics in enumerate(metrics_list):
                if metric in client_metrics and isinstance(
                        client_metrics[metric], (int, float)):
                    values.append(client_metrics[metric])
                    weights.append(num_samples_list[i])

            if values:
                # Promedio ponderado
                weighted_avg = sum(
                    v * w for v, w in zip(values, weights)) / sum(weights)
                aggregated[f'avg_{metric}'] = weighted_avg

        # Métricas adicionales
        aggregated['total_samples'] = total_samples
        aggregated['num_clients'] = len(metrics_list)

        return aggregated


def create_strategy(aggregation_strategy: str = 'fedavg') -> FlowerStrategy:
    """Crear estrategia del servidor"""
    return FlowerStrategy(aggregation_strategy)
