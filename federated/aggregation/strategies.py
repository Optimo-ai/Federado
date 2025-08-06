"""
Estrategias de agregación para aprendizaje federado
"""
import numpy as np
from typing import List, Tuple

class AggregationStrategy:
    """Clase base para estrategias de agregación"""
    
    def __init__(self, strategy='fedavg'):
        self.strategy = strategy
        
    def aggregate(self, parameters_list: List[List[np.ndarray]], 
                 num_samples_list: List[int]) -> List[np.ndarray]:
        """Agregar parámetros de múltiples clientes"""
        if self.strategy == 'fedavg':
            return self._federated_averaging(parameters_list, num_samples_list)
        elif self.strategy == 'fedmed':
            return self._federated_median(parameters_list, num_samples_list)
        else:
            raise ValueError(f"Estrategia de agregación no soportada: {self.strategy}")
    
    def _federated_averaging(self, parameters_list: List[List[np.ndarray]], 
                           num_samples_list: List[int]) -> List[np.ndarray]:
        """Implementar FedAvg (promedio ponderado por número de muestras)"""
        if not parameters_list:
            return []
            
        total_samples = sum(num_samples_list)
        num_params = len(parameters_list[0])
        
        aggregated_params = []
        
        for param_idx in range(num_params):
            # Inicializar parámetro agregado con ceros
            param_shape = parameters_list[0][param_idx].shape
            aggregated_param = np.zeros(param_shape)
            
            # Sumar parámetros ponderados
            for client_idx, client_params in enumerate(parameters_list):
                weight = num_samples_list[client_idx] / total_samples
                aggregated_param += weight * client_params[param_idx]
                
            aggregated_params.append(aggregated_param)
            
        return aggregated_params
    
    def _federated_median(self, parameters_list: List[List[np.ndarray]], 
                         num_samples_list: List[int]) -> List[np.ndarray]:
        """Implementar FedMed (mediana de parámetros)"""
        if not parameters_list:
            return []
            
        num_params = len(parameters_list[0])
        aggregated_params = []
        
        for param_idx in range(num_params):
            # Recopilar parámetros de todos los clientes
            param_stack = []
            for client_params in parameters_list:
                param_stack.append(client_params[param_idx])
            
            # Calcular mediana elemento por elemento
            param_array = np.stack(param_stack, axis=0)
            median_param = np.median(param_array, axis=0)
            
            aggregated_params.append(median_param)
            
        return aggregated_params
