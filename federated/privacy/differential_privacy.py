"""
Implementación de técnicas de privacidad diferencial
"""
import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PRIVACY_CONFIG

class DifferentialPrivacy:
    """Clase para aplicar técnicas de privacidad diferencial"""
    
    def __init__(self, technique='none', **kwargs):
        self.technique = technique
        self.clipping_norm = kwargs.get('clipping_norm', PRIVACY_CONFIG['clipping_norm'])
        self.noise_multiplier = kwargs.get('noise_multiplier', PRIVACY_CONFIG['noise_multiplier'])
        self.epsilon = kwargs.get('epsilon', PRIVACY_CONFIG['epsilon'])
        self.delta = kwargs.get('delta', PRIVACY_CONFIG['delta'])
        
    def apply_privacy(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Aplicar técnica de privacidad diferencial a los parámetros"""
        if self.technique == 'none':
            return parameters
        elif self.technique == 'clipping':
            return self._apply_clipping(parameters)
        elif self.technique == 'noising':
            return self._apply_noising(parameters)
        elif self.technique == 'clipping_noising':
            clipped_params = self._apply_clipping(parameters)
            return self._apply_noising(clipped_params)
        else:
            raise ValueError(f"Técnica de privacidad no soportada: {self.technique}")
    
    def _apply_clipping(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Aplicar gradient clipping"""
        clipped_params = []
        
        for param in parameters:
            # Calcular norma L2
            param_norm = np.linalg.norm(param)
            
            # Aplicar clipping si es necesario
            if param_norm > self.clipping_norm:
                clipped_param = param * (self.clipping_norm / param_norm)
            else:
                clipped_param = param.copy()
                
            clipped_params.append(clipped_param)
            
        return clipped_params
    
    def _apply_noising(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Aplicar ruido gaussiano"""
        noisy_params = []
        
        for param in parameters:
            # Calcular escala del ruido basada en sensibilidad
            noise_scale = self.noise_multiplier * self.clipping_norm / self.epsilon
            
            # Generar ruido gaussiano
            noise = np.random.normal(0, noise_scale, param.shape)
            
            # Añadir ruido
            noisy_param = param + noise
            noisy_params.append(noisy_param)
            
        return noisy_params
    
    def calculate_privacy_budget(self, num_rounds: int) -> Tuple[float, float]:
        """Calcular presupuesto de privacidad total"""
        if self.technique in ['noising', 'clipping_noising']:
            # Composición simple para múltiples rondas
            total_epsilon = self.epsilon * num_rounds
            total_delta = self.delta * num_rounds
            return total_epsilon, total_delta
        else:
            return 0.0, 0.0
    
    def get_privacy_metrics(self) -> dict:
        """Obtener métricas de privacidad"""
        return {
            'technique': self.technique,
            'clipping_norm': self.clipping_norm,
            'noise_multiplier': self.noise_multiplier,
            'epsilon': self.epsilon,
            'delta': self.delta
        }
