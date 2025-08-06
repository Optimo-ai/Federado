"""
Modelos base para el aprendizaje federado
"""
import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class BaseModel:
    """Clase base para todos los modelos"""

    def __init__(self, model_type='ridge', **kwargs):
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.training_time = 0
        self.inference_time = 0

    def _create_model(self, **kwargs):
        """Crear modelo según el tipo especificado"""
        if self.model_type == 'ols':
            return LinearRegression(**kwargs)
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42, **kwargs)
        elif self.model_type == 'lasso':
            return Lasso(alpha=1.0, random_state=42, **kwargs)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        elif self.model_type == 'decision_tree':
            return DecisionTreeRegressor(random_state=42, **kwargs)
        elif self.model_type == 'knn':
            return KNeighborsRegressor(**kwargs)
        elif self.model_type == 'bayesian_ridge':
            return BayesianRidge(**kwargs)
        elif self.model_type == 'mlp':
            return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

    def fit(self, X, y):
        """Entrenar el modelo"""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        return self

    def predict(self, X):
        """Hacer predicciones"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.inference_time = time.time() - start_time
        return predictions

    def evaluate(self, X, y):
        """Evaluar el modelo"""
        predictions = self.predict(X)
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'r2': r2_score(y, predictions),
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }
        return metrics

    def get_parameters(self):
        """Obtener parámetros del modelo para agregación federada"""
        if hasattr(self.model, 'coef_'):
            # Modelos lineales
            params = [self.model.coef_]
            if hasattr(self.model, 'intercept_'):
                params.append(np.array([self.model.intercept_]))
            return params
        elif hasattr(self.model, 'estimators_'):
            # Random Forest - simplificado
            return [np.array([1.0])]  # Placeholder
        elif hasattr(self.model, 'coefs_'):
            # MLP
            params = []
            for coef in self.model.coefs_:
                params.append(coef)
            for intercept in self.model.intercepts_:
                params.append(intercept)
            return params
        else:
            return [np.array([1.0])]  # Fallback

    def set_parameters(self, parameters):
        """Establecer parámetros del modelo desde agregación federada"""
        try:
            if hasattr(self.model, 'coef_') and len(parameters) >= 1:
                # Modelos lineales
                self.model.coef_ = parameters[0]
                if len(parameters) > 1 and hasattr(self.model, 'intercept_'):
                    self.model.intercept_ = parameters[1][0]
            elif hasattr(self.model, 'coefs_') and len(parameters) >= 2:
                # MLP
                num_coefs = len(self.model.coefs_)
                self.model.coefs_ = parameters[:num_coefs]
                self.model.intercepts_ = parameters[num_coefs:]
        except Exception as e:
            print(f"Error estableciendo parámetros: {e}")
