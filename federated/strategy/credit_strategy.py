"""
Estrategia de aprendizaje federado para credit scoring
"""
from abc import ABC, abstractmethod
from models.base_model import BaseModel

class BaseCreditScoringStrategy(ABC):
    """
    Clase base para una estrategia de aprendizaje federado en scoring crediticio
    """

    def __init__(self, model_type='ridge', **kwargs):
        self.model_type = model_type
        self.model = BaseModel(model_type=model_type, **kwargs)

    @abstractmethod
    def evaluate(self, X_val, y_val):
        """Evaluar el modelo en el conjunto de validación"""
        pass


class LocalTrainingStrategy(BaseCreditScoringStrategy):
    """
    Estrategia concreta que entrena localmente un modelo y lo evalúa
    """

    def evaluate(self, X_val, y_val):
        return self.model.evaluate(X_val, y_val)
