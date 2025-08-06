"""
Servicio de predicción usando el modelo entrenado
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODELS_DIR, PROCESSED_DATA_DIR

class PredictionService:
    """Servicio para realizar predicciones con el modelo federado"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()
        self.load_preprocessor()
    
    def load_model(self):
        """Cargar modelo entrenado"""
        try:
            model_path = os.path.join(MODELS_DIR, 'modelo_final.pkl')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("Modelo cargado exitosamente")
            else:
                print(f"Modelo no encontrado en: {model_path}")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
    
    def load_preprocessor(self):
        """Cargar preprocessor para transformar datos"""
        try:
            preprocessor_path = os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                print("Preprocessor cargado exitosamente")
            else:
                print(f"Preprocessor no encontrado en: {preprocessor_path}")
        except Exception as e:
            print(f"Error cargando preprocessor: {e}")
    
    def preprocess_data(self, df):
        """Preprocesar datos de entrada"""
        try:
            # Hacer copia para no modificar original
            df_processed = df.copy()
            
            # Eliminar columna ID si existe
            if 'ID' in df_processed.columns:
                ids = df_processed['ID'].copy()
                df_processed = df_processed.drop('ID', axis=1)
            else:
                ids = pd.Series(range(len(df_processed)))
            
            # Eliminar columna Score si existe (para predicción)
            if 'Score' in df_processed.columns:
                df_processed = df_processed.drop('Score', axis=1)
            
            # Aplicar codificación de variables categóricas
            if self.preprocessor and 'label_encoders' in self.preprocessor:
                label_encoders = self.preprocessor['label_encoders']
                
                for col, encoder in label_encoders.items():
                    if col in df_processed.columns:
                        try:
                            df_processed[col] = encoder.transform(df_processed[col].astype(str))
                        except ValueError:
                            # Manejar valores no vistos durante entrenamiento
                            print(f"Advertencia: Valores no vistos en columna {col}")
                            df_processed[col] = 0  # Valor por defecto
            
            # Aplicar escalado
            if self.preprocessor and 'scaler' in self.preprocessor:
                scaler = self.preprocessor['scaler']
                df_processed = pd.DataFrame(
                    scaler.transform(df_processed),
                    columns=df_processed.columns
                )
            
            return df_processed, ids
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return None, None
    
    def predict_from_csv(self, csv_path):
        """Realizar predicciones desde archivo CSV"""
        try:
            # Verificar que el modelo esté cargado
            if self.model is None:
                print("Error: Modelo no cargado")
                return None
            
            # Cargar datos
            df = pd.read_csv(csv_path)
            print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Preprocesar datos
            df_processed, ids = self.preprocess_data(df)
            
            if df_processed is None:
                return None
            
            # Realizar predicciones
            predictions = self.model.predict(df_processed.values)
            
            # Crear DataFrame de resultados
            results = pd.DataFrame({
                'ID': ids,
                'Score_Predicho': np.round(predictions, 2)
            })
            
            # Añadir datos originales relevantes si están disponibles
            original_cols = ['Customer_Age', 'Gender', 'Income_Category', 'Credit_Limit']
            for col in original_cols:
                if col in df.columns:
                    results[col] = df[col].values
            
            # Añadir categoría de riesgo basada en score
            results['Categoria_Riesgo'] = results['Score_Predicho'].apply(self._categorize_risk)
            
            print(f"Predicciones completadas: {len(results)} registros")
            return results
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None
    
    def _categorize_risk(self, score):
        """Categorizar riesgo basado en score crediticio"""
        if score >= 750:
            return "Excelente"
        elif score >= 700:
            return "Muy Bueno"
        elif score >= 650:
            return "Bueno"
        elif score >= 600:
            return "Regular"
        elif score >= 550:
            return "Malo"
        else:
            return "Muy Malo"
    
    def predict_single(self, customer_data):
        """Realizar predicción para un solo cliente"""
        try:
            if self.model is None:
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame([customer_data])
            
            # Preprocesar
            df_processed, _ = self.preprocess_data(df)
            
            if df_processed is None:
                return None
            
            # Predecir
            prediction = self.model.predict(df_processed.values)[0]
            
            return {
                'score': round(prediction, 2),
                'risk_category': self._categorize_risk(prediction)
            }
            
        except Exception as e:
            print(f"Error en predicción individual: {e}")
            return None
