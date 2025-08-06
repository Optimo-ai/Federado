"""
Script para preprocesar el dataset y dividirlo en clientes simulados
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

class DataPreprocessor:
    """Clase para preprocesar y dividir el dataset"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Cargar dataset original"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e:
            print(f"Error cargando dataset: {e}")
            return None
    
    def clean_data(self, df):
        """Limpiar y preparar los datos reales"""
        print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")
    # Detectar columnas anónimas y renombrar si es necesario
    
        if df.columns[0].startswith("x") and df.columns[-1] == "y":
         print("Detectadas columnas anónimas. Aplicando nombres reales...")
        # Define el mapeo esperable basado en tu estructura real
        mapping = {
            'x001': 'ID',
            'x002': 'Customer_Age',
            'x003': 'Gender',
            'x004': 'Dependent_count',
            'x005': 'Education_Level',
            'x006': 'Marital_Status',
            'x007': 'Income_Category',
            'x008': 'Card_Category',
            'x009': 'Months_on_book',
            'x010': 'Total_Relationship_Count',
            'x011': 'Credit_Limit',
            'x012': 'Total_Trans_Amt',
            'x013': 'Total_Trans_Ct',
            'x014': 'Avg_Open_To_Buy',
            'y': 'Score'
        }
        df.rename(columns=mapping, inplace=True)
     
        # Mostrar información básica del dataset
        print("Columnas encontradas:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Verificar columnas requeridas
        required_columns = [
            'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
            'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
            'Total_Relationship_Count', 'Credit_Limit', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Avg_Open_To_Buy', 'Score'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Columnas faltantes: {missing_columns}")
            print("Verifica que tu CSV tenga todas las columnas requeridas")
        
        # Eliminar filas con valores nulos en Score (variable objetivo)
        initial_rows = len(df)
        df = df.dropna(subset=['Score'])
        if len(df) < initial_rows:
            print(f"🧹 Eliminadas {initial_rows - len(df)} filas con Score nulo")
        
        # Eliminar duplicados basados en ID si existe
        if 'ID' in df.columns:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['ID'])
            if len(df) < initial_rows:
                print(f"🧹 Eliminados {initial_rows - len(df)} duplicados basados en ID")
            # Eliminar columna ID para el entrenamiento
            df = df.drop('ID', axis=1)
        
        # Manejar valores nulos en otras columnas
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(" Valores nulos encontrados:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"   {col}: {count} valores nulos")
            
            # Rellenar valores nulos según el tipo de columna
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['object']:
                        # Variables categóricas: usar moda
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                        df[col].fillna(mode_value, inplace=True)
                        print(f"   {col}: rellenado con '{mode_value}'")
                    else:
                        # Variables numéricas: usar mediana
                        median_value = df[col].median()
                        df[col].fillna(median_value, inplace=True)
                        print(f"   {col}: rellenado con {median_value}")
        
        # Verificar tipos de datos
        print("\nTipos de datos:")
        for col in df.columns:
            dtype = df[col].dtype
            unique_vals = df[col].nunique()
            print(f"   {col}: {dtype} ({unique_vals} valores únicos)")
        
        print(f"\nDatos después de limpieza: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Verificar que tenemos suficientes datos para 3 clientes
        min_samples_per_client = 1000
        if len(df) < min_samples_per_client * 3:
            print(f"   Advertencia: Dataset pequeño ({len(df)} filas)")
            print(f"   Cada cliente tendrá aproximadamente {len(df)//3} muestras")
            print(f"   Recomendado: al menos {min_samples_per_client * 3} filas total")
        
        return df
    
    def encode_categorical_features(self, df):
        """Codificar variables categóricas"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'Score']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
                
        return df
    
    def scale_features(self, df, fit=True):
        """Escalar características numéricas"""
        feature_columns = [col for col in df.columns if col != 'Score']
        
        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = self.scaler.transform(df[feature_columns])
            
        return df
    
    def split_into_clients(self, df, num_clients=3, min_samples=1000):
        """Dividir dataset en clientes simulados"""
        # Asegurar que cada cliente tenga al menos min_samples
        total_samples = len(df)
        if total_samples < num_clients * min_samples:
            print(f"Advertencia: Dataset pequeño. Cada cliente tendrá menos de {min_samples} muestras")
        
        # Mezclar datos aleatoriamente
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Dividir en chunks
        client_datasets = []
        chunk_size = len(df_shuffled) // num_clients
        
        for i in range(num_clients):
            start_idx = i * chunk_size
            if i == num_clients - 1:  # Último cliente obtiene las muestras restantes
                end_idx = len(df_shuffled)
            else:
                end_idx = (i + 1) * chunk_size
                
            client_data = df_shuffled.iloc[start_idx:end_idx].copy()
            client_datasets.append(client_data)
            print(f"Cliente {i+1}: {len(client_data)} muestras")
            
        return client_datasets
    
    def save_client_datasets(self, client_datasets):
        """Guardar datasets de clientes"""
        for i, client_data in enumerate(client_datasets):
            filename = f"banco{i}.csv"
            filepath = os.path.join(PROCESSED_DATA_DIR, filename)
            client_data.to_csv(filepath, index=False)
            print(f"Guardado: {filepath}")
    
    def process_dataset(self, input_filepath):
        """Proceso completo de preprocesamiento"""
        print("=== Iniciando preprocesamiento ===")
        
        # Cargar datos
        df = self.load_data(input_filepath)
        if df is None:
            return False
            
        # Limpiar datos
        df = self.clean_data(df)
        
        # Codificar variables categóricas
        df = self.encode_categorical_features(df)
        
        # Escalar características
        df = self.scale_features(df, fit=True)
        
        # Dividir en clientes
        client_datasets = self.split_into_clients(df)
        
        # Guardar datasets
        self.save_client_datasets(client_datasets)
        
        # Guardar preprocessor para uso posterior
        import joblib
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }, os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl'))
        
        print("=== Preprocesamiento completado ===")
        return True

def main():
    """Función principal"""
    # Verificar que existe el archivo de datos real
    input_file = os.path.join(RAW_DATA_DIR, 'CreditScore_test.csv')
    
    if not os.path.exists(input_file):
        print(f" Error: No se encontró el archivo {input_file}")
        print(" Por favor, coloca tu archivo 'CreditScore_test.csv' en la carpeta 'data/raw/'")
        print(" El archivo debe contener las siguientes columnas:")
        required_columns = [
            'ID', 'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
            'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
            'Total_Relationship_Count', 'Credit_Limit', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Avg_Open_To_Buy', 'Score'
        ]
        for i, col in enumerate(required_columns, 1):
            print(f"   {i:2d}. {col}")
        return False
    
    # Procesar dataset real
    preprocessor = DataPreprocessor()
    success = preprocessor.process_dataset(input_file)
    
    if success:
        print(" ¡Preprocesamiento exitoso con datos reales!")
        print(" Datos divididos en 3 clientes simulados (bancos)")
        print(" Ahora puedes ejecutar: python federated/main.py")
    else:
        print(" Error en el preprocesamiento")
        return False
    
    return True

if __name__ == "__main__":
    main()
