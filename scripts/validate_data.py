"""
Script para validar el dataset antes del preprocesamiento
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR

def validate_dataset(filepath):
    """Validar que el dataset tenga el formato correcto"""
    print("Validando dataset...")
    
    try:
        # Cargar dataset
        df = pd.read_csv(filepath)
        # Renombrar columnas genéricas a nombres reales
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

        print(f"Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Columnas requeridas
        required_columns = [
            'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
            'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
            'Total_Relationship_Count', 'Credit_Limit', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Avg_Open_To_Buy', 'Score'
        ]
        
        # Verificar columnas
        missing_columns = []
        present_columns = []
        
        for col in required_columns:
            if col in df.columns:
                present_columns.append(col)
            else:
                missing_columns.append(col)
        
        print(f"\nColumnas presentes: {len(present_columns)}/{len(required_columns)}")
        
        if present_columns:
            print(" Columnas encontradas:")
            for col in present_columns:
                print(f"   • {col}")
        
        if missing_columns:
            print(" Columnas faltantes:")
            for col in missing_columns:
                print(f"   • {col}")
            return False
        
        # Validar datos de Score (variable objetivo)
        if 'Score' in df.columns:
            score_stats = df['Score'].describe()
            print(f"\n Estadísticas de Score:")
            print(f"   • Mínimo: {score_stats['min']:.2f}")
            print(f"   • Máximo: {score_stats['max']:.2f}")
            print(f"   • Promedio: {score_stats['mean']:.2f}")
            print(f"   • Valores nulos: {df['Score'].isnull().sum()}")
            
            # Verificar rango típico de credit score
            if score_stats['min'] < 300 or score_stats['max'] > 850:
                print(" Advertencia: Scores fuera del rango típico (300-850)")
        
        # Validar variables categóricas
        categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 
                             'Income_Category', 'Card_Category']
        
        print(f"\n  Variables categóricas:")
        for col in categorical_columns:
            if col in df.columns:
                unique_vals = df[col].nunique()
                print(f"   • {col}: {unique_vals} categorías únicas")
                if unique_vals <= 10:  # Mostrar categorías si son pocas
                    categories = df[col].value_counts().head().to_dict()
                    for cat, count in categories.items():
                        print(f"     - {cat}: {count} registros")
        
        # Validar variables numéricas
        numeric_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                          'Total_Relationship_Count', 'Credit_Limit', 'Total_Trans_Amt',
                          'Total_Trans_Ct', 'Avg_Open_To_Buy']
        
        print(f"\n Variables numéricas:")
        for col in numeric_columns:
            if col in df.columns:
                col_stats = df[col].describe()
                nulls = df[col].isnull().sum()
                print(f"   • {col}:")
                print(f"     - Rango: {col_stats['min']:.2f} - {col_stats['max']:.2f}")
                print(f"     - Promedio: {col_stats['mean']:.2f}")
                print(f"     - Nulos: {nulls}")
        
        # Verificar suficientes datos para aprendizaje federado
        min_total_samples = 3000  # 1000 por cliente
        if len(df) < min_total_samples:
            print(f"\n  Advertencia: Dataset pequeño ({len(df)} filas)")
            print(f"   • Recomendado: al menos {min_total_samples} filas")
            print(f"   • Cada cliente tendrá ~{len(df)//3} muestras")
        else:
            print(f"\n Dataset adecuado para aprendizaje federado")
            print(f"   • Total: {len(df)} filas")
            print(f"   • Por cliente: ~{len(df)//3} muestras")
        
        return True
        
    except Exception as e:
        print(f" Error validando dataset: {e}")
        return False

def main():
    """Función principal"""
    input_file = os.path.join(RAW_DATA_DIR, 'CreditScore_test.csv')
    
    if not os.path.exists(input_file):
        print(f" No se encontró el archivo: {input_file}")
        print("Coloca tu archivo 'CreditScore_test.csv' en la carpeta 'data/raw/'")
        return
    
    if validate_dataset(input_file):
        print("\n ¡Dataset válido! Puedes proceder con el preprocesamiento.")
        print("  Ejecuta: python scripts/preprocess_data.py")
    else:
        print("\n Dataset inválido. Corrige los problemas antes de continuar.")

if __name__ == "__main__":
    main()
