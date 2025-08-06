"""
Script para ejecutar todos los experimentos de aprendizaje federado
"""
import os
import sys
import subprocess
import time

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_preprocessing():
    """Ejecutar preprocesamiento de datos"""
    print("=== Ejecutando preprocesamiento ===")
    try:
        result = subprocess.run([
            sys.executable, 'scripts/preprocess_data.py'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("Preprocesamiento completado")
            print(result.stdout)
            return True
        else:
            print(" Error en preprocesamiento")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Error ejecutando preprocesamiento: {e}")
        return False

def run_federated_training():
    """Ejecutar entrenamiento federado"""
    print("\n=== Ejecutando entrenamiento federado ===")
    try:
        result = subprocess.run([
            sys.executable, 'federated/main.py'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("Entrenamiento federado completado")
            print(result.stdout)
            return True
        else:
            print("Error en entrenamiento federado")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error ejecutando entrenamiento: {e}")
        return False

def validate_data():
    """Validar datos antes del preprocesamiento"""
    print("=== Validando datos ===")
    try:
        result = subprocess.run([
            sys.executable, 'scripts/validate_data.py'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("Validación de datos completada")
            print(result.stdout)
            return True
        else:
            print(" Error en validación de datos")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error ejecutando validación: {e}")
        return False

def main():
    """Función principal"""
    print("Iniciando experimentos completos de aprendizaje federado")
    print("=" * 60)
    
    start_time = time.time()
    
    # Paso 0: Validar datos
    if not validate_data():
        print("Falló la validación de datos. Verifica tu archivo CSV.")
        return
    
    # Paso 1: Preprocesamiento
    if not run_preprocessing():
        print("Falló el preprocesamiento. Abortando.")
        return
    
    # Paso 2: Entrenamiento federado
    if not run_federated_training():
        print("Falló el entrenamiento federado. Abortando.")
        return
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(" ¡Todos los experimentos completados exitosamente!")
    print(f"  Tiempo total: {total_time:.2f} segundos")
    print(f" Datos procesados de tu archivo CreditScore_test.csv")
    print("\n Ahora puedes:")
    print("   1. Ejecutar la aplicación Flask: python run.py")
    print("   2. Ir a http://localhost:5000 para ver la interfaz")
    print("   3. Usar /predict para hacer predicciones")
    print("   4. Usar /results para ver los resultados del entrenamiento")

if __name__ == "__main__":
    main()
