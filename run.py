"""
Script principal para ejecutar la aplicación Flask
"""
import os
import sys

# Añadir directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("=== Sistema de Aprendizaje Federado - Credit Scoring ===")
    print("Iniciando servidor Flask...")
    print("Accede a: http://localhost:5000")
    print("Ctrl+C para detener el servidor")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
