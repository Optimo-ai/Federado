"""
Inicialización de la aplicación Flask
"""
from flask import Flask
from flask_bootstrap import Bootstrap
import os
import sys

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FLASK_CONFIG

def create_app():
    """Factory para crear la aplicación Flask"""
    app = Flask(__name__)
    
    # Configuración
    app.config.update(FLASK_CONFIG)

    # Aumentar límite de archivo permitido (20 MB)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

    # Inicializar extensiones
    bootstrap = Bootstrap(app)
    
    # Registrar blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
