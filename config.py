"""
Configuración global del sistema de aprendizaje federado
"""
import os

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data','processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuración del aprendizaje federado
FEDERATED_CONFIG = {
    'num_clients': 3,
    'num_rounds': 10,
    'min_fit_clients': 3,
    'min_evaluate_clients': 3,
    'min_available_clients': 3,
}

MODELS = [
    'ols',             # Regresión lineal (mínimos cuadrados)
    'ridge',           # Regresión Ridge (L2)
    'lasso',           # Regresión Lasso (L1)
    'bayesian_ridge',  # Sustituto de Naive Bayes para regresión
    'decision_tree',   # Árbol de regresión
    'random_forest',   # Bosque aleatorio
    'knn',             # K-Nearest Neighbors
    'mlp'              # Red neuronal multicapa
]

# Estrategias de agregación
AGGREGATION_STRATEGIES = ['fedavg', 'fedmed']

# Técnicas de privacidad diferencial
PRIVACY_TECHNIQUES = ['none', 'clipping', 'noising', 'clipping_noising']

# Configuración de privacidad diferencial
PRIVACY_CONFIG = {
    'clipping_norm': 1.0,
    'noise_multiplier': 0.1,
    'epsilon': 1.0,
    'delta': 1e-5
}

# Configuración Flask
FLASK_CONFIG = {
    'SECRET_KEY': 'federated-credit-scoring-key',
    'UPLOAD_FOLDER': os.path.join(BASE_DIR, 'uploads'),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB max file size
}

# Crear directorio de uploads
os.makedirs(FLASK_CONFIG['UPLOAD_FOLDER'], exist_ok=True)
