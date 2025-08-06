"""
Rutas de la aplicación Flask
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.utils import secure_filename
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.utils

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from federated.utils.prediction import PredictionService
from federated.utils.visualization import create_results_plots



main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Página de predicción"""
    if request.method == 'POST':
        try:
            # Verificar si se subió un archivo
            if 'file' not in request.files:
                flash('No se seleccionó ningún archivo', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No se seleccionó ningún archivo', 'error')
                return redirect(request.url)
            
            if file and file.filename.lower().endswith('.csv'):
                # Guardar archivo temporalmente
                filename = secure_filename(file.filename)
                filepath = os.path.join(FLASK_CONFIG['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Realizar predicciones
                prediction_service = PredictionService()
                results = prediction_service.predict_from_csv(filepath)
                
                if results is not None:
                    # Convertir a HTML para mostrar
                    results_html = results.to_html(classes='table table-striped table-hover', 
                                                 table_id='results-table', escape=False)
                    
                    # Guardar resultados para descarga
                    results_path = os.path.join(FLASK_CONFIG['UPLOAD_FOLDER'], f'predictions_{filename}')
                    results.to_csv(results_path, index=False)
                    
                    return render_template('predict.html', 
                                         results_html=results_html,
                                         download_file=f'predictions_{filename}',
                                         num_predictions=len(results))
                else:
                    flash('Error procesando el archivo. Verifica el formato.', 'error')
            else:
                flash('Por favor, sube un archivo CSV válido', 'error')
                
        except Exception as e:
            flash(f'Error procesando archivo: {str(e)}', 'error')
    
    return render_template('predict.html')

@main_bp.route('/results')
def results():
    """Página de resultados del entrenamiento"""
    try:
        # Cargar resultados
        results_path = os.path.join(RESULTS_DIR, 'resumen_resultados.csv')
        
        if not os.path.exists(results_path):
            flash('No se encontraron resultados de entrenamiento. Ejecuta primero los experimentos.', 'warning')
            return render_template('results.html', no_results=True)
        
        df_results = pd.read_csv(results_path)
        
        # Convertir a HTML
        results_html = df_results.to_html(classes='table table-striped table-hover', 
                                        table_id='results-table', escape=False)
        
        # Crear gráficas
        plots = create_results_plots(df_results)
        
        return render_template('results.html', 
                             results_html=results_html,
                             plots=plots,
                             num_experiments=len(df_results))
        
    except Exception as e:
        flash(f'Error cargando resultados: {str(e)}', 'error')
        return render_template('results.html', no_results=True)

@main_bp.route('/download/<filename>')
def download_file(filename):
    """Descargar archivo de resultados"""
    try:
        filepath = os.path.join(FLASK_CONFIG['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash('Archivo no encontrado', 'error')
            return redirect(url_for('main.predict'))
    except Exception as e:
        flash(f'Error descargando archivo: {str(e)}', 'error')
        return redirect(url_for('main.predict'))

@main_bp.route('/api/results')
def api_results():
    """API endpoint para obtener resultados en JSON"""
    try:
        results_path = os.path.join(RESULTS_DIR, 'resumen_resultados.csv')
        
        if not os.path.exists(results_path):
            return jsonify({'error': 'No results found'}), 404
        
        df_results = pd.read_csv(results_path)
        return jsonify(df_results.to_dict('records'))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
