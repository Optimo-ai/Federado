"""
Utilidades para crear visualizaciones de los resultados
"""
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np
import base64
import io

def create_results_plots(df_results):
    """Crear gráficas de los resultados del entrenamiento"""
    plots = {}
    
    try:
        # Gráfica 1: Comparación de MAE por modelo y estrategia
        plots['mae_comparison'] = create_mae_comparison_plot(df_results)
        
        # Gráfica 2: Comparación de R² por técnica de privacidad
        plots['r2_privacy'] = create_r2_privacy_plot(df_results)
        
        # Gráfica 3: Tiempo de entrenamiento por modelo
        plots['training_time'] = create_training_time_plot(df_results)
        
        # Gráfica 4: Heatmap de rendimiento
        plots['performance_heatmap'] = create_performance_heatmap(df_results)
        
    except Exception as e:
        print(f"Error creando gráficas: {e}")
        plots['error'] = str(e)
    
    return plots

def create_mae_comparison_plot(df_results):
    """Crear gráfica de comparación de MAE"""
    try:
        # Agrupar por modelo y estrategia de agregación
        grouped = df_results.groupby(['model_type', 'aggregation_strategy'])['final_mae'].mean().reset_index()
        
        fig = go.Figure()
        
        strategies = grouped['aggregation_strategy'].unique()
        models = grouped['model_type'].unique()
        
        for strategy in strategies:
            strategy_data = grouped[grouped['aggregation_strategy'] == strategy]
            
            fig.add_trace(go.Bar(
                name=strategy.upper(),
                x=strategy_data['model_type'],
                y=strategy_data['final_mae'],
                text=np.round(strategy_data['final_mae'], 3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Comparación de MAE por Modelo y Estrategia de Agregación',
            xaxis_title='Tipo de Modelo',
            yaxis_title='MAE (Mean Absolute Error)',
            barmode='group',
            height=400
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
        
    except Exception as e:
        print(f"Error en gráfica MAE: {e}")
        return None

def create_r2_privacy_plot(df_results):
    """Crear gráfica de R² por técnica de privacidad"""
    try:
        # Agrupar por técnica de privacidad
        grouped = df_results.groupby('privacy_technique')['final_r2'].mean().reset_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=grouped['privacy_technique'],
                y=grouped['final_r2'],
                text=np.round(grouped['final_r2'], 3),
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(grouped)]
            )
        ])
        
        fig.update_layout(
            title='R² Score por Técnica de Privacidad Diferencial',
            xaxis_title='Técnica de Privacidad',
            yaxis_title='R² Score',
            height=400
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
        
    except Exception as e:
        print(f"Error en gráfica R²: {e}")
        return None

def create_training_time_plot(df_results):
    """Crear gráfica de tiempo de entrenamiento"""
    try:
        # Agrupar por modelo
        grouped = df_results.groupby('model_type')['experiment_time'].mean().reset_index()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=grouped['model_type'],
                y=grouped['experiment_time'],
                mode='markers+lines',
                marker=dict(size=10),
                text=np.round(grouped['experiment_time'], 2),
                textposition='top center'
            )
        ])
        
        fig.update_layout(
            title='Tiempo de Entrenamiento por Tipo de Modelo',
            xaxis_title='Tipo de Modelo',
            yaxis_title='Tiempo (segundos)',
            height=400
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
        
    except Exception as e:
        print(f"Error en gráfica tiempo: {e}")
        return None

def create_performance_heatmap(df_results):
    """Crear heatmap de rendimiento"""
    try:
        # Crear matriz de rendimiento (MAE) por modelo y privacidad
        pivot_data = df_results.pivot_table(
            values='final_mae', 
            index='model_type', 
            columns='privacy_technique', 
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            text=np.round(pivot_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title='Heatmap de MAE: Modelo vs Técnica de Privacidad',
            xaxis_title='Técnica de Privacidad',
            yaxis_title='Tipo de Modelo',
            height=400
        )
        
        return plotly.utils.PlotlyJSONEncoder().encode(fig)
        
    except Exception as e:
        print(f"Error en heatmap: {e}")
        return None

def create_matplotlib_plot(df_results, plot_type='mae'):
    """Crear gráfica con matplotlib (alternativa)"""
    try:
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'mae':
            grouped = df_results.groupby('model_type')['final_mae'].mean()
            plt.bar(grouped.index, grouped.values)
            plt.title('MAE Promedio por Tipo de Modelo')
            plt.ylabel('MAE')
            plt.xlabel('Tipo de Modelo')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convertir a base64 para embedding en HTML
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    except Exception as e:
        print(f"Error en matplotlib: {e}")
        return None
