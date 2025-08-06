# Sistema de Aprendizaje Federado para Predicción de Score Crediticio

Un sistema completo de aprendizaje federado que implementa múltiples modelos de regresión con técnicas de privacidad diferencial para la predicción de scores crediticios.

## 🎯 Características Principales

- **Aprendizaje Federado**: Implementado con Flower (FLWR)
- **Múltiples Modelos**: Ridge, Lasso, Random Forest, MLP
- **Privacidad Diferencial**: Clipping, Noising, y combinaciones
- **Estrategias de Agregación**: FedAvg y FedMed
- **Interfaz Web**: Flask con Bootstrap para predicciones
- **Visualizaciones**: Gráficas interactivas con Plotly

## 🏗️ Estructura del Proyecto

\`\`\`
score_buro_federado/
├── app/                    # Aplicación Flask
│   ├── __init__.py
│   ├── routes.py
│   ├── templates/
│   └── utils/
├── federated/              # Lógica de aprendizaje federado
│   ├── client.py
│   ├── server.py
│   ├── main.py
│   ├── models/
│   ├── aggregation/
│   └── privacy/
├── scripts/                # Scripts de utilidad
│   ├── preprocess_data.py
│   └── run_experiments.py
├── data/                   # Datos
│   ├── raw/
│   └── processed/
├── results/                # Resultados y modelos
│   ├── metrics/
│   └── models/
├── config.py              # Configuración global
├── run.py                 # Ejecutar Flask
├── requirements.txt
└── README.md
\`\`\`

## 🚀 Instalación y Uso

### 1. Preparar tus Datos

Coloca tu archivo `CreditScore_test.csv` en la carpeta `data/raw/`:

\`\`\`bash
mkdir -p data/raw
# Copia tu archivo CreditScore_test.csv a data/raw/
\`\`\`

**Formato requerido del CSV:**
- `ID`: identificador del cliente (único)
- `Customer_Age`: edad del cliente
- `Gender`: género
- `Dependent_count`: número de dependientes
- `Education_Level`: nivel educativo
- `Marital_Status`: estado civil
- `Income_Category`: categoría de ingreso
- `Card_Category`: tipo de tarjeta
- `Months_on_book`: tiempo como cliente
- `Total_Relationship_Count`: nº total de productos con el banco
- `Credit_Limit`: límite de crédito
- `Total_Trans_Amt`: monto total de transacciones
- `Total_Trans_Ct`: número total de transacciones
- `Avg_Open_To_Buy`: promedio disponible para compras
- `Score`: score de crédito (columna objetivo, regresión)

### 2. Validar Datos (Opcional pero Recomendado)

\`\`\`bash
python scripts/validate_data.py
\`\`\`

### 3. Instalar Dependencias

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Ejecutar Experimentos Completos

\`\`\`bash
python scripts/run_experiments.py
\`\`\`

Este script ejecutará automáticamente:
- Validación de datos
- Preprocesamiento de datos reales
- División en 3 clientes simulados
- Entrenamiento federado con todas las combinaciones
- Generación del modelo final

### 5. Ejecutar Aplicación Web

\`\`\`bash
python run.py
\`\`\`

Accede a: `http://localhost:5000`

## 📊 Funcionalidades de la Interfaz Web

### Página Principal (`/`)
- Información del sistema
- Características técnicas
- Guía de uso

### Predicción (`/predict`)
- Subir archivo CSV
- Generar predicciones de score crediticio
- Descargar resultados
- Categorización de riesgo

### Resultados (`/results`)
- Tabla de resultados de experimentos
- Gráficas interactivas
- Métricas de rendimiento
- Comparaciones entre modelos

## 🔧 Configuración

### Modelos Disponibles
- **Ridge Regression**: Regresión lineal con regularización L2
- **Lasso Regression**: Regresión lineal con regularización L1
- **Random Forest**: Ensemble de árboles de decisión
- **MLP**: Red neuronal multicapa

### Estrategias de Agregación
- **FedAvg**: Promedio ponderado por número de muestras
- **FedMed**: Mediana de parámetros

### Técnicas de Privacidad Diferencial
- **None**: Sin privacidad
- **Clipping**: Limitación de gradientes
- **Noising**: Adición de ruido gaussiano
- **Clipping + Noising**: Combinación de ambas técnicas

## 📈 Métricas Evaluadas

- **MAE**: Error Absoluto Medio
- **MSE**: Error Cuadrático Medio
- **R²**: Coeficiente de determinación
- **Tiempo de entrenamiento**: Por cliente y global
- **Tiempo de inferencia**: Para predicciones

## 🔒 Privacidad y Seguridad

El sistema implementa privacidad diferencial con:
- Clipping de gradientes con norma configurable
- Ruido gaussiano calibrado
- Presupuesto de privacidad (ε, δ)
- Composición de privacidad a través de rondas

## 📝 Formato de Datos

### Archivo CSV de Entrada
Debe contener las siguientes columnas:

- `Customer_Age`: Edad del cliente
- `Gender`: Género (M/F)
- `Dependent_count`: Número de dependientes
- `Education_Level`: Nivel educativo
- `Marital_Status`: Estado civil
- `Income_Category`: Categoría de ingresos
- `Card_Category`: Tipo de tarjeta
- `Months_on_book`: Meses como cliente
- `Total_Relationship_Count`: Productos con el banco
- `Credit_Limit`: Límite de crédito
- `Total_Trans_Amt`: Monto total de transacciones
- `Total_Trans_Ct`: Número de transacciones
- `Avg_Open_To_Buy`: Promedio disponible para compras

### Salida de Predicción
- `Score_Predicho`: Score crediticio (300-850)
- `Categoria_Riesgo`: Excelente, Muy Bueno, Bueno, Regular, Malo, Muy Malo

## 🛠️ Desarrollo

### Ejecutar Solo Preprocesamiento
\`\`\`bash
python scripts/preprocess_data.py
\`\`\`

### Ejecutar Solo Entrenamiento Federado
\`\`\`bash
python federated/main.py
\`\`\`

### Modo Debug Flask
\`\`\`bash
export FLASK_ENV=development
python run.py
\`\`\`

## 📊 Resultados Esperados

El sistema genera:
- `resumen_resultados.csv`: Métricas de todos los experimentos
- `modelo_final.pkl`: Mejor modelo entrenado
- Gráficas comparativas de rendimiento
- Análisis de trade-offs privacidad vs. precisión

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

Si encuentras problemas:
1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de que los datos estén preprocesados
3. Revisa los logs de error en la consola
4. Abre un issue en el repositorio

---

**Desarrollado con ❤️ para el aprendizaje federado y la privacidad diferencial**
