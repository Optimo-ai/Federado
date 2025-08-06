#!/bin/bash

# Script para configurar la estructura de datos
echo "🏗️  Configurando estructura de directorios..."

# Crear directorios necesarios
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/metrics
mkdir -p results/models
mkdir -p uploads

echo "📁 Directorios creados:"
echo "   • data/raw/ (coloca aquí tu CreditScore_test.csv)"
echo "   • data/processed/ (datos procesados)"
echo "   • results/metrics/ (métricas de experimentos)"
echo "   • results/models/ (modelos entrenados)"
echo "   • uploads/ (archivos subidos en la web)"

echo ""
echo "📋 Próximos pasos:"
echo "   1. Copia tu archivo CreditScore_test.csv a data/raw/"
echo "   2. Ejecuta: python scripts/validate_data.py"
echo "   3. Ejecuta: python scripts/run_experiments.py"

# Verificar si el archivo ya existe
if [ -f "data/raw/CreditScore_test.csv" ]; then
    echo ""
    echo "✅ Archivo CreditScore_test.csv encontrado!"
    echo "🚀 Puedes proceder con la validación y experimentos"
else
    echo ""
    echo "⚠️  No se encontró CreditScore_test.csv en data/raw/"
    echo "📁 Por favor, copia tu archivo a esa ubicación"
fi
