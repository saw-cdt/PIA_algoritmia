# Búsqueda de Hiperparámetros para K-Nearest Neighbors

Proyecto de análisis y clasificación de datos médicos utilizando el algoritmo K-Nearest Neighbors (KNN) con optimización exhaustiva de hiperparámetros.

## Descripción

Este proyecto implementa un sistema automatizado para encontrar la mejor configuración de hiperparámetros en el algoritmo KNN aplicado al dataset de cáncer de mama. Se evalúan 1,600 combinaciones diferentes de parámetros y se generan visualizaciones que ayudan a identificar el modelo óptimo según diferentes métricas de evaluación.

## Características

- **Búsqueda exhaustiva de hiperparámetros**: 
  - 100 valores de K (vecinos más cercanos)
  - 2 tipos de pesos (uniforme y por distancia)
  - 8 métricas de distancia diferentes

- **Evaluación completa**:
  - Validación cruzada con 10 pliegues
  - 5 métricas de rendimiento (Accuracy, Precision, Recall, Specificity, F1-Score)

- **Visualización**:
  - Gráficas de radar mostrando los mejores modelos para cada métrica
  - Exportación de resultados en formato CSV y JSON

- **Interfaz amigable**:
  - Barra de progreso en tiempo real
  - Organización automática de resultados

## Requisitos

```
Python 3.x
numpy
pandas
matplotlib
scikit-learn
```

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/saw-cdt/PIA_algoritmia.git
cd PIA_algoritmia
```

2. Instala las dependencias:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Uso

Ejecuta el script principal:
```bash
python main.py
```

El programa generará automáticamente:
- `results/resultados_knn.csv`: Todos los resultados en formato CSV
- `results/resultados_knn.json`: Todos los resultados en formato JSON
- `results/mejores_5_metricas_knn.png`: Visualización de los 5 mejores modelos

## Métricas de Distancia Implementadas

- Euclidiana
- Manhattan
- Minkowski
- Chebyshev
- Coseno
- Canberra
- Bray-Curtis
- Correlación

## Resultados

El sistema identifica automáticamente el mejor modelo para cada métrica:
- **Accuracy**: Mejor balance general
- **Precision**: Minimiza falsos positivos
- **Recall**: Maximiza detección de casos positivos
- **Specificity**: Maximiza identificación de casos negativos
- **F1-Score**: Balance entre precisión y recall

## Aplicación Práctica

Este proyecto fue desarrollado pensando en aplicaciones médicas, específicamente en el diagnóstico de cáncer de mama. Cada modelo optimizado sirve para diferentes necesidades clínicas:

- El modelo con mejor **Recall** es ideal para screening, asegurando que no se pierda ningún caso
- El modelo con mejor **Precision** reduce estudios innecesarios y ansiedad del paciente
- El modelo con mejor **F1-Score** ofrece un balance óptimo para uso general

## Aprendizajes

Durante el desarrollo de este proyecto aprendí:
- Implementación y optimización de algoritmos de Machine Learning
- Técnicas de validación cruzada para evaluar modelos
- Manejo de diferentes métricas de distancia
- Visualización de datos con gráficas de radar
- Exportación y gestión de resultados en múltiples formatos
- Importancia de la selección de hiperparámetros en el rendimiento del modelo

## Autor

Desarrollado por Alejanro Quintanilla Leal
