import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import json
import os
import warnings
import sys

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')

# Sacamos la specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# Cargar datos
X, y = load_breast_cancer(return_X_y=True)

# Sacamos vecinos, w y d
k_values = np.arange(1, 101) 
weights = ['uniform', 'distance']
metrics_distance = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'braycurtis', 'correlation']
results = []

# Entrenamos los modelos
print("\nEVALUANDO COMBINACIONES DE K, PESOS Y DISTANCIAS")

total_combinations = len(k_values) * len(weights) * len(metrics_distance)
current = 0

for k in k_values:
    for weight in weights:
        for metric in metrics_distance:
            current += 1
            
            # Crear KNN
            model = KNeighborsClassifier(
                n_neighbors=k,
                weights=weight,
                metric=metric
            )
            
            # Calcular métricas
            accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()
            precision = cross_val_score(model, X, y, cv=10, scoring=make_scorer(precision_score)).mean()
            recall = cross_val_score(model, X, y, cv=10, scoring=make_scorer(recall_score)).mean()
            f1 = cross_val_score(model, X, y, cv=10, scoring=make_scorer(f1_score)).mean()
            specificity = cross_val_score(model, X, y, cv=10, scoring=make_scorer(specificity_score)).mean()
            
            # Guardar
            result = {
                'k': int(k),
                'weight': weight,
                'metric': metric,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'specificity': float(specificity)
            }
            results.append(result)
            
            # Barra de progreso
            remaining = total_combinations - current
            percent = (current / total_combinations) * 100
            bar_length = 50
            filled = int(bar_length * current / total_combinations)
            bar = '#' * filled + '-' * (bar_length - filled)
            message = f'[{bar}] {percent:.1f}% - Completados: {current}/{total_combinations} - Faltan: {remaining}'
            sys.stdout.write('\r' + message + ' ' * 20)
            sys.stdout.flush()

print()

# Guardar resultados
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(results_dir, 'resultados_knn.csv'), index=False)
print(f"\nResultados guardados en 'results/resultados_knn.csv'")

with open(os.path.join(results_dir, 'resultados_knn.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Resultados guardados en 'results/resultados_knn.json'")

# Encontrar mejores modelos
print("MEJORES MODELOS POR MÉTRICA")

best_models = {}
for metric_name in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
    best_idx = df_results[metric_name].idxmax()
    best = df_results.loc[best_idx]
    best_models[metric_name] = best
    
    # Mapear nombre para display
    display_name = 'F1-SCORE' if metric_name == 'f1' else metric_name.upper()
    
    print(f"\nMejor {display_name}:")
    print(f"   K={int(best['k'])}, Weight={best['weight']}, Distance={best['metric']}")
    print(f"   Accuracy = {best['accuracy']:.4f}")
    print(f"   Precision = {best['precision']:.4f}")
    print(f"   Recall = {best['recall']:.4f}")
    print(f"   Specificity = {best['specificity']:.4f}")
    print(f"   F1-Score = {best['f1']:.4f}")

# Generamos graficas
print("\nGenerando graficas de estrella...")

# Tomamos el maximo de cada metrica
best_accuracy = df_results.loc[df_results['accuracy'].idxmax()]
best_precision = df_results.loc[df_results['precision'].idxmax()]
best_recall = df_results.loc[df_results['recall'].idxmax()]
best_specificity = df_results.loc[df_results['specificity'].idxmax()]
best_f1 = df_results.loc[df_results['f1'].idxmax()]

selected_models = [best_accuracy, best_precision, best_recall, best_specificity, best_f1]
labels = ['MEJOR ACCURACY', 'MEJOR PRECISION', 'MEJOR RECALL', 'MEJOR SPECIFICITY', 'MEJOR F1-SCORE']
colors = ['#2E86AB', '#F18F01', '#A23B72', '#17B169', '#C73E1D']

fig, axes = plt.subplots(2, 3, figsize=(22, 16), subplot_kw=dict(projection='polar'))
fig.suptitle('Mejores Modelos KNN Optimizados para Cada Métrica', fontsize=20, fontweight='bold', y=0.98)

axes_flat = axes.flatten()

for idx, (model, label, color) in enumerate(zip(selected_models, labels, colors)):
    ax = axes_flat[idx]

    categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    values = [
        model['accuracy'],
        model['precision'],
        model['recall'],
        model['specificity'],
        model['f1']
    ]
    
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2.5, color=color, markersize=8)
    ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='normal')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    metric_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    ax.set_title(f'{label}\n'
                 f'K={int(model["k"])}, W={model["weight"]}, D={model["metric"]}\n'
                 f'Valor={model[metric_keys[idx]]:.4f}',
                 size=12, pad=25, fontweight='bold')

axes_flat[5].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4, w_pad=3)
plt.savefig(os.path.join(results_dir, 'mejores_5_metricas_knn.png'), dpi=300, bbox_inches='tight')
print("Grafica guardada como 'results/mejores_5_metricas_knn.png'")
plt.show()

print("PROCESO COMPLETADO")