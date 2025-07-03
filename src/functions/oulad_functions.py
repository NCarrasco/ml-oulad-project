import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score

class OULADDataLoader:
    """Clase para cargar los datos OULAD y datasets complementarios."""
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_vle(self, filename: str = "Full_vle_train.csv") -> pd.DataFrame:
        path = f"{self.base_path}/{filename}"
        return pd.read_csv(path, encoding='latin1')

    def load_assess(self, filename: str = "Full_assess_train.csv") -> pd.DataFrame:
        path = f"{self.base_path}/{filename}"
        return pd.read_csv(path, encoding='latin1')

class OULADInterpreter:
    """Clase para interpretación de resultados y métricas."""
    def __init__(self):
        pass

    def export_metrics(self, y_test, y_pred, path: str):
        # Exportar y_test, y_pred y métricas manuales a CSV
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        df.to_csv(os.path.join(path, 'y_test_y_pred.csv'), index=False)
        # Cálculo manual de métricas
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
        # Detectar si el problema es binario o multiclase
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            avg = 'binary'
        else:
            avg = 'macro'
        f1 = f1_score(y_test, y_pred, average=avg)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = pd.DataFrame([{'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'f1_score': f1, 'accuracy': acc, 'precision': prec, 'recall': rec, 'mse': mse, 'r2': r2}])
        metrics.to_csv(os.path.join(path, 'metrics_manual.csv'), index=False)
        return cm, metrics

    def plot_confusion_matrix(self, y_test, y_pred, path: str):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.savefig(os.path.join(path, 'confusion_matrix.png'))
        plt.close()

    def plot_feature_importances(self, model, feature_names, path: str):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10,6))
            plt.title('Importancia de variables')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'feature_importances.png'))
            plt.close()
