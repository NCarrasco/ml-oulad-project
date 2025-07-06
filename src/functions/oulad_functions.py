import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score, classification_report

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
    """Clase para interpretación de resultados y métricas - VERSIÓN CORREGIDA."""
    def __init__(self):
        pass

    def export_metrics(self, y_test, y_pred, path: str):
        """
        Exportar métricas corregidas para problemas multiclase.
        Maneja tanto clasificación binaria como multiclase.
        """
        # Exportar y_test, y_pred a CSV
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        df.to_csv(os.path.join(path, 'y_test_y_pred.csv'), index=False)
        
        # Análisis de clases
        n_classes = len(np.unique(y_test))
        unique_classes = sorted(np.unique(y_test))
        
        print(f"🔍 Detectadas {n_classes} clases: {unique_classes}")
        
        # Matriz de confusión completa
        cm = confusion_matrix(y_test, y_pred)
        
        # Cálculo de métricas según tipo de problema
        if n_classes == 2:
            # Problema binario - usar cálculo original
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            avg_method = 'binary'
        else:
            # Problema multiclase - cálculo agregado
            # Para multiclase, TP/FP/TN/FN se calculan de forma agregada
            tp_per_class = np.diag(cm)  # Diagonal = verdaderos positivos por clase
            fp_per_class = cm.sum(axis=0) - np.diag(cm)  # Falsos positivos por clase
            fn_per_class = cm.sum(axis=1) - np.diag(cm)  # Falsos negativos por clase
            
            # Sumar para obtener totales
            tp = tp_per_class.sum()
            fp = fp_per_class.sum()
            fn = fn_per_class.sum()
            tn = cm.sum() - (tp + fp + fn)
            avg_method = 'macro'
        
        # Calcular métricas estándar
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        
        # MSE y R² (útiles para análisis numérico de clases)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # DataFrame con métricas principales
        metrics = pd.DataFrame([{
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'f1_score': f1, 'accuracy': acc, 'precision': prec, 'recall': rec,
            'mse': mse, 'r2': r2, 'n_classes': n_classes,
            'avg_method': avg_method
        }])
        
        # Guardar métricas principales
        metrics.to_csv(os.path.join(path, 'metrics_manual.csv'), index=False)
        
        # Generar reporte detallado por clase
        try:
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(path, 'classification_report_detailed.csv'))
            
            print(f"✅ Reporte por clase guardado en: classification_report_detailed.csv")
        except Exception as e:
            print(f"⚠️ No se pudo generar reporte detallado: {e}")
        
        # Mostrar resumen en consola
        print(f"\n📊 Métricas Calculadas ({avg_method} average):")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall: {rec:.4f}")
        print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        return cm, metrics

    def plot_confusion_matrix(self, y_test, y_pred, path: str):
        """Matriz de confusión mejorada con etiquetas de clase."""
        cm = confusion_matrix(y_test, y_pred)
        
        # Determinar etiquetas de clase
        unique_classes = sorted(np.unique(y_test))
        
        # Mapeo de clases para OULAD (ajustar según tu codificación)
        class_mapping = {
            0: "Fail", 1: "Pass", 2: "Withdrawn", 3: "Distinction"
        }
        
        # Usar mapeo si está disponible, sino usar números
        if all(cls in class_mapping for cls in unique_classes):
            labels = [class_mapping[cls] for cls in unique_classes]
        else:
            labels = [f"Class {cls}" for cls in unique_classes]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusión - Clasificación OULAD')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Matriz de confusión guardada: confusion_matrix.png")

    def plot_feature_importances(self, model, feature_names, path: str, top_n: int = 15):
        """Feature importance mejorado con límite de variables mostradas."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Limitar a top_n features para mejor visualización
            top_indices = indices[:top_n]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Variables de Mayor Importancia')
            bars = plt.bar(range(len(top_importances)), top_importances, align='center')
            plt.xticks(range(len(top_importances)), top_features, rotation=45, ha='right')
            plt.ylabel('Importancia')
            
            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'feature_importances.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar importancias en CSV
            importance_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_importances
            })
            importance_df.to_csv(os.path.join(path, 'feature_importances.csv'), index=False)
            
            print(f"✅ Feature importance guardado: feature_importances.png y .csv")
        else:
            print("⚠️ El modelo no tiene feature_importances_. Usa un modelo de árbol.")

    def export_model_summary(self, model_results: dict, path: str):
        """Exportar resumen completo de todos los modelos entrenados."""
        summary_df = pd.DataFrame(model_results).transpose()
        summary_df.to_csv(os.path.join(path, 'models_summary.csv'))
        
        print(f"✅ Resumen de modelos guardado: models_summary.csv")