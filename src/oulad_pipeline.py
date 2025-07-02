"""
oulad_pipeline.py

Pipeline orientado a objetos para el flujo OSEMN sobre el dataset OULAD.

Colaboradores y documentación de contribución
============================================
- Norman Carrasco (Coordinador, Programador principal):
  * Refactorización a POO, modularización, integración con MySQL, desarrollo de clases principales (OULADDBConnector, OULADPreprocessor, OULADEDA, OULADModel, OULADInterpreter).
  * Implementación de limpieza, feature engineering, EDA, modelado supervisado/no supervisado, métricas y visualizaciones.
  * Separación de lógica de conexión y consultas SQL, configuración de entorno y dependencias.
  * Documentación técnica y validación de entregables.
- [Nombre Colaborador 2] (Rol):
  * [Descripción de aportes: revisión de EDA, validación de hipótesis, apoyo en interpretación de resultados, redacción de anexos, etc.]
- [Nombre Colaborador 3] (Rol):
  * [Descripción de aportes: apoyo en revisión bibliográfica, pruebas de reproducibilidad, edición de README, etc.]

Notas:
- Todos los integrantes participaron en la discusión de hipótesis, revisión de resultados y redacción del informe final.
- El pipeline es reproducible y cumple con los requerimientos académicos del proyecto final de maestría.
- Para reproducir el entorno, instalar dependencias desde requirements.txt y configurar la conexión MySQL en config/settings.py.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sqlalchemy import create_engine
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from db_connector import OULADDBConnector
from db_queries import read_table, run_query, list_tables
from preprocess import OULADPreprocessor
from eda import OULADEDA
from models.model import OULADModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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

# Ejemplo de uso del pipeline (main)
def main(max_rows: int = 100_000):
    output_dir = "results"  # Carpeta de salida para métricas y gráficos
    os.makedirs(output_dir, exist_ok=True)  # Crear carpeta si no existe
    print("\nConectando a la base de datos y cargando datos...")
    db = OULADDBConnector()
    db.test_connection()
    print("Cargando datos desde la tabla 'studentVle' y uniendo con 'studentInfo' para obtener 'final_result'...")
    df_vle = read_table("studentVle", db)
    df_info = read_table("studentInfo", db)
    # Unir por id_student y code_presentation
    df_merged = pd.merge(df_vle, df_info[["id_student", "code_presentation", "final_result"]],
                        on=["id_student", "code_presentation"], how="inner")
    if max_rows is not None and len(df_merged) > max_rows:
        df_merged = df_merged.sample(n=max_rows, random_state=42)
        print(f"\nMuestreo aleatorio aplicado: usando {max_rows:,} registros de {len(df_merged):,} disponibles.")
    else:
        print(f"Leídos {len(df_merged):,} registros tras el join.")
    preprocessor = OULADPreprocessor()
    print("\nLimpieza y preprocesamiento de datos...")
    df_clean, num_vars = preprocessor.clean(df_merged)
    print("Feature engineering...")
    df_feat = preprocessor.feature_engineering(df_clean)

    # DEBUG: Revisar variable objetivo antes de modelado
    print("\n[DEBUG] Columnas disponibles en df_feat:", df_feat.columns)
    if 'final_result' in df_feat.columns:
        print("[DEBUG] Conteo de valores en final_result:\n", df_feat['final_result'].value_counts())
    if 'procatina' in df_feat.columns:
        print("[DEBUG] Conteo de valores en procatina:\n", df_feat['procatina'].value_counts())

    eda = OULADEDA()
    print("\nAnálisis exploratorio de datos (EDA)...")
    for step, func in tqdm(list(enumerate([
        lambda: eda.univariate_analysis(df_feat, output_dir=output_dir),
        lambda: eda.bivariate_analysis(df_feat, output_dir=output_dir),
        lambda: eda.plot_boxplots(df_feat, output_dir=output_dir)
    ], 1)), desc="EDA", unit="tarea"):
        func()
    modeler = OULADModel()
    # Ejemplo: clasificación binaria (ajustar target según el caso)
    if 'procatina' in df_feat.columns:
        X = df_feat.drop(columns=['procatina'])
        y = df_feat['procatina']
    else:
        X = df_feat.drop(columns=['final_result']) if 'final_result' in df_feat.columns else df_feat
        y = df_feat['final_result'] if 'final_result' in df_feat.columns else None
    if y is not None:
        print("\nEntrenando clasificadores...")
        # Asegurar solo variables numéricas para modelado
        X_model = X.select_dtypes(include=[np.number])
        if X_model.shape[1] == 0:
            # Si no hay variables numéricas, convertir categóricas a dummies
            X_model = pd.get_dummies(X, drop_first=True)
        for _ in tqdm([0], desc="Clasificadores", unit="modelo"):
            X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42, stratify=y)
            pipe = Pipeline([
                ('smote', SMOTE()),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            interpreter = OULADInterpreter()
            cm, metrics = interpreter.export_metrics(y_test, y_pred, output_dir)
            interpreter.plot_confusion_matrix(y_test, y_pred, output_dir)
            classifier = pipe.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                interpreter.plot_feature_importances(classifier, X_model.columns, output_dir)
                print("feature_importances.png generado en 'results/'.")
            else:
                print("El modelo no tiene feature_importances_. Usa un modelo de árbol para obtener la importancia de variables.")
            print("\nMatriz de confusión y métricas manuales exportadas a 'results/'.")
        print("\nEntrenando regresores...")
        for _ in tqdm([0], desc="Regresores", unit="modelo"):
            modeler.train_regressors(X_model, y, output_dir=output_dir)
        print("\nEntrenando clustering...")
        for _ in tqdm([0], desc="Clustering", unit="modelo"):
            modeler.train_clustering(X_model, output_dir=output_dir)
    else:
        print("No se encontró variable objetivo para modelado.")
    print("\nPrimeras filas del DataFrame cargado desde la base de datos:")
    print(df_vle.head())
    print("\nEstructura de las tablas en la base de datos:")
    tables = list_tables(db)
    for table in tqdm(tables, desc="Tablas", unit="tabla"):
        print(f"\nTabla: {table}")
        df_info = read_table(table, db).head(0)
        print(df_info.dtypes)
    print("\n\033[92m¡Pipeline finalizado exitosamente!\033[0m")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline OULAD optimizado para datasets grandes.")
    parser.add_argument('--max_rows', type=int, default=100_000, help='Número máximo de filas a usar (default: 100000)')
    args = parser.parse_args()
    main(max_rows=args.max_rows)
