import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

class OULADModel:
    """Clase mejorada para entrenamiento y evaluación de modelos de ML."""
    
    def __init__(self):
        # Importar algoritmos
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        self.classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'LogisticRegression': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),  # probability=True para ROC-AUC
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'GaussianNB': GaussianNB(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Bagging': BaggingClassifier(random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'MLP': MLPClassifier(max_iter=1000, random_state=42, hidden_layer_sizes=(50,), learning_rate='adaptive'),
        }
        
        # Regresores
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        
        self.regressors = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            'LinearRegression': LinearRegression(n_jobs=-1),
        }
        
        # Clustering
        from sklearn.cluster import KMeans
        self.KMeans = KMeans
        
        # Para gráficos
        self.plt = plt
        self.sns = sns

    def train_multiple_classifiers(self, X, y, output_dir: str = "results", test_size: float = 0.2) -> Dict[str, Any]:
        """
        Entrena múltiples clasificadores y compara su rendimiento.
        Requerido por el proyecto: mínimo 3 algoritmos supervisados.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import StandardScaler
        
        print(f"\n🤖 Entrenando {len(self.classifiers)} clasificadores...")
        
        # Preparar datos
        X_model = X.select_dtypes(include=[np.number])
        if X_model.shape[1] == 0:
            X_model = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, classifier in self.classifiers.items():
            print(f"   Entrenando {name}...")
            
            try:
                # Pipeline con SMOTE y escalado
                pipeline = Pipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('scaler', StandardScaler()),
                    ('classifier', classifier)
                ])
                
                # Entrenar
                pipeline.fit(X_train, y_train)
                
                # Predecir
                y_pred = pipeline.predict(X_test)
                
                # Calcular métricas
                n_classes = len(np.unique(y))
                avg_method = 'binary' if n_classes == 2 else 'macro'
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average=avg_method, zero_division=0),
                    'precision': precision_score(y_test, y_pred, average=avg_method, zero_division=0),
                    'recall': recall_score(y_test, y_pred, average=avg_method, zero_division=0),
                    'n_classes': n_classes
                }
                
                # Cross-validation para mayor robustez
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                results[name] = metrics
                
                # Guardar modelo si es el mejor hasta ahora
                if name == 'RandomForest':  # Guardar RF como referencia
                    joblib.dump(pipeline, f"{output_dir}/best_classifier_{name}.joblib")
                    
                    # Guardar predicciones del mejor modelo
                    pred_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
                    pred_df.to_csv(f"{output_dir}/y_test_y_pred_{name}.csv", index=False)
                
            except Exception as e:
                print(f"   ❌ Error con {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Guardar resumen de resultados
        results_df = pd.DataFrame(results).transpose()
        results_df.to_csv(f"{output_dir}/classifiers_comparison.csv")
        
        # Mostrar mejores resultados
        print(f"\n📊 Resumen de Clasificadores:")
        if len(results) > 0:
            for name, metrics in results.items():
                if 'accuracy' in metrics:
                    print(f"   {name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
        
        print(f"✅ Resultados guardados en: {output_dir}/classifiers_comparison.csv")
        
        return results

    def train_regressors(self, X, y, output_dir: str = "results"):
        """Entrena y evalúa regresores estándar sobre los datos proporcionados."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        print(f"\n📈 Entrenando regresores...")
        
        # Solo usar variables numéricas
        X_reg = X.select_dtypes(include=[np.number])
        if X_reg.shape[1] == 0:
            X_reg = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.2, random_state=42)
        
        # Escalar datos para regresión
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for name, regressor in self.regressors.items():
            print(f"   Entrenando {name}...")
            
            try:
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
                
                results[name] = metrics
                
                # Guardar mejor modelo
                if name == 'RandomForestRegressor':
                    joblib.dump(regressor, f"{output_dir}/best_regressor.joblib")
                    joblib.dump(scaler, f"{output_dir}/regressor_scaler.joblib")
                    
                    # Guardar predicciones
                    pred_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
                    pred_df.to_csv(f"{output_dir}/regression_preds.csv", index=False)
                
            except Exception as e:
                print(f"   ❌ Error con {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Guardar métricas
        results_df = pd.DataFrame(results).transpose()
        results_df.to_csv(f"{output_dir}/regression_metrics.csv")
        
        # Mostrar resultados
        print(f"\n📊 Resumen de Regresores:")
        for name, metrics in results.items():
            if 'r2' in metrics:
                print(f"   {name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
        
        print(f"✅ Resultados guardados en: {output_dir}/regression_metrics.csv")

    def train_clustering(self, X, output_dir: str = "results", n_clusters: int = 3):
        """Entrena un modelo de clustering KMeans y exporta etiquetas y métricas básicas."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.preprocessing import StandardScaler
        
        print(f"\n🔍 Entrenando clustering (k={n_clusters})...")
        
        # Solo usar variables numéricas
        X_clust = X.select_dtypes(include=[np.number])
        if X_clust.shape[1] == 0:
            X_clust = pd.get_dummies(X, drop_first=True)
        
        # Escalar datos para clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)
        
        # Entrenar KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calcular métricas
        sil_score = silhouette_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
        
        # Guardar resultados
        pd.DataFrame({'cluster': labels}).to_csv(f"{output_dir}/clustering_labels.csv", index=False)
        
        metrics_df = pd.DataFrame([{
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_
        }])
        metrics_df.to_csv(f"{output_dir}/clustering_metrics.csv", index=False)
        
        # Guardar modelo
        joblib.dump(kmeans, f"{output_dir}/clustering_model.joblib")
        joblib.dump(scaler, f"{output_dir}/clustering_scaler.joblib")
        
        print(f"   Silhouette Score: {sil_score:.3f}")
        print(f"   Calinski-Harabasz Score: {ch_score:.1f}")
        print(f"✅ Resultados guardados en: {output_dir}/clustering_*.csv")
        
        return labels, sil_score