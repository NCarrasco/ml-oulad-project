# Pipeline OULAD - Reporte Resumen

**Fecha:** 2025-07-05 11:52:01

## Configuración del Pipeline

- **Datos limpios:** ✅
- **Máximo features:** 50
- **Total registros:** 1,000,000
- **Total features:** 7

## Archivos Generados

- `classification_report.csv`
- `confusion_matrix.png`
- `feature_importance.csv`
- `feature_importance.png`
- `model_metrics.csv`
- `numeric_features_distribution.png`
- `optimized_model.joblib`
- `predictions.csv`
- `target_distribution.png`

## Optimizaciones Aplicadas

- ⚡ Timeout en entrenamiento (5 min máximo)
- 🎛️ Limitación de features para evitar overfitting
- 🤖 Modelo RandomForest optimizado
- 📊 Pipeline SMOTE + StandardScaler
- 💾 Guardado automático de modelos
