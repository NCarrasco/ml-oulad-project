# Machine Learning – OULAD Project

Este repositorio contiene el desarrollo completo del proyecto final colaborativo de la asignatura **Ciencia de Datos I** de la Maestría en Ciencia de Datos e Inteligencia Artificial (UASD). Se aplica Machine Learning para analizar el desempeño académico de estudiantes, utilizando el dataset OULAD y un dataset complementario basado en SABER11/SABERPRO.

## 🎯 Objetivo

Construir modelos predictivos que identifiquen patrones de éxito académico y participación estudiantil, utilizando variables sociodemográficas, académicas y de interacción (clickstream).

## 📦 Estructura del Proyecto

```
ml-oulad-project/
│
├── config/           → Configuración de conexión (settings.py)
├── data/             → (Vacía por defecto) Carpeta para datasets locales (opcional)
├── src/              → Scripts Python (pipeline principal, módulos y utilidades)
│   ├── oulad_pipeline.py
│   ├── db_connector.py
│   ├── db_queries.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── models/
│   └── utils.py
├── notebooks/        → (Vacía por defecto) Jupyter/Colab para EDA y experimentos (opcional)
├── docs/             → (Vacía por defecto) Artículo científico (APA) y anexos
├── results/          → CSV, métricas, gráficos finales
├── schema.sql        → (Opcional) Script SQL si se usó RDBMS
├── requirements.txt  → Librerías necesarias
└── README.md         → Este archivo
```

## ⚙️ Instalación y configuración

1. **Clona el repositorio y accede a la carpeta:**
   ```bash
   git clone <url-del-repo>
   cd ml-oulad-project
   ```
2. **Crea un entorno virtual (opcional pero recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configura la conexión a MySQL:**
   - Edita `config/settings.py` con tus credenciales y nombre de base de datos.
   - Ejemplo de cadena de conexión:
     ```python
     SQLALCHEMY_URL = "mysql+mysqlconnector://usuario:contraseña@localhost/ouladdb"
     ```

## 🗄️ Exportar el esquema de la base de datos (opcional)

Si necesitas exportar solo el esquema (sin datos):
```bash
mysqldump -u <usuario> -p --no-data ouladdb > schema.sql
```

Si tu base de datos está en Docker, ejecuta este comando (ajusta el nombre del contenedor si es necesario):

```bash
docker exec -i <nombre_contenedor_mysql> mysqldump -u root -p --no-data ouladdb > esquema_ouladdb.sql
```

Por ejemplo, si tu contenedor se llama `mysql-oulad`:

```bash
docker exec -i mysql-sakila mysqldump -u root -p --no-data ouladdb > esquema_ouladdb.sql
```

Cuando pida la contraseña, ingresa la que corresponda (por defecto: xxxxx).

## 🚀 Ejecución del pipeline principal

1. Asegúrate de que la base de datos MySQL esté activa y accesible.
2. Ejecuta el pipeline desde la terminal (puedes limitar el tamaño para pruebas rápidas):
   ```bash
   python src/oulad_pipeline.py --max_rows 20000
   ```
   - El parámetro `--max_rows` permite trabajar con un subconjunto aleatorio del dataset para acelerar pruebas y evitar problemas de memoria en equipos personales.
   - Por defecto, el pipeline usa 100,000 filas si no se especifica.
3. Los resultados (métricas, gráficos, predicciones, clustering) se guardarán en la carpeta `results/`.

## 🧠 Tecnologías utilizadas

- Python 3.x
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- SQLAlchemy, tqdm, imbalanced-learn
- Google Colab / Visual Studio Code
- MySQL (opcional)

## 📊 Metodología

- Pipeline OSEMN: Obtener, Seleccionar, Explorar, Modelar, Comunicar
- EDA: univariado, bivariado, boxplots, correlaciones, kurtosis
- ML supervisado: Regresión logística, Random Forest, SVM
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC, MSE, R²
- Visualizaciones y análisis de variables más influyentes

## 📈 Salidas esperadas

- Predicciones (`y_test`, `y_pred`) exportadas a CSV
- Cálculo manual del F1-score (TP, FP, TN, FN)
- Visualizaciones: matriz de confusión, scatter, importancias

## 👥 Autores y colaboración

- Norman Carrasco
- Miguel Pimentel
- Miguel Consoro

Ver detalles de roles y aportes en el docstring de `src/oulad_pipeline.py`.

---

## 👥 Distribución de Tareas

Este proyecto fue desarrollado de forma colaborativa, con la siguiente división de responsabilidades:

| Etapa | Tarea                                              | Responsable         | Fecha límite |
|-------|----------------------------------------------------|---------------------|--------------|
| 1     | Definición de hipótesis y objetivos                | Todos               | 22-jun       |
| 2     | Complementar dataset OULAD con fuentes adicionales | Norman Carrasco     | 24-jun       |
| 3     | Limpieza y codificación de datos                   | Norman Carrasco     | 26-jun       |
| 4     | Análisis exploratorio (EDA)                        | Norman Carrasco     | 27-jun       |
| 5     | Entrenamiento de modelos ML                        | Norman Carrasco     | 28-jun       |
| 6     | Validación de modelos                              | Miguel Pimentel     | 29-jun       |
| 7     | Evaluación final de métricas                       | Miguel Consoro      | 30-jun       |
| 8     | Exportación de resultados                          | Norman Carrasco     | 30-jun       |
| 9     | Visualizaciones finales                            | Miguel Pimentel     | 01-jul       |
| 10    | Redacción del artículo                             | Miguel Consoro      | 02-jul       |
| 11    | Formato APA y referencias                          | Miguel Consoro      | 03-jul       |
| 12    | Documentación técnica del código                   | Norman Carrasco     | 03-jul       |
| 13    | Revisión final del entregable                      | Todos               | 04-jul       |

## 📄 Licencia

Uso exclusivamente académico. No comercial.

---

## 🛠️ Uso avanzado y troubleshooting

- **Ejecución directa:**
  - Ejecuta el pipeline completo desde terminal con:
    ```bash
    python src/oulad_pipeline.py
    ```
  - Si usas Docker para MySQL, asegúrate de que el contenedor esté corriendo y la configuración en `config/settings.py` apunte al host/puerto correcto.
- **Reproducibilidad:**
  - El pipeline es completamente reproducible usando solo `requirements.txt` y la configuración indicada.
  - Todos los resultados se generan en la carpeta `results/`.
- **Interpretación de resultados:**
  - Los archivos CSV y PNG generados pueden ser usados como anexos en artículos científicos.
  - Las métricas manuales (`metrics_manual.csv`) incluyen TP, FP, TN, FN, F1, accuracy, precision, recall, MSE, R².

## 📂 Tabla de outputs generados

| Archivo/Gráfico                  | Descripción                                      |
|----------------------------------|--------------------------------------------------|
| y_test_y_pred.csv                | Predicciones y valores reales                    |
| metrics_manual.csv               | Métricas manuales de clasificación/regresión     |
| confusion_matrix.png             | Matriz de confusión visual                       |
| feature_importances.png          | Importancia de variables (modelos de árbol)      |
| metrics_classifiers.csv          | Métricas de validación cruzada (clasificadores)  |
| metrics_regressors.csv           | Métricas de validación cruzada (regresores)      |
| kmeans_elbow.png                 | Gráfico del método del codo (clustering)         |
| random_projection_gaussian.png   | Proyección aleatoria 2D de los datos             |

## 📊 Visualizaciones generadas

A continuación se muestran ejemplos de los gráficos generados automáticamente en la carpeta `results/`:

| Boxplot por módulo | Boxplot por fecha | Boxplot por resultado final |
|--------------------|-------------------|----------------------------|
| ![boxplot_code_module](results/boxplot_code_module.png) | ![boxplot_date](results/boxplot_date.png) | ![boxplot_final_result](results/boxplot_final_result.png) |

| Histograma resultado final | Heatmap de correlación | Scatter final_result vs Withdrawn |
|---------------------------|------------------------|-----------------------------------|
| ![hist_final_result](results/hist_final_result.png) | ![correlation_heatmap](results/correlation_heatmap.png) | ![scatter_final_result_vs_FResult02_Withdrawn](results/scatter_final_result_vs_FResult02_Withdrawn.png) |

## ❓ Preguntas frecuentes (FAQ)

- **Error de conexión a MySQL:**
  - Verifica usuario, contraseña, host y puerto en `config/settings.py`.
  - Si usas Docker, expón el puerto 3306 y revisa el nombre del contenedor.
- **Faltan dependencias:**
  - Ejecuta `pip install -r requirements.txt`.
- **No se generan archivos en `results/`:**
  - Verifica permisos de escritura y que la ruta exista.
- **¿Cómo interpreto las métricas?**
  - Consulta la sección de outputs y la documentación del notebook para ejemplos y explicaciones.

> Para dudas adicionales, revisa los comentarios en el código y el notebook de ejemplo.

## 📉 Visualizaciones avanzadas y análisis de errores

- **Curva ROC:**
  - Evalúa la capacidad de discriminación del modelo en clasificación binaria.
  - Ejemplo de código:
    ```python
    from sklearn.metrics import roc_curve, auc
    y_score = modelo.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = auc(fpr, tpr)
    ```
- **Interpretabilidad con SHAP:**
  - Explica el aporte de cada variable a la predicción de modelos de árbol.
  - Ejemplo de código:
    ```python
    import shap
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    ```
- **Matriz de confusión normalizada:**
  - Permite analizar los errores relativos por clase.
  - Ejemplo de código:
    ```python
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, normalize='true')
    ```

> Estas visualizaciones pueden ser incluidas como anexos o figuras en el informe científico para enriquecer la interpretación de resultados.
