# Machine Learning – OULAD Project

Este repositorio contiene el desarrollo completo del proyecto **Machine Learning – OULAD** de la asignatura **Ciencia de Datos I** de la Maestría en Ciencia de Datos e Inteligencia Artificial (UASD). Se aplica **Machine Learning** para analizar el desempeño académico de estudiantes, utilizando el dataset OULAD y un dataset complementario basado en SABER11/SABERPRO.

---
## 🎯 Objetivo

Construir modelos predictivos que identifiquen patrones de éxito académico y participación estudiantil, utilizando variables sociodemográficas, académicas y de interacción (clickstream).

## 📦 Estructura del Proyecto

```
ml-oulad-project/
│
├── data/             → Datasets OULAD + SABER (anónimos)
├── src/              → Scripts Python estructurados (POO + TAD)
│   ├── eda.py
│   ├── preprocess.py
│   ├── models/
│   └── utils.py
├── notebooks/        → Colab para EDA, modelos y experimentos
├── docs/             → Artículo científico (APA)
├── results/          → CSV, métricas, gráficos finales
├── schema.sql        → (Opcional) Script SQL si se usó RDBMS
├── requirements.txt  → Librerías necesarias
└── README.md         → Este archivo
```

## 🧠 Tecnologías utilizadas

- Python 3.x
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Google Colab / Visual Studio Code
- SQL (opcional)

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

---
## 👥 Autores

- Norman Carrasco
- Miguel Pimentel
- Miguel Consoro

---
## 📄 Licencia

> Proyecto educativo realizado como parte del desarrollo en Ciencia de Datos & AI. Dataset público de OULAD. Todos los datos son anónimos y para uso académico.

---
## 📫 Contacto

- 📧 normcarrasco@gmail.com  
- 🔗 [LinkedIn](https://www.linkedin.com/in/nocarrasco)  
- 🌐 [Blog AprenTICs](https://apren2tics.wordpress.com)

