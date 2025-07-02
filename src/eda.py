# Exploratory Data Analysis

import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

class OULADEDA:
    """Clase para análisis exploratorio de datos (EDA)."""
    def __init__(self):
        self.plt = plt
        self.sns = sns

    def univariate_analysis(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Análisis univariado: describe, histogramas, kurtosis."""
        desc = df.describe(include='all')
        print("\nResumen estadístico:\n", desc)
        if output_dir:
            desc.to_csv(f"{output_dir}/univariate_describe.csv")
        # Histogramas y kurtosis
        for col in df.select_dtypes(include=[np.number]).columns:
            self.plt.figure()
            df[col].hist(bins=30)
            self.plt.title(f"Histograma de {col}")
            self.plt.xlabel(col)
            self.plt.ylabel("Frecuencia")
            if output_dir:
                self.plt.savefig(f"{output_dir}/hist_{col}.png")
            self.plt.close()
            print(f"Kurtosis de {col}: {df[col].kurtosis():.2f}")

    def bivariate_analysis(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Análisis bivariado: matriz de correlación y scatter plot de las principales variables."""
        corr = df.corr(numeric_only=True)
        print("\nMatriz de correlación:\n", corr)
        if output_dir:
            corr.to_csv(f"{output_dir}/correlation_matrix.csv")
        # Heatmap de correlación
        self.plt.figure(figsize=(10,8))
        self.sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        self.plt.title("Matriz de correlación")
        if output_dir:
            self.plt.savefig(f"{output_dir}/correlation_heatmap.png")
        self.plt.close()
        # Scatter plot de las dos variables más correlacionadas
        if len(corr.columns) >= 2:
            top_corr = corr.abs().unstack().sort_values(ascending=False)
            top_corr = top_corr[top_corr < 1].index[0]
            x, y = top_corr
            self.plt.figure()
            self.sns.scatterplot(x=df[x], y=df[y])
            self.plt.title(f"Scatter plot: {x} vs {y}")
            if output_dir:
                self.plt.savefig(f"{output_dir}/scatter_{x}_vs_{y}.png")
            self.plt.close()

    def plot_boxplots(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Boxplots para variables numéricas."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.plt.figure()
            self.sns.boxplot(y=df[col])
            self.plt.title(f"Boxplot de {col}")
            if output_dir:
                self.plt.savefig(f"{output_dir}/boxplot_{col}.png")
            self.plt.close()
