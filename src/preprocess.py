import pandas as pd
from typing import Tuple
from sklearn import preprocessing

class OULADPreprocessor:
    """Clase para limpieza y preprocesamiento de datos OULAD."""
    def __init__(self):
        pass

    def cap_outliers(self, col: pd.Series) -> pd.Series:
        """Capa los valores atípicos por encima del percentil 98."""
        if col.empty:
            return col
        return col.clip(upper=col.quantile(0.98))

    def clean(self, df: pd.DataFrame, rq: int = 1) -> Tuple[pd.DataFrame, list]:
        """Limpieza básica: missing, duplicados, outliers."""
        df = df.copy()
        nulls = df.isnull().sum()
        drop_cols = [col for col, v in nulls.items() if v == len(df)]
        df = df.drop(columns=drop_cols)
        if 'imd_band' in df.columns:
            df = df.dropna(axis=0, subset=['imd_band'])
        df = df.fillna(0)
        df = df.drop_duplicates()
        prefixes = ['n_day', 'avg_sum']
        num_vars = ['num_of_prev_attempts', 'studied_credits'] + [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
        if rq == 3 and 'score' in df.columns:
            num_vars.append('score')
        for col in num_vars:
            if col in df.columns and col != 'score':
                df[col] = self.cap_outliers(df[col])
        if len(num_vars) == 0:
            print("No hay variables numéricas para limpiar outliers.")
        return df, num_vars

    def feature_engineering(self, df: pd.DataFrame, rq: int = 1) -> pd.DataFrame:
        """Transformaciones y codificación de variables."""
        df = df.copy()
        if 'code_presentation' in df.columns:
            df['year'] = df['code_presentation'].astype(str).str.strip().str[0:4]
            df['semester'] = df['code_presentation'].astype(str).str.strip().str[-1]
        df2 = df.copy(deep=True)
        if 'final_result' in df2.columns:
            df2['FResult02'] = df2['final_result']
            df2 = pd.get_dummies(df2, columns=['FResult02'], drop_first=True)
        label_encoder = preprocessing.LabelEncoder()
        if rq == 2:
            le_cols = ['final_result', 'age_band', 'imd_band', 'disability', 'gender', 'region', 'highest_education', 'code_module', 'assessment_type', 'semester']
        else:
            le_cols = ['final_result', 'age_band', 'imd_band', 'disability', 'gender', 'region', 'highest_education', 'code_module', 'semester']
        for col in le_cols:
            if col in df2.columns:
                try:
                    df2[col] = label_encoder.fit_transform(df2[col])
                except Exception as e:
                    print(f"No se pudo codificar la columna {col}: {e}")
        if 'total_n_days' in df2.columns and 'avg_total_sum_clicks' in df2.columns:
            df2['overall_total_clicks'] = df2['total_n_days'] * df2['avg_total_sum_clicks']
        return df2
