"""
db_queries.py
Funciones para consultas y lecturas SQL usando OULADDBConnector
"""
import pandas as pd
from db_connector import OULADDBConnector

# Ejemplo: Leer una tabla completa a DataFrame
def read_table(table_name: str, connector: OULADDBConnector) -> pd.DataFrame:
    engine = connector.get_engine()
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(f"Leídos {len(df)} registros de la tabla {table_name}.")
    return df

# Ejemplo: Ejecutar una consulta SQL personalizada
def run_query(query: str, connector: OULADDBConnector) -> pd.DataFrame:
    engine = connector.get_engine()
    df = pd.read_sql(query, engine)
    print(f"Consulta ejecutada. Registros obtenidos: {len(df)}")
    return df

# Ejemplo: Obtener solo los nombres de las tablas
def list_tables(connector: OULADDBConnector) -> list:
    from sqlalchemy import inspect
    engine = connector.get_engine()
    insp = inspect(engine)
    tables = insp.get_table_names()
    print(f"Tablas en la base de datos: {tables}")
    return tables
