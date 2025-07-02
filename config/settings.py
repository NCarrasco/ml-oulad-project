# settings.py - Configuración de conexión a la base de datos para OULAD

import os

# Configuración de conexión a la base de datos MySQL para OULAD
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Admin.123'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'ouladdb')
}

# Cadena de conexión SQLAlchemy
SQLALCHEMY_URL = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"

