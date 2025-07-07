# settings.py - Configuración mejorada para OULAD

import os

# Configuración de conexión a la base de datos MySQL para OULAD
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Admin.123'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'ouladdb'),
    'port': os.getenv('DB_PORT', 3306)
}

# Cadenas de conexión SQLAlchemy (múltiples opciones)
# Opción 1: PyMySQL (recomendado)
SQLALCHEMY_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"

# Opción 2: MySQL Connector (backup)
#SQLALCHEMY_URL_BACKUP = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Parámetros adicionales para conexión segura
CONNECTION_PARAMS = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'connect_args': {
        'charset': 'utf8mb4',
        'auth_plugin': 'mysql_native_password'  # Evita problemas de autenticación
    }
}