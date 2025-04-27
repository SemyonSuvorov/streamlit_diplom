"""
Конфигурационный файл приложения
"""

# Настройки приложения
APP_CONFIG = {
    "title": "Time-series analysis",
    "icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Настройки данных
DATA_CONFIG = {
    "allowed_file_types": ["csv", "xlsx"],
    "default_date_format": "%Y-%m-%d",
    "max_preview_rows": 1000
}

# Настройки визуализации
VISUALIZATION_CONFIG = {
    "chart_height": 400,
    "chart_margin": dict(l=20, r=20, t=40, b=20),
    "color_sequence": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
}

# Настройки предобработки
PREPROCESSING_CONFIG = {
    "default_rolling_window": 7,
    "default_seasonal_period": 12,
    "default_outlier_threshold": 3.0
}

# Настройки трансформации
TRANSFORMATION_CONFIG = {
    "default_scaler": "StandardScaler",
    "default_outlier_method": "IQR",
    "default_replacement_method": "median"
} 