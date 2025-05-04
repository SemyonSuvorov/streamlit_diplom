from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.sarima_model import SARIMAModel
from components.forecasting.model_registry import ModelRegistry, ModelType

def register_models():
    """Register all available models"""
    # Register XGBoost model
    ModelRegistry.register(
        ModelType.XGBOOST,
        XGBoostModel,
        {
            'window_size': 14,
            'train_size': 0.8,
            'n_splits': 5,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    )

    # Register CatBoost model
    ModelRegistry.register(
        ModelType.CATBOOST,
        CatBoostModel,
        {
            'window_size': 14,
            'train_size': 0.8,
            'n_splits': 5,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    )

    # Register SARIMA model
    ModelRegistry.register(
        ModelType.SARIMA,
        SARIMAModel,
        {
            'p_values': [0, 1, 2],
            'd_values': [0, 1],
            'q_values': [0, 1, 2],
            'seasonal_p': [0, 1],
            'seasonal_d': [0, 1],
            'seasonal_q': [0, 1],
            'seasonal_period': 0,
            'train_size': 0.8,
            'n_splits': 5
        }
    ) 