from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.arima_model import ARIMAModel
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
    
    # Register ARIMA model
    ModelRegistry.register(
        ModelType.ARIMA,
        ARIMAModel,
        {
            'p_values': [0, 1, 2],
            'd_values': [0, 1],
            'q_values': [0, 1, 2],
            'train_size': 0.8,
            'n_splits': 5
        }
    ) 