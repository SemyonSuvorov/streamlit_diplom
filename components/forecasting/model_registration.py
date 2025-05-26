from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMModel
from models.dmen_model import DMENModel
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

    # Register LSTM model
    ModelRegistry.register(
        ModelType.LSTM,
        LSTMModel,
        {
            'window_size': 14,
            'train_size': 0.8,
            'n_splits': 5,
            'epochs': 20,
            'batch_size': 32,
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001
        }
    )

    # Register DMEN model
    ModelRegistry.register(
        ModelType.DMEN,
        DMENModel,
        {
            'window_size': 30,
            'train_size': 0.8,
            'n_splits': 5,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.01,
            'epochs': 50,
            'batch_size': 32,
            'alpha': 1.0,
            'beta': 0.5,
            'gamma': 0.5
        }
    ) 