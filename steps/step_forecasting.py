import streamlit as st
from state.session import state
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from components.forecasting.model_selection import show_model_selection_tab
from components.forecasting.training_tab import show_training_tab
from components.forecasting.forecast_tab import show_forecast_tab

class ModelType(Enum):
    ARIMA = "ARIMA"
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"
    LSTM = "LSTM"
    GRU = "GRU"
    TRANSFORMER = "Transformer"
    PROPHET = "Prophet"
    RANDOM_FOREST = "Random Forest"

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    target_col: str
    forecast_horizon: int
    window_size: Optional[int] = None
    train_size: Optional[float] = None
    n_splits: Optional[int] = None

class BaseModel(ABC):
    """Abstract base class for all forecasting models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def fit(self, ts: pd.Series, progress_callback=None) -> Tuple[Any, Dict]:
        """Fit the model to the time series data"""
        pass
    
    @abstractmethod
    def forecast(self, ts: pd.Series, horizon: int) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Generate forecast for the given horizon"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        pass
    
    @abstractmethod
    def plot_forecast(self, ts: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None) -> Any:
        """Plot the forecast results"""
        pass

class ModelRegistry:
    """Registry for managing model implementations"""
    
    _models: Dict[ModelType, type] = {}
    _configs: Dict[ModelType, Dict] = {}
    
    @classmethod
    def register(cls, model_type: ModelType, model_class: type, default_config: Dict):
        """Register a new model implementation"""
        cls._models[model_type] = model_class
        cls._configs[model_type] = default_config
    
    @classmethod
    def get_model(cls, model_type: ModelType, config: ModelConfig) -> BaseModel:
        """Create a model instance with the given configuration"""
        if model_type not in cls._models:
            raise ValueError(f"Model type {model_type} not registered")
        return cls._models[model_type](config)
    
    @classmethod
    def get_default_config(cls, model_type: ModelType) -> Dict:
        """Get default configuration for a model type"""
        return cls._configs.get(model_type, {})

class ModelFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def create_model(model_type: ModelType, config: ModelConfig) -> BaseModel:
        """Create a model instance based on type and configuration"""
        return ModelRegistry.get_model(model_type, config)

def get_available_weights(model_name: str) -> List[str]:
    """Get available pre-trained weights for a model"""
    weights_dict = {
        ModelType.LSTM.value: ["LSTM_energy_2023.h5", "LSTM_sales_2022.h5"],
        ModelType.PROPHET.value: ["Prophet_retail_2021.pkl"],
        ModelType.TRANSFORMER.value: ["Transformer_finance_2024.ckpt"],
        ModelType.GRU.value: [],
        ModelType.ARIMA.value: [],
        ModelType.RANDOM_FOREST.value: [],
        ModelType.XGBOOST.value: [],
        ModelType.CATBOOST.value: []
    }
    return weights_dict.get(model_name, [])

def show_forecasting_step():
    """Display the forecasting step interface"""
    st.subheader("Шаг 4. Прогнозирование временного ряда")

    if state.get('filtered_df') is None or state.get('filtered_df').empty:
        st.warning("Данные не загружены!")
        return

    # Create tabs for model selection, training, and forecasting
    tab1, tab2 = st.tabs(["Обучение", "Прогноз"])
    
    with tab1:
        show_training_tab()
    with tab2:
        show_forecast_tab()

def run_step():
    """Run the forecasting step"""
    show_forecasting_step() 