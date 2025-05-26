from typing import Dict, Any
from enum import Enum
from components.forecasting.base_model import BaseModel, ModelConfig

class ModelType(Enum):
    SARIMA = "SARIMA"
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"
    LSTM = "LSTM"
    DMEN = "DMEN"

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