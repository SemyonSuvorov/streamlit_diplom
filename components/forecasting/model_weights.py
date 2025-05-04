from typing import Dict, List
from components.forecasting.model_registry import ModelType

def get_available_weights(model_name: str) -> List[str]:
    """Get available pre-trained weights for a model"""
    weights_dict = {
        ModelType.LSTM.value: ["LSTM_energy_2023.h5", "LSTM_sales_2022.h5"],
        ModelType.PROPHET.value: ["Prophet_retail_2021.pkl"],
        ModelType.TRANSFORMER.value: ["Transformer_finance_2024.ckpt"],
        ModelType.GRU.value: [],
        ModelType.SARIMA.value: [],
        ModelType.RANDOM_FOREST.value: [],
        ModelType.XGBOOST.value: [],
        ModelType.CATBOOST.value: []
    }
    return weights_dict.get(model_name, []) 