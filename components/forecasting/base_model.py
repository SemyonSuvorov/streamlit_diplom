from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    target_col: str
    window_size: Optional[int] = None
    train_size: Optional[float] = None
    n_splits: Optional[int] = None
    forecast_approach: Optional[str] = None  
    

    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    

    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    

    alpha: Optional[float] = None  
    beta: Optional[float] = None   
    gamma: Optional[float] = None  
    

    p_values: Optional[List[int]] = None
    d_values: Optional[List[int]] = None
    q_values: Optional[List[int]] = None
    seasonal_p: Optional[List[int]] = None
    seasonal_d: Optional[List[int]] = None
    seasonal_q: Optional[List[int]] = None
    seasonal_period: Optional[int] = None
    
    # Additional parameters for model loading
    pretrained_option: Optional[str] = None
    selected_weights: Optional[str] = None
    uploaded_weights: Optional[Any] = None

class BaseModel(ABC):
    """Abstract base class for all forecasting models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, ts: pd.Series, progress_callback=None) -> Tuple[Any, Dict]:
        """Fit the model to the time series data"""
        pass
    
    def is_fitted(self) -> bool:
        """Check if the model has been fitted/trained"""
        return self._is_fitted
    
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