import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any
from components.forecasting.base_model import BaseModel, ModelConfig

class SARIMAModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.train_size = config.__dict__.get('train_size', 0.8)
        self.n_splits = config.__dict__.get('n_splits', 5)
        self.seasonal_period = config.__dict__.get('seasonal_period', 12)
        self.model = None
        self.best_model = None
        self.metrics = {}
        self.test_data = None
        self.freq = None
        self.data_mean = 0
        self.data_std = 1
        self._is_fitted = False
        self.best_params = None

    def _prepare_ts(self, ts):
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("Индекс временного ряда должен быть DatetimeIndex")
        if self.freq is None:
            self.freq = pd.infer_freq(ts.index)
            if self.freq is None:
                self.freq = 'D'
        ts = ts.asfreq(self.freq)
        self.data_mean = ts.mean()
        self.data_std = ts.std()
        ts = (ts - self.data_mean) / self.data_std
        if ts.isna().any():
            ts = ts.interpolate(method='time')
        return ts

    def _objective(self, trial, train):
        # Define the hyperparameter search space with more focused ranges
        p = trial.suggest_int('p', 0, 2)  
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 2)  
        P = trial.suggest_int('P', 0, 2)
        D = trial.suggest_int('D', 0, 2)
        Q = trial.suggest_int('Q', 0, 2)  
        
        # Create a unique key for caching
        param_key = f"p{p}_d{d}_q{q}_P{P}_D{D}_Q{Q}"
        
        # Check if we've already tried these parameters
        if hasattr(self, '_param_cache') and param_key in self._param_cache:
            return self._param_cache[param_key]
        
        try:
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Add early stopping by checking AIC during fitting
            results = model.fit(disp=False, maxiter=50)  # Limit iterations
            
            # Cache the result
            if not hasattr(self, '_param_cache'):
                self._param_cache = {}
            self._param_cache[param_key] = results.aic
            
            return results.aic
        except:
            # Cache failed attempts too
            if not hasattr(self, '_param_cache'):
                self._param_cache = {}
            self._param_cache[param_key] = float('inf')
            return float('inf')

    def fit(self, ts: pd.Series, progress_callback=None, n_trials: int = 5) -> Tuple[Any, Dict]:
        ts = self._prepare_ts(ts)
        train_size = int(len(ts) * self.train_size)
        train = ts[:train_size]
        test = ts[train_size:]
        
        try:
            # Create and optimize the study
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self._objective(trial, train), n_trials=n_trials)
            
            # Get the best parameters
            self.best_params = study.best_params
            
            # Fit the model with best parameters
            self.model = SARIMAX(
                train,
                order=(self.best_params['p'], self.best_params['d'], self.best_params['q']),
                seasonal_order=(self.best_params['P'], self.best_params['D'], 
                              self.best_params['Q'], self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.best_model = self.model.fit(disp=False)
            self._is_fitted = True
            return self.best_model
            
        except Exception as e:
            raise ValueError(f"Ошибка обучения SARIMA: {e}")

    def cross_validate(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Dict[str, list]:
        ts = df[target_col]
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mse_scores, mae_scores, r2_scores = [], [], []
        
        # Ensure we have a fitted model first
        if self.model is None or not self._is_fitted:
            raise ValueError("Model must be fitted first. Call fit() before cross_validate().")
        
        last_test_actual = None
        last_test_pred = None
        last_test_dates = None
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(ts)):
            train_ts = ts.iloc[train_idx]
            test_ts = ts.iloc[test_idx]
            
            try:
                # Подготовка данных для прогноза
                self._prepare_ts(train_ts.copy())
                
                # Прогноз используя только существующую модель (self.model)
                forecast_series, _ = self.forecast(train_ts, len(test_ts))
                
                # Handle date alignment issues - make sure indices match exactly
                aligned_forecast = forecast_series.copy()
                if not all(idx in test_ts.index for idx in aligned_forecast.index):
                    # Reindex forecast to match test data exactly
                    aligned_forecast = pd.Series(
                        forecast_series.values, 
                        index=test_ts.index
                    )
                
                # Calculate metrics
                mse = mean_squared_error(test_ts, aligned_forecast)
                mae = mean_absolute_error(test_ts, aligned_forecast)
                r2 = r2_score(test_ts, aligned_forecast)
                
                # Save metrics
                mse_scores.append(mse)
                mae_scores.append(mae)
                r2_scores.append(r2)
                
                # Save the last fold's data for plotting
                if i == self.n_splits - 1 or i == len(list(tscv.split(ts))) - 1:
                    last_test_actual = test_ts
                    last_test_pred = aligned_forecast
                    last_test_dates = test_ts.index
                
                if progress_callback:
                    progress_callback(i+1, self.n_splits)
            except Exception as e:
                print(f"Ошибка в фолде {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save the last fold's data for plotting
        if last_test_actual is not None and last_test_pred is not None:
            self.test_data = {
                'dates': last_test_dates,
                'actual': last_test_actual,
                'predicted': last_test_pred
            }
            
            # Also save the last metrics
            if mse_scores and mae_scores and r2_scores:
                self.metrics = {
                    'mse': mse_scores[-1],
                    'mae': mae_scores[-1],
                    'r2': r2_scores[-1]
                }
        
        return {'mse': mse_scores, 'mae': mae_scores, 'r2': r2_scores}

    def get_metrics(self) -> Dict[str, float]:
        if not self.metrics:
            return {"error": "No metrics available. Model may not be trained."}
        return self.metrics

    def plot_test_predictions(self) -> go.Figure:
        if not hasattr(self, 'test_data') or self.test_data is None:
            raise ValueError("Нет данных теста для построения графика. Сначала выполните fit() или cross_validate().")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.test_data['dates'],
            y=self.test_data['actual'],
            mode='lines+markers',
            name='Факт'
        ))
        fig.add_trace(go.Scatter(
            x=self.test_data['dates'],
            y=self.test_data['predicted'],
            mode='lines+markers',
            name='Прогноз',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='SARIMA: прогноз vs факт (тестовая выборка)',
            xaxis_title='Дата',
            yaxis_title='Значение',
            showlegend=True,
            hovermode='x unified'
        )
        return fig

    def forecast(self, ts: pd.Series, horizon: int) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        if self.model is None or not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        ts = self._prepare_ts(ts)
        
        # Use get_forecast instead of predict
        forecast_result = self.best_model.get_forecast(steps=horizon)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # Denormalize the forecast
        forecast_values = forecast_values * self.data_std + self.data_mean
        conf_int = conf_int * self.data_std + self.data_mean
        
        last_date = ts.index[-1]
        if self.freq is None:
            if len(ts.index) > 1:
                self.freq = pd.infer_freq(ts.index)
            if self.freq is None:
                self.freq = 'D'
        try:
            forecast_index = pd.date_range(start=last_date, periods=horizon+1, freq=self.freq)[1:]
        except Exception:
            forecast_index = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
            
        if isinstance(forecast_values, pd.Series) and isinstance(forecast_values.index, pd.DatetimeIndex):
            forecast_series = forecast_values.copy()
            forecast_series.index = forecast_index
        else:
            forecast_series = pd.Series(forecast_values, index=forecast_index)
            
        conf_intervals = pd.DataFrame({
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        }, index=forecast_index)
        
        return forecast_series, conf_intervals

    def plot_forecast(self, ts: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None) -> Any:
        if forecast is None or len(forecast) == 0:
            raise ValueError("Forecast is empty or None. Проверьте, что прогноз рассчитан корректно.")
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Прогноз временного ряда",))
        fig.add_trace(
            go.Scatter(x=ts.index, y=ts.values, name="Исторические данные")
        )
        fig.add_trace(
            go.Scatter(x=forecast.index, y=forecast.values, name="Прогноз", line=dict(color='red'))
        )
        if conf_int is not None and not conf_int.empty:
            fig.add_trace(
                go.Scatter(x=forecast.index, y=conf_int['lower'], 
                          name="Нижняя граница", line=dict(color='rgba(255,0,0,0.2)'))
            )
            fig.add_trace(
                go.Scatter(x=forecast.index, y=conf_int['upper'], 
                          name="Верхняя граница", line=dict(color='rgba(255,0,0,0.2)'),
                          fill='tonexty')
            )
        fig.update_layout(
            height=600,
            showlegend=True,
            title="Прогноз SARIMA"
        )
        fig.update_xaxes(title_text="Дата")
        fig.update_yaxes(title_text="Значение")
        return fig 