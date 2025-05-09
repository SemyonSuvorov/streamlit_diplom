import pandas as pd
import pmdarima as pm
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
        self.best_model = None
        self.metrics = {}
        self.test_data = None
        self.freq = None
        self.data_mean = 0
        self.data_std = 1
        self._is_fitted = False

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

    def fit(self, ts: pd.Series, progress_callback=None, n_trials: int = 15) -> Tuple[Any, Dict]:
        ts = self._prepare_ts(ts)
        train_size = int(len(ts) * self.train_size)
        train = ts[:train_size]
        test = ts[train_size:]
        try:
            m = self.seasonal_period if self.seasonal_period else 1
            model = pm.auto_arima(
                train,
                start_p=1, start_q=1,
                test='adf',
                max_p=3, max_q=3,
                m=m,    
                start_P=0, seasonal=True,
                d=None, D=1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            self.best_model = model
            self._is_fitted = True
            forecast = model.predict(n_periods=len(test))
            forecast = forecast * self.data_std + self.data_mean
            test_orig = test * self.data_std + self.data_mean
            self.metrics = {
                'mse': mean_squared_error(test_orig, forecast),
                'mae': mean_absolute_error(test_orig, forecast),
                'r2': r2_score(test_orig, forecast)
            }
            self.test_data = {
                'dates': test.index,
                'actual': test_orig,
                'predicted': forecast
            }
            return self.best_model, {'aic': getattr(self.best_model, 'aic', None)}
        except Exception as e:
            raise ValueError(f"Ошибка обучения auto_arima: {e}")

    def cross_validate(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Dict[str, list]:
        ts = df[target_col]
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        ts = self._prepare_ts(ts)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mse_scores, mae_scores, r2_scores = [], [], []
        train_size = int(len(ts) * self.train_size)
        train = ts[:train_size]
        m = self.seasonal_period if self.seasonal_period else 1
        model = self.best_model
        for i, (train_idx, test_idx) in enumerate(tscv.split(ts)):
            test = ts.iloc[test_idx]
            try:
                forecast = model.predict(n_periods=len(test))
                forecast = forecast * self.data_std + self.data_mean
                test_orig = test * self.data_std + self.data_mean
                mse_scores.append(mean_squared_error(test_orig, forecast))
                mae_scores.append(mean_absolute_error(test_orig, forecast))
                r2_scores.append(r2_score(test_orig, forecast))
                self.test_data = {
                    'dates': test.index,
                    'actual': test_orig,
                    'predicted': forecast
                }
                self.metrics = {
                    'mse': mse_scores[-1],
                    'mae': mae_scores[-1],
                    'r2': r2_scores[-1]
                }
                if progress_callback:
                    progress_callback(i+1, self.n_splits)
            except Exception as e:
                continue
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
        if self.best_model is None or not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        ts = self._prepare_ts(ts)
        forecast_values, conf_int = self.best_model.predict(n_periods=horizon, return_conf_int=True)
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
            'lower': conf_int[:, 0],
            'upper': conf_int[:, 1]
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