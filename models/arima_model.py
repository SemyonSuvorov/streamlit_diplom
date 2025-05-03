import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable, Optional, Tuple, Dict, Any, List
from sklearn.model_selection import TimeSeriesSplit
from components.forecasting.base_model import BaseModel, ModelConfig

class ARIMAModel(BaseModel):
    def __init__(self, config: ModelConfig):
        """
        Инициализация модели ARIMA
        
        Args:
            config: Конфигурация модели
        """
        super().__init__(config)
        self.p_values = config.__dict__.get('p_values', [0, 1, 2])
        self.d_values = config.__dict__.get('d_values', [0, 1])
        self.q_values = config.__dict__.get('q_values', [0, 1, 2])
        self.train_size = config.__dict__.get('train_size', 0.8)
        self.freq = None
        self.best_model = None
        self.best_params = None
        self.best_aic = float('inf')
        self.progress_callback = None
        self.seasonal_order = (0, 0, 0, 0)
        self.n_splits = config.__dict__.get('n_splits', 5)
        self.cv_scores = []
        self.data_mean = 0
        self.data_std = 1
        self.metrics = {}
        
    def _prepare_ts(self, ts):
        """
        Подготовка временного ряда: установка частоты и обработка пропусков
        """
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("Индекс временного ряда должен быть DatetimeIndex")
            
        if self.freq is None:
            self.freq = pd.infer_freq(ts.index)
            if self.freq is None:
                self.freq = 'D'
                
        ts = ts.asfreq(self.freq)
        
        # Нормализация данных
        self.data_mean = ts.mean()
        self.data_std = ts.std()
        ts = (ts - self.data_mean) / self.data_std
        
        # Интерполяция пропусков
        if ts.isna().any():
            ts = ts.interpolate(method='time')
            
        return ts
        
    def _detect_seasonality(self, ts):
        """
        Определение сезонности
        """
        try:
            if self.freq == 'M':
                period = 12
            elif self.freq == 'D':
                period = 7
            elif self.freq == 'Q':
                period = 4
            else:
                period = 12
                
            decomposition = seasonal_decompose(ts.dropna(), period=period)
            seasonal_std = decomposition.seasonal.std()
            return seasonal_std > 0.1
        except:
            return False
            
    def _cross_validate(self, ts, p, d, q, seasonal_order):
        """
        Кросс-валидация для оценки качества модели
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(ts):
            train = ts.iloc[train_idx]
            test = ts.iloc[test_idx]
            
            try:
                model = ARIMA(train, 
                            order=(p, d, q),
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                model_fit = model.fit()
                
                forecast = model_fit.forecast(steps=len(test))
                score = mean_squared_error(test, forecast)
                scores.append(score)
            except:
                continue
                
        return np.mean(scores) if scores else float('inf')
        
    def _fit_model(self, train, p, d, q, seasonal_order):
        """
        Обучение модели с несколькими стратегиями оптимизации
        
        Args:
            train: Обучающие данные
            p: Параметр авторегрессии
            d: Параметр дифференцирования
            q: Параметр скользящего среднего
            seasonal_order: Сезонный порядок
            
        Returns:
            model_fit: Обученная модель или None в случае неудачи
        """
        # Список методов оптимизации для перебора
        optimization_methods = [
            {
                'method': 'statespace',
                'kwargs': {'maxiter': 2000, 'disp': 0},
                'cov_type': 'robust'
            },
            {
                'method': 'css-mle',
                'kwargs': {'maxiter': 2000, 'disp': 0},
                'cov_type': 'robust'
            },
            {
                'method': 'css',
                'kwargs': {'maxiter': 2000, 'disp': 0},
                'cov_type': 'robust'
            }
        ]
        
        best_model = None
        best_aic = float('inf')
        last_error = None
        
        for opt_method in optimization_methods:
            try:
                model = ARIMA(train, 
                            order=(p, d, q),
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                
                model_fit = model.fit(
                    method=opt_method['method'],
                    method_kwargs=opt_method['kwargs'],
                    cov_type=opt_method['cov_type']
                )
                
                # Проверяем сходимость
                if model_fit.mle_retvals['converged']:
                    current_aic = model_fit.aic
                    if current_aic < best_aic:
                        best_model = model_fit
                        best_aic = current_aic
                
            except Exception as e:
                last_error = str(e)
                continue
        
        if best_model is None:
            print(f"Все методы оптимизации не смогли обучить модель. Последняя ошибка: {last_error}")
            return None
            
        return best_model

    def fit(self, ts: pd.Series, progress_callback=None) -> Tuple[Any, Dict]:
        """
        Обучение модели с кросс-валидацией
        """
        self.progress_callback = progress_callback
        ts = self._prepare_ts(ts)
        train_size = int(len(ts) * self.train_size)
        train = ts[:train_size]
        
        has_seasonality = self._detect_seasonality(train)
        
        if has_seasonality:
            if self.freq == 'M':
                period = 12
            elif self.freq == 'D':
                period = 7
            elif self.freq == 'Q':
                period = 4
            else:
                period = 12
            self.seasonal_order = (1, 1, 1, period)
            
        total_combinations = len(self.p_values) * len(self.d_values) * len(self.q_values)
        current_combination = 0
        
        for p in self.p_values:
            for d in self.d_values:
                for q in self.q_values:
                    current_combination += 1
                    
                    if self.progress_callback:
                        self.progress_callback(current_combination, total_combinations)
                    
                    try:
                        cv_score = self._cross_validate(train, p, d, q, self.seasonal_order)
                        
                        if cv_score < self.best_aic:
                            model_fit = self._fit_model(train, p, d, q, self.seasonal_order)
                            
                            if model_fit is not None and model_fit.mle_retvals['converged']:
                                self.best_aic = cv_score
                                self.best_model = model_fit
                                self.best_params = (p, d, q)
                                self.cv_scores.append(cv_score)
                            
                    except Exception as e:
                        print(f"Ошибка при подборе параметров: {str(e)}")
                        continue
        
        if self.best_model is None:
            raise ValueError("Не удалось построить модель ARIMA. Попробуйте другие параметры или предобработайте данные.")
            
        # Calculate metrics on test set
        test = ts[train_size:]
        forecast = self.best_model.forecast(steps=len(test))
        self.metrics = {
            'mse': mean_squared_error(test, forecast),
            'mae': mean_absolute_error(test, forecast),
            'r2': r2_score(test, forecast)
        }
        
        # Mark the model as fitted
        self._is_fitted = True
            
        return self.best_model, {'params': self.best_params, 'aic': self.best_aic}
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        if not self.metrics:
            return {"error": "No metrics available. Model may not be trained."}
        return self.metrics
    
    def cross_validate(self, df: pd.DataFrame, target_col: str) -> Dict[str, List[float]]:
        """Perform cross validation"""
        ts = df[target_col]
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        # We'll use the same validation approach as in fit
        self.fit(ts, self.progress_callback)
        
        # Model has been fitted in the fit method, but let's explicitly set it here too
        if self.best_model is not None:
            self._is_fitted = True
        else:
            self._is_fitted = False
        
        # Return metrics from fit
        return {
            'mse': [self.metrics['mse']],
            'mae': [self.metrics['mae']],
            'r2': [self.metrics['r2']]
        }
    
    def plot_test_predictions(self, ts=None):
        """
        Визуализация предсказаний на тестовых данных
        
        Args:
            ts: Исходный временной ряд
            
        Returns:
            plotly Figure
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        if ts is None:
            # If no data is provided, use a dummy plot
            fig = go.Figure()
            fig.add_annotation(
                text="No test data provided",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        ts = (ts - self.data_mean) / self.data_std
        train_size = int(len(ts) * self.train_size)
        train = ts[:train_size]
        test = ts[train_size:]
        
        # Получаем предсказания для тестовых данных
        test_forecast = self.best_model.forecast(steps=len(test))
        test_forecast = test_forecast * self.data_std + self.data_mean
        test = test * self.data_std + self.data_mean
        
        # Вычисляем метрики
        metrics = {
            'mse': mean_squared_error(test, test_forecast),
            'mae': mean_absolute_error(test, test_forecast),
            'r2': r2_score(test, test_forecast)
        }
        
        # Создаем график
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=("Предсказания на тестовых данных", "Ошибки предсказаний"),
                          vertical_spacing=0.2)
        
        # Добавляем исторические данные
        fig.add_trace(
            go.Scatter(x=ts.index, y=ts.values * self.data_std + self.data_mean, 
                      name="Исходные данные", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Добавляем предсказания
        fig.add_trace(
            go.Scatter(x=test.index, y=test_forecast, 
                      name="Предсказания", line=dict(color='red')),
            row=1, col=1
        )
        
        # Добавляем ошибки
        errors = test - test_forecast
        fig.add_trace(
            go.Scatter(x=test.index, y=errors, name="Ошибки"),
            row=2, col=1
        )
        
        # Добавляем метрики в заголовок
        metrics_text = f"MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.2f}"
        fig.update_layout(
            title_text=f"Результаты на тестовых данных<br>{metrics_text}",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Дата", row=1, col=1)
        fig.update_xaxes(title_text="Дата", row=2, col=1)
        fig.update_yaxes(title_text="Значение", row=1, col=1)
        fig.update_yaxes(title_text="Ошибка", row=2, col=1)
        
        return fig
    
    def forecast(self, ts: pd.Series, horizon: int) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Прогнозирование
        
        Args:
            ts: Временной ряд
            horizon: Горизонт прогнозирования
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: прогноз и доверительные интервалы
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first. Use the 'Обучение' tab to properly train the model.")
        
        if not self._is_fitted:
            raise ValueError("Model is not properly fitted. The is_fitted flag is False. Please retrain the model in the 'Обучение' tab.")
            
        # Предобработка временного ряда
        ts = self._prepare_ts(ts)
        
        # Прогнозирование
        forecast_values = self.best_model.forecast(steps=horizon)
        
        # Денормализация результатов
        forecast_values = forecast_values * self.data_std + self.data_mean
        
        # Создание временного индекса для прогноза
        last_date = ts.index[-1]
        
        # Determine frequency from the input time series if not set
        if self.freq is None:
            if len(ts.index) > 1:
                self.freq = pd.infer_freq(ts.index)
            if self.freq is None:
                self.freq = 'D'  # Default to daily frequency
        
        # Create forecast index
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq=self.freq
        )
        
        # Создание Series с прогнозом
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        # Получение доверительных интервалов
        conf_int = self.best_model.get_forecast(steps=horizon).conf_int()
        conf_int = conf_int * self.data_std + self.data_mean
        
        # Преобразование в формат, ожидаемый интерфейсом
        conf_intervals = pd.DataFrame({
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        }, index=forecast_index)
        
        return forecast_series, conf_intervals
    
    def plot_forecast(self, ts: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None) -> Any:
        """
        Визуализация прогноза
        
        Args:
            ts: Исходный временной ряд
            forecast: Прогноз
            conf_int: Доверительные интервалы
            
        Returns:
            plotly Figure
        """
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Прогноз временного ряда",))
        
        # Добавляем исторические данные
        fig.add_trace(
            go.Scatter(x=ts.index, y=ts.values, name="Исторические данные", line=dict(color='blue'))
        )
        
        # Добавляем прогноз
        fig.add_trace(
            go.Scatter(x=forecast.index, y=forecast.values, name="Прогноз", line=dict(color='red'))
        )
        
        # Добавляем доверительные интервалы
        if conf_int is not None:
            fig.add_trace(
                go.Scatter(x=forecast.index, y=conf_int['lower'], 
                          name="Нижняя граница", line=dict(color='rgba(255,0,0,0.2)'))
            )
            fig.add_trace(
                go.Scatter(x=forecast.index, y=conf_int['upper'], 
                          name="Верхняя граница", line=dict(color='rgba(255,0,0,0.2)'),
                          fill='tonexty')
            )
        
        # Обновляем макет
        fig.update_layout(
            height=600,
            showlegend=True,
            title="Прогноз ARIMA"
        )
        
        fig.update_xaxes(title_text="Дата")
        fig.update_yaxes(title_text="Значение")
        
        return fig 