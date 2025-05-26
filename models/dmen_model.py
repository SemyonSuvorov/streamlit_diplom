import numpy as np
import pandas as pd
from scipy.special import expit  # сигмоида
from collections import deque
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, List, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Импорты для базовых моделей
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from catboost import CatBoostRegressor
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from components.forecasting.base_model import BaseModel, ModelConfig
from state.session import state

warnings.filterwarnings('ignore')

class DMENModel(BaseModel):
    """
    Dynamic Mutual Enhancement Network (DMEN) - комбинированная модель
    с динамическими связями между базовыми моделями
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Параметры DMEN
        self.window_size = getattr(config, 'window_size', 30)
        self.learning_rate = getattr(config, 'learning_rate', 0.01)
        self.alpha = getattr(config, 'alpha', 1.0)  # вес производительности
        self.beta = getattr(config, 'beta', 0.5)    # вес вариабельности
        self.gamma = getattr(config, 'gamma', 0.5)  # вес консистентности
        
        # Параметры базовых моделей
        self.n_estimators = getattr(config, 'n_estimators', 100)
        self.max_depth = getattr(config, 'max_depth', 6)
        self.epochs = getattr(config, 'epochs', 50)
        self.batch_size = getattr(config, 'batch_size', 32)
        self.train_size = getattr(config, 'train_size', 0.8)
        self.n_splits = getattr(config, 'n_splits', 5)
        
        # Инициализация матрицы взаимного усиления
        self.theta = np.zeros((4, 4))
        np.fill_diagonal(self.theta, 0)  # диагональ всегда 0
        
        # История для расчета динамических весов
        self.performance_history = {
            'sarima': deque(maxlen=self.window_size),
            'xgboost': deque(maxlen=self.window_size),
            'catboost': deque(maxlen=self.window_size),
            'lstm': deque(maxlen=self.window_size)
        }
        
        self.residuals_history = {
            'sarima': deque(maxlen=self.window_size),
            'xgboost': deque(maxlen=self.window_size),
            'catboost': deque(maxlen=self.window_size),
            'lstm': deque(maxlen=self.window_size)
        }
        
        self.y_true_history = deque(maxlen=self.window_size)
        self.y_pred_history = {
            'sarima': deque(maxlen=self.window_size),
            'xgboost': deque(maxlen=self.window_size),
            'catboost': deque(maxlen=self.window_size),
            'lstm': deque(maxlen=self.window_size)
        }
        
        self.models = {}
        self.model_names = ['sarima', 'xgboost', 'catboost', 'lstm']
        
        # Инициализируем приоритетными весами (регрессоры получают больший вес)
        model_priorities = {
            'sarima': 0.8,      # Меньший приоритет для SARIMA
            'xgboost': 1.3,     # Больший приоритет для регрессоров
            'catboost': 1.3,    # Больший приоритет для регрессоров
            'lstm': 1.0         # Нейтральный приоритет для LSTM
        }
        total_priority = sum(model_priorities.values())
        self.dynamic_weights = {name: model_priorities[name]/total_priority for name in self.model_names}
        
        self.feature_names = []
        self.cv_metrics = None
        self.test_data = None
        
    def _calculate_reliability_score(self, model_name: str) -> float:
        """Расчет оценки надежности модели"""
        if len(self.performance_history[model_name]) < 5:
            return 0.0  # дефолтное значение
        
        errors = list(self.performance_history[model_name])
        
        # Производительность (1 - MAPE)
        if len(self.y_true_history) > 0:
            y_true_values = list(self.y_true_history)
            mean_true = np.mean(np.abs(y_true_values))
            if mean_true > 1e-8:
                mape = np.mean(np.abs(errors)) / mean_true
                performance = 1 - min(mape, 1)
            else:
                performance = 0.5
        else:
            performance = 0.5
        
        # Вариабельность ошибок
        mean_abs_errors = np.mean(np.abs(errors))
        if mean_abs_errors > 1e-8:
            variance = np.std(errors) / mean_abs_errors
        else:
            variance = 0.0
        
        # Консистентность (корреляция с истинными значениями)
        if len(self.y_pred_history[model_name]) >= 5 and len(self.y_true_history) >= 5:
            y_true = list(self.y_true_history)[-len(self.y_pred_history[model_name]):]
            y_pred = list(self.y_pred_history[model_name])
            
            if len(y_true) == len(y_pred) and len(y_true) > 1:
                try:
                    corr_matrix = np.corrcoef(y_true, y_pred)
                    if corr_matrix.shape == (2, 2):
                        consistency = corr_matrix[0, 1]
                        consistency = max(0, consistency) if not np.isnan(consistency) else 0.5
                    else:
                        consistency = 0.5
                except:
                    consistency = 0.5
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        
        # Комбинированная оценка
        z = (self.alpha * performance - 
             self.beta * variance + 
             self.gamma * consistency)
        
        # Проверяем валидность результата
        if np.isnan(z) or np.isinf(z):
            z = 0.0
        
        return z
    
    def _calculate_dynamic_weights(self) -> Dict[str, float]:
        """Расчет динамических весов моделей с приоритетом для регрессоров"""
        weights = {}
        
        # Коэффициенты приоритета для разных типов моделей
        model_priorities = {
            'sarima': 0.8,      # Меньший приоритет для SARIMA
            'xgboost': 1.3,     # Больший приоритет для регрессоров
            'catboost': 1.3,    # Больший приоритет для регрессоров
            'lstm': 1.0         # Нейтральный приоритет для LSTM
        }
        
        for model_name in self.model_names:
            z = self._calculate_reliability_score(model_name)
            # Проверяем, что z является числом
            if np.isnan(z) or np.isinf(z):
                z = 0.0
            
            # Применяем приоритет модели
            priority = model_priorities.get(model_name, 1.0)
            adjusted_z = z * priority
            
            weights[model_name] = expit(adjusted_z)  # сигмоида для [0, 1]
        
        # Нормализация весов
        total = sum(weights.values())
        if total > 0 and not np.isnan(total) and not np.isinf(total):
            weights = {k: v/total for k, v in weights.items()}
        else:
            # Если нет валидных весов, используем приоритетные веса
            total_priority = sum(model_priorities.values())
            weights = {k: model_priorities[k]/total_priority for k in self.model_names}
        
        # Дополнительная проверка на валидность
        for k, v in weights.items():
            if np.isnan(v) or np.isinf(v) or v is None:
                # Используем приоритетный вес как fallback
                priority = model_priorities.get(k, 1.0)
                total_priority = sum(model_priorities.values())
                weights[k] = priority / total_priority
        
        return weights
    
    def _update_theta_matrix(self, errors: Dict[str, float]):
        """Обновление матрицы взаимного усиления"""
        model_idx = {name: i for i, name in enumerate(self.model_names)}
        
        for i, model_i in enumerate(self.model_names):
            if model_i not in errors:
                continue
                
            for j, model_j in enumerate(self.model_names):
                if i == j or model_j not in self.residuals_history:
                    continue
                
                if len(self.residuals_history[model_j]) > 0:
                    # Градиент по theta_ij
                    e_i = errors[model_i]
                    r_j = list(self.residuals_history[model_j])[-1]
                    phi_i = self.dynamic_weights.get(model_i, 0.25)
                    
                    gradient = -2 * e_i * r_j * phi_i
                    
                    # Обновление с ограничением
                    self.theta[i, j] -= self.learning_rate * gradient
                    self.theta[i, j] = np.clip(self.theta[i, j], -1, 1)
    
    def _enhance_features(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Усиление признаков остатками других моделей"""
        if model_name not in self.model_names:
            return X
            
        model_idx = self.model_names.index(model_name)
        enhanced_X = X.copy()
        
        # Добавляем взвешенные остатки других моделей
        enhancement = np.zeros(X.shape[0])
        
        for j, other_model in enumerate(self.model_names):
            if j == model_idx or other_model not in self.residuals_history:
                continue
            
            if len(self.residuals_history[other_model]) > 0:
                recent_residuals = list(self.residuals_history[other_model])
                # Берем последние остатки, соответствующие размеру X
                residuals_to_use = recent_residuals[-X.shape[0]:]
                
                if len(residuals_to_use) == X.shape[0]:
                    enhancement += self.theta[model_idx, j] * np.array(residuals_to_use)
                elif len(residuals_to_use) > 0:
                    # Если размеры не совпадают, используем последнее значение
                    enhancement += self.theta[model_idx, j] * residuals_to_use[-1]
        
        # Добавляем усиление как новый признак
        if enhancement.any():
            enhanced_X = np.column_stack([enhanced_X, enhancement])
        
        return enhanced_X
    
    def _create_basic_features(self, y: pd.Series) -> np.ndarray:
        """Создание базовых признаков из временного ряда"""
        n = len(y)
        features = []
        
        # Временной индекс
        features.append(np.arange(n).reshape(-1, 1))
        
        # Лаги
        for lag in [1, 7, 14]:
            if lag < n:
                lagged = np.concatenate([np.full(lag, y.iloc[0]), y.iloc[:-lag]])
                features.append(lagged.reshape(-1, 1))
        
        # Скользящие средние
        for window in [7, 14]:
            if window < n:
                ma = y.rolling(window, min_periods=1).mean().values
                features.append(ma.reshape(-1, 1))
        
        # Временные признаки, если индекс - datetime
        if isinstance(y.index, pd.DatetimeIndex):
            features.append((y.index.dayofweek).values.reshape(-1, 1))
            features.append((y.index.month).values.reshape(-1, 1))
            features.append((y.index.quarter).values.reshape(-1, 1))
        else:
            # День недели и месяц (синтетические)
            features.append((np.arange(n) % 7).reshape(-1, 1))
            features.append((np.arange(n) // 30 % 12).reshape(-1, 1))
        
        return np.hstack(features)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей для LSTM"""
        sequences = []
        targets = []
        
        for i in range(len(X) - seq_length):
            sequences.append(X[i:i+seq_length])
            targets.append(y[i+seq_length])
        
        return np.array(sequences), np.array(targets)
    
    def _get_categorical_features(self, X: np.ndarray) -> List[int]:
        """Определение категориальных признаков для CatBoost"""
        return []
    
    def fit(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Tuple[Any, Dict]:
        """Обучение DMEN"""
        if target_col not in df.columns:
            raise ValueError(f"Целевая переменная '{target_col}' не найдена в данных")
        
        # Подготовка данных
        y = df[target_col]
        
        # Создаем признаки из всех доступных колонок или базовые признаки
        feature_cols = [col for col in df.columns if col != target_col]
        if feature_cols:
            X = df[feature_cols].values
            self.feature_names = feature_cols
        else:
            X = self._create_basic_features(y)
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Разделение на обучение и валидацию
        val_size = min(30, len(y) // 5)
        train_size = len(y) - val_size
        
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        X_train, X_val = X[:train_size], X[train_size:]
        
        total_steps = 4  # количество моделей
        current_step = 0
        
        # 1. Обучение SARIMA
        if progress_callback:
            progress_callback(current_step, total_steps)
        
        try:
            # Пробуем разные конфигурации SARIMA
            sarima_configs = [
                ((2, 1, 2), (1, 0, 1, 7)),
                ((1, 1, 1), (0, 0, 0, 0)),
                ((2, 1, 1), (0, 0, 0, 0)),
                ((1, 1, 2), (0, 0, 0, 0))
            ]
            
            best_aic = float('inf')
            best_sarima = None
            
            for order, seasonal_order in sarima_configs:
                try:
                    if seasonal_order == (0, 0, 0, 0):
                        model = ARIMA(y_train, order=order).fit()
                    else:
                        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
                    
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_sarima = model
                except:
                    continue
            
            if best_sarima is not None:
                self.models['sarima'] = best_sarima
            else:
                # Fallback к простой ARIMA
                self.models['sarima'] = ARIMA(y_train, order=(1, 1, 1)).fit()
                
        except Exception as e:
            print(f"Ошибка обучения SARIMA: {e}")
            # Создаем заглушку
            self.models['sarima'] = None
        
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        
        # 2. Обучение XGBoost
        try:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=0
            )
            self.models['xgboost'].fit(X_train, y_train)
        except Exception as e:
            print(f"Ошибка обучения XGBoost: {e}")
            self.models['xgboost'] = None
        
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        
        # 3. Обучение CatBoost
        try:
            cat_features = self._get_categorical_features(X_train)
            self.models['catboost'] = CatBoostRegressor(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_seed=42,
                verbose=False
            )
            # Передаем cat_features только если есть категориальные признаки
            if cat_features:
                self.models['catboost'].fit(X_train, y_train, cat_features=cat_features)
            else:
                self.models['catboost'].fit(X_train, y_train)
        except Exception as e:
            print(f"Ошибка обучения CatBoost: {e}")
            self.models['catboost'] = None
        
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        
        # 4. Обучение LSTM
        if TENSORFLOW_AVAILABLE:
            try:
                seq_length = min(7, len(y_train) // 4)
                if len(y_train) > seq_length * 2:
                    X_seq, y_seq = self._create_sequences(X_train, y_train.values, seq_length)
                    
                    model = Sequential([
                        LSTM(50, activation='tanh', input_shape=(seq_length, X_seq.shape[2])),
                        Dropout(0.2),
                        Dense(25, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                    model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
                    self.models['lstm'] = model
                    self.lstm_seq_length = seq_length
                else:
                    self.models['lstm'] = None
            except Exception as e:
                print(f"Ошибка обучения LSTM: {e}")
                self.models['lstm'] = None
        else:
            print("TensorFlow недоступен, LSTM модель пропущена")
            self.models['lstm'] = None
        
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        
        # Инициализация историй на валидационных данных
        self._initialize_histories_with_validation(X_val, y_val)
        
        # Настройка матрицы theta через несколько итераций
        for epoch in range(10):
            self._validation_epoch(X_val, y_val)
        
        # Обновляем динамические веса
        self.dynamic_weights = self._calculate_dynamic_weights()
        
        self._is_fitted = True
        
        # Сохраняем данные для состояния
        if state.get('feature_df') is None:
            state.set('feature_df', df.copy())
        
        return self, {}
    
    def _initialize_histories_with_validation(self, X_val: np.ndarray, y_val: pd.Series):
        """Инициализация историй на валидационных данных"""
        self.recent_history = deque(maxlen=max(30, self.window_size))
        
        # Заполняем начальными значениями
        for i in range(len(y_val)):
            self.y_true_history.append(y_val.iloc[i])
            
            for model_name in self.model_names:
                if model_name in self.models and self.models[model_name] is not None:
                    try:
                        if model_name == 'sarima':
                            pred = self.models['sarima'].forecast(steps=1)[0]
                        elif model_name in ['xgboost', 'catboost']:
                            pred = self.models[model_name].predict(X_val[i:i+1])[0]
                        elif model_name == 'lstm' and hasattr(self, 'lstm_seq_length'):
                            if i >= self.lstm_seq_length:
                                X_seq = X_val[i-self.lstm_seq_length:i].reshape(1, self.lstm_seq_length, -1)
                                pred = self.models['lstm'].predict(X_seq, verbose=0)[0, 0]
                            else:
                                pred = y_val.iloc[i]  # fallback
                        else:
                            pred = y_val.iloc[i]  # fallback
                        
                        self.y_pred_history[model_name].append(pred)
                        error = y_val.iloc[i] - pred
                        self.performance_history[model_name].append(error)
                        self.residuals_history[model_name].append(error)
                        
                    except Exception as e:
                        # В случае ошибки используем среднее значение
                        pred = y_val.iloc[i]
                        self.y_pred_history[model_name].append(pred)
                        self.performance_history[model_name].append(0)
                        self.residuals_history[model_name].append(0)
    
    def _validation_epoch(self, X_val: np.ndarray, y_val: pd.Series):
        """Одна эпоха валидации для настройки theta"""
        for i in range(len(y_val)):
            step_errors = {}
            
            for model_name in self.model_names:
                if model_name not in self.models or self.models[model_name] is None:
                    continue
                
                try:
                    # Получаем усиленные признаки
                    X_enhanced = self._enhance_features(X_val[i:i+1], model_name)
                    
                    # Прогноз
                    if model_name == 'sarima':
                        pred = self.models['sarima'].forecast(steps=1)[0]
                    elif model_name in ['xgboost', 'catboost']:
                        pred = self.models[model_name].predict(X_enhanced)[0]
                    elif model_name == 'lstm' and hasattr(self, 'lstm_seq_length'):
                        if i >= self.lstm_seq_length:
                            X_seq = X_enhanced[0:self.lstm_seq_length].reshape(1, self.lstm_seq_length, -1)
                            pred = self.models['lstm'].predict(X_seq, verbose=0)[0, 0]
                        else:
                            pred = y_val.iloc[i]
                    else:
                        pred = y_val.iloc[i]
                    
                    error = y_val.iloc[i] - pred
                    step_errors[model_name] = error
                    
                    # Обновляем истории
                    self.residuals_history[model_name].append(error)
                    
                except Exception as e:
                    # В случае ошибки пропускаем
                    continue
            
            # Обновляем матрицу theta
            if step_errors:
                self._update_theta_matrix(step_errors)
    
    def cross_validate(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Dict[str, List[float]]:
        """Кросс-валидация DMEN"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_metrics = {'mse': [], 'mae': [], 'r2': []}
        
        y = df[target_col]
        feature_cols = [col for col in df.columns if col != target_col]
        
        if feature_cols:
            X = df[feature_cols].values
        else:
            X = self._create_basic_features(y)
        
        last_test_idx = None
        last_y_test = None
        last_y_pred = None
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            try:
                # Создаем временную модель для этого фолда
                temp_config = ModelConfig(
                    target_col=target_col,
                    window_size=self.window_size,
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    epochs=min(20, self.epochs),  # Уменьшаем эпохи для CV
                    batch_size=self.batch_size,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma
                )
                
                temp_model = DMENModel(temp_config)
                
                # Обучаем на тренировочных данных
                train_df = df.iloc[train_idx]
                temp_model.fit(train_df, target_col)
                
                # Прогнозируем на тестовых данных
                test_df = df.iloc[test_idx]
                horizon = len(test_idx)
                
                # Используем последние данные из тренировочного набора для прогноза
                forecast_series, _ = temp_model.forecast(train_df, horizon)
                
                y_test = test_df[target_col].values
                y_pred = forecast_series.values
                
                # Обрезаем до минимальной длины
                min_len = min(len(y_test), len(y_pred))
                y_test = y_test[:min_len]
                y_pred = y_pred[:min_len]
                
                # Вычисляем метрики
                cv_metrics['mse'].append(mean_squared_error(y_test, y_pred))
                cv_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
                cv_metrics['r2'].append(r2_score(y_test, y_pred))
                
                last_test_idx = test_idx[:min_len]
                last_y_test = y_test
                last_y_pred = y_pred
                
            except Exception as e:
                print(f"Ошибка в фолде {i}: {e}")
                continue
            
            if progress_callback:
                progress_callback(i+1, self.n_splits)
        
        # Сохраняем данные последнего фолда для визуализации
        if last_test_idx is not None:
            self.test_data = {
                'dates': df.index[last_test_idx],
                'actual': last_y_test,
                'predicted': last_y_pred
            }
        
        self.cv_metrics = cv_metrics
        return cv_metrics
    
    def forecast(self, ts: pd.DataFrame, horizon: int, progress_callback=None) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Прогнозирование с динамическим взаимным усилением"""
        if not self._is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        target_col = self.config.target_col
        predictions = []
        
        # Подготовка признаков
        feature_cols = [col for col in ts.columns if col != target_col]
        if feature_cols:
            X_base = ts[feature_cols].values
        else:
            X_base = self._create_basic_features(ts[target_col])
        
        # Создаем будущие даты
        if isinstance(ts.index, pd.DatetimeIndex):
            last_date = ts.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        else:
            future_dates = pd.RangeIndex(start=len(ts), stop=len(ts) + horizon)
        
        # Инициализируем историю последними значениями
        recent_values = ts[target_col].tail(self.window_size).values
        self.recent_history = deque(recent_values, maxlen=max(30, self.window_size))
        
        for h in range(horizon):
            step_predictions = {}
            
            # Получаем прогнозы от каждой модели
            for model_name in self.model_names:
                if model_name not in self.models or self.models[model_name] is None:
                    continue
                
                try:
                    if model_name == 'sarima':
                        # SARIMA прогноз
                        pred = self.models['sarima'].forecast(steps=1)[0]
                        
                    elif model_name in ['xgboost', 'catboost']:
                        # Создаем признаки для будущего шага
                        if h < len(X_base):
                            X_h = X_base[-(horizon-h):-(horizon-h-1)] if h < horizon-1 else X_base[-1:].copy()
                        else:
                            # Экстраполируем признаки
                            X_h = X_base[-1:].copy()
                        
                        # Усиливаем признаки остатками других моделей
                        X_enhanced = self._enhance_features(X_h, model_name)
                        
                        # Прогноз
                        pred = self.models[model_name].predict(X_enhanced)[0]
                        
                    elif model_name == 'lstm' and hasattr(self, 'lstm_seq_length'):
                        # LSTM требует последовательность
                        if len(self.recent_history) >= self.lstm_seq_length:
                            # Берем последние значения для создания последовательности
                            recent_values = list(self.recent_history)[-self.lstm_seq_length:]
                            
                            # Создаем признаки для последовательности
                            if len(X_base) >= self.lstm_seq_length:
                                X_seq = X_base[-self.lstm_seq_length:].reshape(1, self.lstm_seq_length, -1)
                            else:
                                # Дополняем последовательность
                                X_seq = np.tile(X_base[-1], (self.lstm_seq_length, 1)).reshape(1, self.lstm_seq_length, -1)
                            
                            pred = self.models['lstm'].predict(X_seq, verbose=0)[0, 0]
                        else:
                            pred = np.mean(list(self.recent_history)) if self.recent_history else 0
                    else:
                        # Fallback
                        pred = np.mean(list(self.recent_history)) if self.recent_history else 0
                    
                    step_predictions[model_name] = pred
                    
                except Exception as e:
                    # В случае ошибки используем среднее значение
                    pred = np.mean(list(self.recent_history)) if self.recent_history else 0
                    step_predictions[model_name] = pred
            
            # Обновляем динамические веса
            self.dynamic_weights = self._calculate_dynamic_weights()
            
            # Взвешенная комбинация
            if step_predictions:
                final_prediction = sum(
                    (self.dynamic_weights.get(model, self._get_default_weight(model)) or self._get_default_weight(model)) * pred 
                    for model, pred in step_predictions.items()
                )
            else:
                final_prediction = np.mean(list(self.recent_history)) if self.recent_history else 0
            
            predictions.append(final_prediction)
            
            # Обновляем историю для следующего шага
            self.recent_history.append(final_prediction)
            
            # Обновляем остатки (используем простую эвристику)
            for model_name, pred in step_predictions.items():
                residual = final_prediction - pred  # Разность между ансамблем и индивидуальной моделью
                self.residuals_history[model_name].append(residual)
            
            if progress_callback:
                progress_callback(h+1, horizon)
        
        # Создаем Series с прогнозом
        forecast_series = pd.Series(predictions, index=future_dates)
        
        return forecast_series, None  # DMEN не предоставляет доверительные интервалы
    
    def get_metrics(self) -> Dict[str, float]:
        """Получить метрики производительности"""
        if self.cv_metrics is None:
            return {}
        
        # Возвращаем средние значения метрик
        return {
            'mse': np.mean(self.cv_metrics['mse']) if self.cv_metrics['mse'] else 0,
            'mae': np.mean(self.cv_metrics['mae']) if self.cv_metrics['mae'] else 0,
            'r2': np.mean(self.cv_metrics['r2']) if self.cv_metrics['r2'] else 0
        }
    
    def plot_test_predictions(self) -> go.Figure:
        """Построить график прогноз vs факт для последнего тестового фолда"""
        if not hasattr(self, 'test_data') or self.test_data is None:
            raise ValueError("Нет данных теста для построения графика. Сначала выполните cross_validate().")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.test_data['dates'],
            y=self.test_data['actual'],
            mode='lines+markers',
            name='Факт',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.test_data['dates'],
            y=self.test_data['predicted'],
            mode='lines+markers',
            name='DMEN Прогноз',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='DMEN: прогноз vs факт (последний фолд)',
            xaxis_title='Дата',
            yaxis_title='Значение',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_forecast(self, ts: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None) -> go.Figure:
        """Построить график прогноза"""
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            mode='lines',
            name='Исторические данные',
            line=dict(color='blue')
        ))
        
        # Прогноз
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='DMEN Прогноз',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='DMEN: Прогноз временного ряда',
            xaxis_title='Дата',
            yaxis_title='Значение',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def get_model_weights(self) -> Dict[str, float]:
        """Получить текущие динамические веса моделей"""
        return self.dynamic_weights.copy()
    
    def get_theta_matrix(self) -> np.ndarray:
        """Получить матрицу взаимного усиления"""
        return self.theta.copy()
    
    def plot_model_weights(self) -> go.Figure:
        """Построить график динамических весов моделей"""
        weights = self.get_model_weights()
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                marker_color=['blue', 'green', 'orange', 'red']
            )
        ])
        
        fig.update_layout(
            title='Динамические веса моделей в DMEN',
            xaxis_title='Модель',
            yaxis_title='Вес',
            showlegend=False
        )
        
        return fig
    
    def _get_default_weight(self, model_name: str) -> float:
        """Получить приоритетный вес для модели"""
        model_priorities = {
            'sarima': 0.8,      # Меньший приоритет для SARIMA
            'xgboost': 1.3,     # Больший приоритет для регрессоров
            'catboost': 1.3,    # Больший приоритет для регрессоров
            'lstm': 1.0         # Нейтральный приоритет для LSTM
        }
        total_priority = sum(model_priorities.values())
        return model_priorities.get(model_name, 1.0) / total_priority 