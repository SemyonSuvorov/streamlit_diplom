import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional, Any
from components.forecasting.base_model import BaseModel, ModelConfig
from state.session import state

class XGBoostModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.train_size = config.train_size or 0.8
        self.n_splits = config.n_splits or 5
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.random_state = 42
        self.model = None
        self.scaler = None
        self.cv_metrics = None
        self.requires_datetime_features = True
        
    def create_features(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Использовать все фичи кроме целевой переменной"""
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        return X, y
    
    def cross_validate(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Dict[str, List[float]]:
        X, y = self.create_features(df, target_col)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_metrics = {'mse': [], 'mae': [], 'r2': []}
        last_test_idx = None
        last_y_test = None
        last_y_pred = None
        n_folds = self.n_splits
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Prepare training data
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            
            # Save feature names before fitting
            self.feature_names = [col for col in train_df.columns if col != target_col]
            
            # Fit model on training data
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
            self.model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
            
            # Use predict method directly on test data instead of forecast
            try:
                X_test = test_df.drop(columns=[target_col])
                y_test = test_df[target_col].values
                y_pred = self.model.predict(X_test)
                
                # Compute metrics
                cv_metrics['mse'].append(mean_squared_error(y_test, y_pred))
                cv_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
                cv_metrics['r2'].append(r2_score(y_test, y_pred))
                
                last_test_idx = test_idx
                last_y_test = y_test
                last_y_pred = y_pred
            except Exception as e:
                print(f"Error in fold {i}: {e}")
                continue
            
            if progress_callback is not None:
                progress_callback(i+1, n_folds)
                
        if last_test_idx is not None:
            self.test_data = {
                'dates': df.index[last_test_idx],
                'actual': last_y_test,
                'predicted': last_y_pred
            }
        self.cv_metrics = cv_metrics
        return cv_metrics
    
    def fit(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Tuple[Any, Dict]:
        feature_names = [col for col in df.columns if col != target_col]
        if not feature_names:
            raise ValueError("Нет признаков для обучения модели. Добавьте признаки на этапе подготовки данных.")
        
        #train_size = int(len(df) * self.train_size)
        train_df = df
        #test_df = df.iloc[train_size:].copy()

        if state.get('feature_df') is None:
            state.set('feature_df', df.copy())

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
        
        # # Use forecast method to predict test set
        # horizon = len(test_df)
        
        # # Override to use direct forecasting for evaluation
        # old_approach = getattr(self.config, 'forecast_approach', 'recursive')
        # self.config.forecast_approach = 'direct'
        
        # try:
        #     forecast_series, _ = self.forecast(train_df, horizon)
            
        #     # Calculate metrics
        #     y_test = test_df[target_col].values
        #     y_pred = forecast_series.values
            
        #     # Ensure same length
        #     min_len = min(len(y_test), len(y_pred))
        #     y_test = y_test[:min_len]
        #     y_pred = y_pred[:min_len]
            
        #     metrics = {
        #         'mse': mean_squared_error(y_test, y_pred),
        #         'mae': mean_absolute_error(y_test, y_pred),
        #         'r2': r2_score(y_test, y_pred)
        #     }
            
        #     self.test_data = {
        #         'dates': test_df.index[:min_len],
        #         'actual': y_test,
        #         'predicted': y_pred
        #     }
        # finally:
        #     # Restore original approach
        #     self.config.forecast_approach = old_approach
            
        self.feature_names = feature_names
        self._is_fitted = True
        return self.model#, metrics
    
    def plot_test_predictions(self) -> go.Figure:
        """Построить график прогноз vs факт для последнего тестового фолда после cross_validate"""
        if not hasattr(self, 'test_data') or self.test_data is None:
            raise ValueError("Нет данных теста для построения графика. Сначала выполните cross_validate().")
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
            title='XGBoost: прогноз vs факт (последний фолд)',
            xaxis_title='Дата',
            yaxis_title='Значение',
            showlegend=True,
            hovermode='x unified'
        )
        return fig
    
    def _future_time_features(self, future_index, feature_names):
        df = pd.DataFrame(index=future_index)
        for col in feature_names:
            if col == 'day':
                df[col] = df.index.day
            elif col == 'month':
                df[col] = df.index.month
            elif col == 'year':
                df[col] = df.index.year
            elif col == 'day_of_week':
                df[col] = df.index.dayofweek
            elif col == 'quarter':
                df[col] = df.index.quarter
            elif col == 'day_of_year':
                df[col] = df.index.dayofyear
            elif col == 'week':
                df[col] = df.index.isocalendar().week.astype(int)
        return df

    def _cyclic_time_features(self, df):
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        return df

    def _update_dynamic_features(self, df, target_col):
        for col in df.columns:
            if col.startswith('rolling_mean_'):
                window = int(col.split('_')[-1])
                df[col] = df[target_col].rolling(window=window, min_periods=1).mean()
            elif col.startswith('rolling_std_'):
                window = int(col.split('_')[-1])
                df[col] = df[target_col].rolling(window=window, min_periods=1).std()
            elif col.startswith('diff_'):
                order = int(col.split('_')[-1])
                df[col] = df[target_col].diff(order)
            elif col.startswith('pct_change_'):
                periods = int(col.split('_')[-1])
                df[col] = df[target_col].pct_change(periods)
        return df

    def _generate_next_features(self, ts_for_pred, new_date, required_features, future_time_df=None, i=None):
        new_row = {}
        target_col = self.config.target_col
        for col in required_features:
            if col == target_col:
                continue
            if col in ts_for_pred.columns:
                if col.startswith('rolling_mean_'):
                    window = int(col.split('_')[-1])
                    new_row[col] = ts_for_pred[target_col].iloc[-window:].mean()
                elif col.startswith('rolling_std_'):
                    window = int(col.split('_')[-1])
                    new_row[col] = ts_for_pred[target_col].iloc[-window:].std()
                elif col.startswith('diff_'):
                    order = int(col.split('_')[-1])
                    if len(ts_for_pred[target_col]) > order:
                        new_row[col] = ts_for_pred[target_col].iloc[-1] - ts_for_pred[target_col].iloc[-1 - order]
                    else:
                        new_row[col] = 0.0
                elif col.startswith('pct_change_'):
                    periods = int(col.split('_')[-1])
                    if len(ts_for_pred[target_col]) > periods and ts_for_pred[target_col].iloc[-1 - periods] != 0:
                        new_row[col] = (ts_for_pred[target_col].iloc[-1] / ts_for_pred[target_col].iloc[-1 - periods]) - 1
                    else:
                        new_row[col] = 0.0
                elif future_time_df is not None and col in future_time_df.columns and i is not None:
                    new_row[col] = future_time_df.iloc[i][col]
                else:
                    new_row[col] = ts_for_pred[col].iloc[-1]
            else:
                new_row[col] = 0.0
        return new_row

    def forecast(self, ts: pd.DataFrame, horizon: int, progress_callback=None) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if isinstance(ts, pd.Series):
            ts = ts.to_frame(name=self.config.target_col) if hasattr(ts, 'to_frame') else pd.DataFrame({self.config.target_col: ts})
            
        required_features = list(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else self.feature_names
        feature_cols = [col for col in ts.columns if col != self.config.target_col]
        if not feature_cols:
            raise ValueError("Для прогнозирования нужны признаки, кроме целевой переменной. Добавьте признаки на этапе подготовки данных.")
        missing = [f for f in required_features if f not in ts.columns]
        if missing:
            raise ValueError(f"Для прогноза не хватает признаков: {missing}. Проверьте этап подготовки данных.")
            
        last_date = ts.index[-1]
        freq = pd.infer_freq(ts.index)
        if freq is None:
            freq = 'D'
        future_index = pd.date_range(start=last_date + pd.Timedelta(1, unit=freq[0]), periods=horizon, freq=freq)
        
        if getattr(self.config, 'forecast_approach', 'recursive') == 'direct':
            X = ts[feature_cols].values
            y = ts[self.config.target_col].values
            models = []
            forecasts = []
            self.feature_names = feature_cols  # Сохраняем feature_names
            
            for h in range(1, horizon+1):
                y_shifted = np.roll(y, -h)
                X_train = X[:-h]
                y_train = y_shifted[:-h]
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)
                X_pred = X[-1:].copy()
                y_pred = model.predict(X_pred)[0]
                forecasts.append(y_pred)
                models.append(model)
                if progress_callback is not None:
                    progress_callback(h, horizon)
            forecast_series = pd.Series(forecasts, index=future_index)
            self.direct_models = models
            return forecast_series, None
        else:
            # Создаем начальное окно для прогнозирования
            ts_for_pred = ts.copy()
            forecast_values = []
            future_time_df = self._future_time_features(future_index, required_features) if hasattr(self, '_future_time_features') else None
            
            for i in range(horizon):
                # Генерируем признаки для следующего шага
                new_date = future_index[i]
                new_row = self._generate_next_features(ts_for_pred, new_date, required_features, future_time_df, i)
                X_pred = np.array([new_row[f] for f in required_features]).reshape(1, -1)
                
                # Делаем прогноз
                y_pred = self.model.predict(X_pred)[0]
                forecast_values.append(y_pred)
                
                # Обновляем данные для следующего шага
                new_row[self.config.target_col] = y_pred
                new_row_df = pd.DataFrame([new_row], index=[new_date])
                ts_for_pred = pd.concat([ts_for_pred, new_row_df], sort=False)
                
                if progress_callback is not None:
                    progress_callback(i+1, horizon)
                    
            forecast_series = pd.Series(forecast_values, index=future_index)
            return forecast_series, None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        if not hasattr(self, 'test_data'):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return {
            'mse': mean_squared_error(self.test_data['actual'], self.test_data['predicted']),
            'mae': mean_absolute_error(self.test_data['actual'], self.test_data['predicted']),
            'r2': r2_score(self.test_data['actual'], self.test_data['predicted'])
        }
    
    def plot_forecast(self, ts: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None) -> Any:
        """Plot the forecast results"""
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            name='Historical'
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='XGBoost Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            hovermode='x unified'
        )
        
        # Add metrics annotation
        metrics = self.get_metrics()
        metrics_text = '<br>'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text=metrics_text,
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig 