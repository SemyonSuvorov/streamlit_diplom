import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional, Any
from components.forecasting.base_model import BaseModel, ModelConfig
from state.session import state
from tensorflow import keras
from keras import layers

class LSTMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.train_size = getattr(config, 'train_size', 0.8) or 0.8
        self.n_splits = getattr(config, 'n_splits', 5) or 5
        self.window_size = getattr(config, 'window_size', 14) or 14
        self.epochs = getattr(config, 'epochs', 20) or 20
        self.batch_size = getattr(config, 'batch_size', 32) or 32
        self.hidden_size = getattr(config, 'hidden_size', 64) or 64
        self.num_layers = getattr(config, 'num_layers', 2) or 2
        self.learning_rate = getattr(config, 'learning_rate', 0.001) or 0.001
        self.model = None
        self.cv_metrics = None
        self.feature_names = None
        self.test_data = None
        self._is_fitted = False
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def create_sequences(self, X: np.ndarray, y: np.ndarray, window_size: int):
        Xs, ys = [], []
        for i in range(len(X) - window_size):
            Xs.append(X[i:i+window_size])
            ys.append(y[i+window_size])
        return np.array(Xs), np.array(ys)

    def create_features(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[[target_col]].values.astype(np.float32)
        if self._is_fitted:
            X = self.scaler_x.transform(X)
            y = self.scaler_y.transform(y)
        else:
            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y)
        self.feature_names = feature_cols
        return X, y.squeeze()

    def build_model(self, input_shape):
        model = keras.Sequential()
        model.add(keras.Input(shape=input_shape))
        
        # First LSTM layer with return sequences
        model.add(layers.LSTM(
            self.hidden_size,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.01),
            recurrent_regularizer=keras.regularizers.l2(0.01)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Second LSTM layer
        model.add(layers.LSTM(
            self.hidden_size // 2,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.01),
            recurrent_regularizer=keras.regularizers.l2(0.01)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Third LSTM layer
        model.add(layers.LSTM(
            self.hidden_size // 4,
            kernel_regularizer=keras.regularizers.l2(0.01),
            recurrent_regularizer=keras.regularizers.l2(0.01)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(1))
        
        # Use Adam optimizer with learning rate decay
        initial_learning_rate = self.learning_rate
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def fit(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Tuple[Any, Dict]:
        feature_names = [col for col in df.columns if col != target_col]
        if not feature_names:
            raise ValueError("Нет признаков для обучения модели. Добавьте признаки на этапе подготовки данных.")
        
        # Split data into train and validation sets
        train_size = int(len(df) * self.train_size)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        if state.get('feature_df') is None:
            state.set('feature_df', df.copy())
            
        X_train, y_train = self.create_features(train_df, target_col)
        X_test, y_test = self.create_features(test_df, target_col)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, self.window_size)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, self.window_size)
        
        # Build and compile model
        self.model = self.build_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
        
        if progress_callback:
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, total_epochs):
                    super().__init__()
                    self.total_epochs = total_epochs
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress_callback(epoch+1, self.total_epochs)
            callbacks.append(ProgressCallback(self.epochs))
        
        # Train model with validation split
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self._is_fitted = True
        
        # Make predictions using predict
        preds = self.model.predict(X_test_seq).squeeze()
        preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).squeeze()
        y_test_seq = self.scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).squeeze()
        
        min_len = min(len(y_test_seq), len(preds))
        y_test_seq = y_test_seq[:min_len]
        preds = preds[:min_len]
        
        metrics = {
            'mse': mean_squared_error(y_test_seq, preds),
            'mae': mean_absolute_error(y_test_seq, preds),
            'r2': r2_score(y_test_seq, preds)
        }
        
        self.test_data = {
            'dates': test_df.index[self.window_size:self.window_size+min_len],
            'actual': y_test_seq,
            'predicted': preds
        }
        
        return self.model, metrics

    def cross_validate(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Dict[str, List[float]]:
        if not self._is_fitted:
            raise ValueError("Model must be trained first using fit() method")
            
        X, y = self.create_features(df, target_col)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_metrics = {'mse': [], 'mae': [], 'r2': []}
        last_test_idx = None
        last_y_test = None
        last_y_pred = None
        n_folds = self.n_splits
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            test_X, test_y = X[test_idx], y[test_idx]
            X_test_seq, y_test_seq = self.create_sequences(test_X, test_y, self.window_size)
            
            # Use predict instead of forecast for cross-validation
            preds = self.model.predict(X_test_seq).squeeze()
            preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).squeeze()
            y_test_seq = self.scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).squeeze()
            
            min_len = min(len(y_test_seq), len(preds))
            y_test_seq = y_test_seq[:min_len]
            preds = preds[:min_len]
            
            cv_metrics['mse'].append(mean_squared_error(y_test_seq, preds))
            cv_metrics['mae'].append(mean_absolute_error(y_test_seq, preds))
            cv_metrics['r2'].append(r2_score(y_test_seq, preds))
            
            last_test_idx = test_idx[self.window_size:self.window_size+min_len]
            last_y_test = y_test_seq
            last_y_pred = preds
            
            if progress_callback:
                progress_callback(i+1, n_folds)
        
        if last_test_idx is not None:
            self.test_data = {
                'dates': df.index[last_test_idx],
                'actual': last_y_test,
                'predicted': last_y_pred
            }
        
        self.cv_metrics = cv_metrics
        return cv_metrics

    def forecast(self, df: pd.DataFrame, horizon: int, progress_callback=None) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        if self.model is None or not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.config.target_col) if hasattr(df, 'to_frame') else pd.DataFrame({self.config.target_col: df})
        
        X, y = self.create_features(df, self.config.target_col)
        
        # Create sequences for prediction
        X_seq, _ = self.create_sequences(X, y, self.window_size)
        
        # Make predictions only for the specified horizon
        preds = self.model.predict(X_seq[:horizon]).squeeze()
        preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).squeeze()
        
        if progress_callback:
            progress_callback(horizon, horizon)

        # Create dates for the forecast
        last_date = df.index[-1]
        freq = pd.infer_freq(df.index)
        if freq is None:
            freq = 'D'
        forecast_index = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        # Create forecast series
        forecast_series = pd.Series(preds, index=forecast_index)
        return forecast_series, None

    def get_metrics(self) -> Dict[str, float]:
        if not hasattr(self, 'test_data') or self.test_data is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return {
            'mse': mean_squared_error(self.test_data['actual'], self.test_data['predicted']),
            'mae': mean_absolute_error(self.test_data['actual'], self.test_data['predicted']),
            'r2': r2_score(self.test_data['actual'], self.test_data['predicted'])
        }

    def plot_forecast(self, ts: pd.Series, forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None) -> Any:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Исторические данные'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Прогноз', line=dict(color='red', dash='dash')))
        fig.update_layout(
            title='LSTM Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            hovermode='x unified'
        )
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

    def plot_test_predictions(self) -> go.Figure:
        if not hasattr(self, 'test_data') or self.test_data is None:
            raise ValueError("Нет данных теста для построения графика. Сначала выполните cross_validate() или fit().")
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
            title='LSTM: прогноз vs факт (последний фолд)',
            xaxis_title='Дата',
            yaxis_title='Значение',
            showlegend=True,
            hovermode='x unified'
        )
        return fig 