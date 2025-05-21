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
        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {', '.join(df.columns)}")
            
        # Use feature_names if available, otherwise use all columns except target
        if self.feature_names is not None:
            # Make a copy to avoid modifying the original
            feature_cols = [col for col in self.feature_names if col != target_col and col in df.columns]
        else:
            feature_cols = [col for col in df.columns if col != target_col]
            
        # Validate we have at least some features
        if not feature_cols:
            raise ValueError(f"No feature columns found. Available columns: {', '.join(df.columns)}")
            
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
        
        # Reduce complexity and regularization strength
        # First LSTM layer with return sequences
        model.add(layers.LSTM(
            self.hidden_size,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.001),  # Reduced from 0.01
            recurrent_regularizer=keras.regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))  # Reduced from 0.3
        
        # Second LSTM layer
        model.add(layers.LSTM(
            self.hidden_size // 2,
            kernel_regularizer=keras.regularizers.l2(0.001),
            recurrent_regularizer=keras.regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        
        # Simplified - removed extra LSTM layer and reduced dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.1))  # Reduced from 0.2
        
        model.add(layers.Dense(1))
        
        # Use Adam optimizer with fixed learning rate
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def fit(self, df: pd.DataFrame, target_col: str, progress_callback=None) -> Tuple[Any, Dict]:
        feature_names = [col for col in df.columns if col != target_col]
        if not feature_names:
            raise ValueError("Нет признаков для обучения модели. Добавьте признаки на этапе подготовки данных.")
        
        # Store both feature names and target column for later use
        self.feature_names = feature_names
        self.config.target_col = target_col
        
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
                patience=15,  # Increased from 10
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
            batch_size=min(32, len(X_train_seq) // 10),  # Dynamic batch size based on data size
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
        
        # Ensure the target column is in the dataframe
        target_col = self.config.target_col
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Columns available: {', '.join(df.columns)}")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Create a new dataframe for forecasted values with same index structure
        last_date = df_copy.index[-1]
        freq = pd.infer_freq(df_copy.index)
        if freq is None:
            # Try to infer frequency from the last few dates
            try:
                freq = pd.infer_freq(df_copy.index[-5:])
            except:
                freq = 'D'  # Default to daily
        
        # Create future index
        future_index = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        # Ensure all required features are present
        all_features = self.feature_names + [target_col] if self.feature_names else df_copy.columns.tolist()
        for feature in all_features:
            if feature not in df_copy.columns and '_lag' not in feature:
                raise ValueError(f"Feature '{feature}' not found in dataframe")
        
        # Create lag features if needed
        for feature in all_features:
            if '_lag' in feature and feature not in df_copy.columns:
                parts = feature.split('_lag')
                base_feature = parts[0]
                lag_num = int(parts[1])
                if base_feature in df_copy.columns:
                    df_copy[feature] = df_copy[base_feature].shift(lag_num)
        
        # Handle NaN values from creating lags
        df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
        
        # Ensure we have enough history for the window size
        if len(df_copy) < self.window_size:
            raise ValueError(f"Not enough data points ({len(df_copy)}) for the window size ({self.window_size})")
        
        # Initialize with the last window_size rows of actual data
        forecast_input = df_copy.iloc[-self.window_size:].copy()
        forecasted_values = []
        
        # If the dataframe has datetime features, calculate them for the future dates
        datetime_features = ['day', 'month', 'year', 'day_of_week', 'quarter', 
                           'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 
                           'dayofweek_sin', 'dayofweek_cos']
        has_datetime_features = any(feat in df_copy.columns for feat in datetime_features)
        
        # Generate forecasts one step at a time
        for i in range(horizon):
            # Format the data for prediction
            X_forecast = forecast_input.copy()
            X, _ = self.create_features(X_forecast, target_col)
            X_seq = X.reshape(1, self.window_size, -1)
            
            # Make prediction
            y_pred = self.model.predict(X_seq, verbose=0)[0][0]
            
            # Convert prediction back to original scale
            y_pred_orig = self.scaler_y.inverse_transform(np.array([[y_pred]]))[0][0]
            forecasted_values.append(y_pred_orig)
            
            # Create next row for the rolling forecast
            next_date = future_index[i]
            next_row = pd.Series(index=forecast_input.columns)
            
            # Set the target value to the prediction
            next_row[target_col] = y_pred_orig
            
            # If we have datetime features, calculate them for the future date
            if has_datetime_features:
                if 'day' in forecast_input.columns:
                    next_row['day'] = next_date.day
                if 'month' in forecast_input.columns:
                    next_row['month'] = next_date.month
                if 'year' in forecast_input.columns:
                    next_row['year'] = next_date.year
                if 'day_of_week' in forecast_input.columns:
                    next_row['day_of_week'] = next_date.dayofweek
                if 'quarter' in forecast_input.columns:
                    next_row['quarter'] = next_date.quarter
                
                # Calculate cyclical features if present
                if 'month_sin' in forecast_input.columns:
                    next_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
                if 'month_cos' in forecast_input.columns:
                    next_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
                if 'quarter_sin' in forecast_input.columns:
                    next_row['quarter_sin'] = np.sin(2 * np.pi * next_date.quarter / 4)
                if 'quarter_cos' in forecast_input.columns:
                    next_row['quarter_cos'] = np.cos(2 * np.pi * next_date.quarter / 4)
                if 'dayofweek_sin' in forecast_input.columns:
                    next_row['dayofweek_sin'] = np.sin(2 * np.pi * next_date.dayofweek / 7)
                if 'dayofweek_cos' in forecast_input.columns:
                    next_row['dayofweek_cos'] = np.cos(2 * np.pi * next_date.dayofweek / 7)
            
            # Copy other feature values from the last prediction (if any)
            for col in forecast_input.columns:
                if col != target_col and col not in datetime_features and pd.isna(next_row[col]):
                    if col.endswith('_lag1'):
                        # For lag 1 features, use the predicted value
                        base_col = col.replace('_lag1', '')
                        if base_col == target_col:
                            next_row[col] = y_pred_orig
                        else:
                            next_row[col] = forecast_input.iloc[-1][base_col]
                    elif '_lag' in col:
                        # For other lag features, shift appropriately
                        parts = col.split('_lag')
                        base_col = parts[0]
                        lag = int(parts[1])
                        if lag > 1:
                            prev_lag_col = f"{base_col}_lag{lag-1}"
                            if prev_lag_col in forecast_input.columns:
                                next_row[col] = forecast_input.iloc[-1][prev_lag_col]
                    else:
                        # For other features, carry forward
                        next_row[col] = forecast_input.iloc[-1][col]
            
            # Add the new row to the forecast_input
            next_row_df = pd.DataFrame([next_row], index=[next_date])
            forecast_input = pd.concat([forecast_input.iloc[1:], next_row_df])
            
            if progress_callback:
                progress_callback(i+1, horizon)
        
        # Create the forecast series
        forecast_series = pd.Series(forecasted_values, index=future_index)
        
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