"""
Сервис для трансформации данных
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import median_abs_deviation
from config import TRANSFORMATION_CONFIG

class TransformationService:
    """Класс для трансформации данных"""
    
    @staticmethod
    def detect_outliers(
        data: pd.Series,
        method: str = 'IQR',
        **kwargs
    ) -> pd.Series:
        """
        Обнаружение выбросов
        
        Args:
            data: Временной ряд
            method: Метод обнаружения
            **kwargs: Дополнительные параметры
            
        Returns:
            pd.Series: Маска выбросов
        """
        outliers_mask = pd.Series(False, index=data.index)
        
        if method == 'IQR':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            threshold = kwargs.get('threshold', 1.5)
            outliers_mask = (data < (q1 - threshold*iqr)) | (data > (q3 + threshold*iqr))
        elif method == 'Z-score':
            threshold = kwargs.get('threshold', 3)
            z = np.abs((data - data.mean()) / data.std())
            outliers_mask = z > threshold
        elif method == 'Isolation Forest':
            model = IsolationForest(
                contamination=kwargs.get('contamination', 0.1),
                random_state=42
            )
            preds = model.fit_predict(data.values.reshape(-1, 1))
            outliers_mask = preds == -1
        elif method == 'DBSCAN':
            model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
            clusters = model.fit_predict(data.values.reshape(-1, 1))
            outliers_mask = clusters == -1
        elif method == 'LOF':
            lof = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 0.1)
            )
            preds = lof.fit_predict(data.values.reshape(-1, 1))
            outliers_mask = preds == -1
        elif method == 'Robust Z-score':
            median = data.median()
            mad = median_abs_deviation(data, scale='normal')
            z_scores = np.abs((data - median) / mad)
            outliers_mask = z_scores > kwargs.get('threshold', 3)
        
        return outliers_mask
    
    @staticmethod
    def replace_outliers(
        data: pd.Series,
        outliers_mask: pd.Series,
        method: str = 'median',
        **kwargs
    ) -> pd.Series:
        """
        Замена выбросов
        
        Args:
            data: Временной ряд
            outliers_mask: Маска выбросов
            method: Метод замены
            **kwargs: Дополнительные параметры
            
        Returns:
            pd.Series: Ряд с замененными выбросами
        """
        if method == 'median':
            return data.mask(outliers_mask, data.median())
        elif method == 'moving_average':
            window_size = kwargs.get('window_size', 5)
            rolling_mean = data.rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean().ffill().bfill()
            return data.mask(outliers_mask, rolling_mean)
        elif method == 'interpolation':
            interpolation_method = kwargs.get('interpolation_method', 'linear')
            temp = data.mask(outliers_mask, np.nan)
            return temp.interpolate(
                method=interpolation_method,
                limit_direction='both',
                limit_area='inside'
            ).ffill().bfill()
        
        return data
    
    @staticmethod
    def scale_data(
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'StandardScaler'
    ) -> pd.DataFrame:
        """
        Масштабирование данных
        
        Args:
            df: DataFrame с данными
            columns: Список столбцов для масштабирования
            method: Метод масштабирования
            
        Returns:
            pd.DataFrame: Масштабированный DataFrame
        """
        scaled_df = df.copy()
        
        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif method == "RobustScaler":
            scaler = RobustScaler()
        
        scaled_df[columns] = scaler.fit_transform(scaled_df[columns])
        return scaled_df
    
    @staticmethod
    def create_features(
        df: pd.DataFrame,
        target_col: str,
        feature_type: str,
        date_col: str = 'date',
        **kwargs
    ) -> pd.Series:
        """
        Создание признаков
        
        Args:
            df: DataFrame с данными
            target_col: Название столбца с целевой переменной
            feature_type: Тип признака
            date_col: Название столбца с датой
            **kwargs: Дополнительные параметры
            
        Returns:
            pd.Series: Новый признак
        """
        ts = df.set_index(date_col)[target_col]
        
        if feature_type == "Скользящее среднее":
            window = kwargs.get('window', 7)
            min_periods = kwargs.get('min_periods', 1)
            return ts.rolling(window=window, min_periods=min_periods).mean()
        elif feature_type == "Скользящее стандартное отклонение":
            window = kwargs.get('window', 7)
            min_periods = kwargs.get('min_periods', 2)
            return ts.rolling(window=window, min_periods=min_periods).std()
        elif feature_type == "Разница":
            diff_order = kwargs.get('diff_order', 1)
            return ts.diff(diff_order)
        elif feature_type == "Процентное изменение":
            periods = kwargs.get('periods', 1)
            return ts.pct_change(periods)
        elif feature_type == "Месяц":
            return ts.index.month
        elif feature_type == "Квартал":
            return ts.index.quarter
        elif feature_type == "День недели":
            return ts.index.dayofweek
        elif feature_type == "День месяца":
            return ts.index.day
        elif feature_type == "День года":
            return ts.index.dayofyear
        elif feature_type == "Неделя года":
            return ts.index.isocalendar().week.astype(int)
        
        raise ValueError(f"Неизвестный тип признака: {feature_type}") 