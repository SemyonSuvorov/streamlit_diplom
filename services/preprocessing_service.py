"""
Сервис для предобработки данных
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import PREPROCESSING_CONFIG

class PreprocessingService:
    """Класс для предобработки данных"""
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        target_col: str,
        method: str = 'linear',
        **kwargs
    ) -> pd.DataFrame:
        """
        Обработка пропущенных значений
        
        Args:
            df: DataFrame с данными
            target_col: Название столбца с целевой переменной
            method: Метод заполнения пропусков
            **kwargs: Дополнительные параметры
            
        Returns:
            pd.DataFrame: DataFrame с заполненными пропусками
        """
        filled_df = df.copy()
        
        if method == 'time':
            temp_df = filled_df.set_index('date')
            temp_df[target_col] = temp_df[target_col].interpolate(method='time')
            filled_df = temp_df.reset_index()
        elif method == 'linear':
            filled_df[target_col] = filled_df[target_col].interpolate(method='linear')
        elif method in ['ffill', 'bfill']:
            filled_df[target_col] = filled_df[target_col].fillna(method=method)
        elif method == 'mean':
            filled_df[target_col] = filled_df[target_col].fillna(filled_df[target_col].mean())
        elif method == 'zero':
            filled_df[target_col] = filled_df[target_col].fillna(0)
            
        return filled_df
    
    @staticmethod
    def handle_duplicates(
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        Обработка дубликатов
        
        Args:
            df: DataFrame с данными
            date_col: Название столбца с датой
            target_col: Название столбца с целевой переменной
            strategy: Стратегия объединения
            
        Returns:
            pd.DataFrame: DataFrame без дубликатов
        """
        dedup_df = df.groupby(date_col, as_index=False).agg({target_col: strategy})
        return dedup_df
    
    @staticmethod
    def add_missing_dates(
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Добавление недостающих дат
        
        Args:
            df: DataFrame с данными
            date_col: Название столбца с датой
            target_col: Название столбца с целевой переменной
            freq: Частота временного ряда
            
        Returns:
            pd.DataFrame: DataFrame с добавленными датами
        """
        df[date_col] = pd.to_datetime(df[date_col])
        full_range = pd.date_range(
            start=df[date_col].min(),
            end=df[date_col].max(),
            freq=freq
        )
        
        new_index_df = pd.DataFrame({date_col: full_range})
        merged_df = pd.merge(
            new_index_df,
            df,
            on=date_col,
            how='left'
        )
        
        return merged_df
    
    @staticmethod
    def perform_stl_decomposition(
        df: pd.DataFrame,
        target_col: str,
        seasonal_period: int,
        seasonal_smoothing: int = 7,
        trend_smoothing: int = 13,
        low_pass_smoothing: Optional[int] = None,
        decomposition_type: str = "additive"
    ) -> Dict[str, pd.Series]:
        """
        Выполнение STL-декомпозиции
        
        Args:
            df: DataFrame с данными
            target_col: Название столбца с целевой переменной
            seasonal_period: Период сезонности
            seasonal_smoothing: Параметр сглаживания сезонности
            trend_smoothing: Параметр сглаживания тренда
            low_pass_smoothing: Параметр низкочастотного сглаживания
            decomposition_type: Тип декомпозиции
            
        Returns:
            Dict[str, pd.Series]: Компоненты декомпозиции
        """
        from statsmodels.tsa.seasonal import STL
        
        if low_pass_smoothing is None:
            low_pass_smoothing = seasonal_period + (1 if seasonal_period%2 == 0 else 2)
        
        ts = df[target_col].ffill().dropna()
        
        if decomposition_type == "multiplicative":
            if (ts <= 0).any():
                raise ValueError("Мультипликативная модель требует положительных значений")
            ts = np.log(ts)
        
        stl = STL(
            ts,
            period=seasonal_period,
            seasonal=seasonal_smoothing,
            trend=trend_smoothing,
            low_pass=low_pass_smoothing,
            robust=True
        ).fit()
        
        if decomposition_type == "multiplicative":
            trend = np.exp(stl.trend)
            seasonal = np.exp(stl.seasonal) - 1
            resid = np.exp(stl.resid) - 1
            original = np.exp(ts)
        else:
            trend = stl.trend
            seasonal = stl.seasonal
            resid = stl.resid
            original = ts
        
        return {
            'original': original,
            'trend': trend,
            'seasonal': seasonal,
            'resid': resid
        } 