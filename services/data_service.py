"""
Сервисный слой для работы с данными
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from config import DATA_CONFIG

class DataService:
    """Класс для работы с данными"""
    
    @staticmethod
    def load_data(file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Загрузка данных из файла
        
        Args:
            file: Загруженный файл
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame с данными и метаданные
        """
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            metadata = {
                'columns': df.columns.tolist(),
                'rows': len(df),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            return df, metadata
        except Exception as e:
            raise ValueError(f"Ошибка загрузки данных: {str(e)}")
    
    @staticmethod
    def validate_data(df: pd.DataFrame, date_col: str, target_col: str) -> Tuple[bool, str]:
        """
        Валидация данных
        
        Args:
            df: DataFrame с данными
            date_col: Название столбца с датой
            target_col: Название столбца с целевой переменной
            
        Returns:
            Tuple[bool, str]: Результат валидации и сообщение об ошибке
        """
        if date_col not in df.columns:
            return False, f"Столбец с датой '{date_col}' не найден"
        
        if target_col not in df.columns:
            return False, f"Столбец с целевой переменной '{target_col}' не найден"
        
        if date_col == target_col:
            return False, "Столбцы даты и целевой переменной должны быть разными"
        
        try:
            pd.to_datetime(df[date_col])
        except:
            return False, f"Столбец '{date_col}' не содержит корректные даты"
        
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return False, f"Столбец '{target_col}' должен содержать числовые значения"
        
        return True, ""
    
    @staticmethod
    def prepare_time_series(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """
        Подготовка временного ряда
        
        Args:
            df: DataFrame с данными
            date_col: Название столбца с датой
            target_col: Название столбца с целевой переменной
            
        Returns:
            pd.DataFrame: Подготовленный DataFrame
        """
        prepared_df = df.copy()
        prepared_df[date_col] = pd.to_datetime(prepared_df[date_col])
        prepared_df = prepared_df.sort_values(date_col)
        prepared_df = prepared_df[[date_col, target_col]]
        return prepared_df
    
    @staticmethod
    def get_data_statistics(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Получение статистики по данным
        
        Args:
            df: DataFrame с данными
            target_col: Название столбца с целевой переменной
            
        Returns:
            Dict[str, Any]: Статистика по данным
        """
        stats = df[target_col].agg([
            'mean', 'median', 'std', 'min', 'max', 'skew'
        ]).to_dict()
        
        stats['missing'] = df[target_col].isnull().sum()
        stats['duplicates'] = df.duplicated().sum()
        
        return stats
    
    @staticmethod
    def get_time_series_info(df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Получение информации о временном ряде
        
        Args:
            df: DataFrame с данными
            date_col: Название столбца с датой
            
        Returns:
            Dict[str, Any]: Информация о временном ряде
        """
        dates = pd.to_datetime(df[date_col])
        freq = pd.infer_freq(dates)
        
        info = {
            'start_date': dates.min(),
            'end_date': dates.max(),
            'period': (dates.max() - dates.min()).days,
            'frequency': freq,
            'missing_dates': None
        }
        
        try:
            full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
            missing = full_range.difference(dates)
            info['missing_dates'] = len(missing)
        except:
            pass
        
        return info

    @staticmethod
    def clean_percentage_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка процентных значений в DataFrame
        
        Args:
            df: DataFrame с данными
            
        Returns:
            pd.DataFrame: DataFrame с очищенными значениями
        """
        cleaned_df = df.copy()
        
        # Проходим по всем колонкам
        for col in cleaned_df.columns:
            # Проверяем, содержит ли колонка строковые значения
            if cleaned_df[col].dtype == 'object':
                # Пробуем преобразовать значения в числа, удаляя знак процента
                try:
                    cleaned_df[col] = cleaned_df[col].str.replace('%', '').astype(float) / 100
                except:
                    # Если не получилось преобразовать, оставляем как есть
                    continue
        
        return cleaned_df 