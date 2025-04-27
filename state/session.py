"""
Модуль для управления состоянием приложения
"""
import streamlit as st
from typing import Any, Dict, Optional

class SessionState:
    """Класс для управления состоянием сессии"""
    
    def __init__(self):
        """Инициализация состояния сессии"""
        self._init_default_state()
    
    def _init_default_state(self):
        """Инициализация значений по умолчанию"""
        defaults = {
            'step': 1,
            'raw_df': None,
            'processed_df': None,
            'filtered_df': None,
            'original_columns': [],
            'current_columns': [],
            'temp_columns': [],
            'file_uploaded': False,
            'date_col': None,
            'target_col': None,
            'current_file': None,
            'preprocessing_history': [],
            'original_missing': None,
            'seasonal_period': None,
            'filled_df': None,
            'initial_transformation_state': None,
            'initial_preprocessing_state': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получение значения из состояния"""
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Установка значения в состояние"""
        st.session_state[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Обновление нескольких значений в состоянии"""
        for key, value in updates.items():
            st.session_state[key] = value
    
    def reset(self, key: Optional[str] = None) -> None:
        """Сброс состояния"""
        if key:
            if key in st.session_state:
                del st.session_state[key]
        else:
            st.session_state.clear()
            self._init_default_state()
    
    def save_state(self, key: str) -> None:
        """Сохранение текущего состояния"""
        backup_key = f"{key}_backup"
        st.session_state[backup_key] = st.session_state[key].copy()
    
    def restore_state(self, key: str) -> None:
        """Восстановление состояния из резервной копии"""
        backup_key = f"{key}_backup"
        if backup_key in st.session_state:
            st.session_state[key] = st.session_state[backup_key].copy()
            del st.session_state[backup_key]

# Создаем глобальный экземпляр состояния
state = SessionState() 