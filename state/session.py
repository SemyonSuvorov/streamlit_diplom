"""
Модуль для управления состоянием приложения
"""
import streamlit as st
from typing import Any, Dict, Optional
import json
from supabase import create_client, Client
import pandas as pd

# Подключение к Supabase через secrets
SUPABASE_URL = st.secrets.supabase.url
SUPABASE_KEY = st.secrets.supabase.anon_key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class SessionState:
    """Класс для управления состоянием сессии"""
    
    def __init__(self):
        """Инициализация состояния сессии"""
        self._init_default_state()
    
    def _init_default_state(self):
        """Инициализация значений по умолчанию (только если их нет)"""
        defaults = {
            'step': 1,  # Всегда начинаем с шага 1
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
        # Принудительно устанавливаем step в 1, если нет данных
        raw_df = st.session_state.get('raw_df')
        if raw_df is None or (isinstance(raw_df, pd.DataFrame) and raw_df.empty):
            st.session_state['step'] = 1
            
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
        """Обновление нескольких значений состояния"""
        for key, value in updates.items():
            st.session_state[key] = value
        self.save_to_supabase()
    
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

    def _is_json_serializable(self, v):
        """Проверка, можно ли сериализовать значение в JSON"""
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    def save_to_supabase(self):
        """Сохранение состояния в Supabase"""
        if 'user' not in st.session_state:
            return  # Не сохраняем для неаутентифицированных пользователей
            
        try:
            # Преобразуем DataFrame в JSON-совместимый формат
            state_data = {}
            for key, value in st.session_state.items():
                if self._is_json_serializable(value):
                    if isinstance(value, pd.DataFrame):
                        state_data[key] = value.to_json(orient='split')
                    else:
                        state_data[key] = value
            
            # Сохраняем в Supabase
            supabase.table('user_states').upsert({
                'user_id': st.session_state['user'].id,
                'state_data': state_data
            }).execute()
        except Exception as e:
            print(f"Ошибка сохранения состояния в Supabase: {e}")

    def restore_from_supabase(self):
        """Восстановление состояния из Supabase"""
        if 'user' not in st.session_state:
            return  # Не восстанавливаем для неаутентифицированных пользователей
            
        try:
            # Получаем состояние из Supabase
            response = supabase.table('user_states').select('state_data').eq('user_id', st.session_state['user'].id).execute()
            
            if response.data:
                state_data = response.data[0]['state_data']
                for key, value in state_data.items():
                    if isinstance(value, str) and '{"columns"' in value:  # Проверяем, является ли это DataFrame
                        st.session_state[key] = pd.read_json(value, orient='split')
                    else:
                        st.session_state[key] = value
        except Exception as e:
            print(f"Ошибка восстановления состояния из Supabase: {e}")

# Создаем глобальный экземпляр состояния
state = SessionState() 