"""
Основной файл приложения
"""
import streamlit as st
from config import APP_CONFIG
from state.session import state
from components.navigation import navigation_buttons, sidebar_navigation
from components.auth import show_auth_form, show_logout_button, is_authenticated
from steps import step_upload, step_preprocessing, step_transformation, step_forecasting
from components.forecasting.model_registration import register_models
import uuid

def init_app():
    """Инициализация приложения"""
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon=APP_CONFIG['icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state=APP_CONFIG['initial_sidebar_state']
    )
    # Register models
    register_models()

def main():
    """Основная функция приложения"""
    # Генерируем session_id, если его нет
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    
    # Сброс состояния только при первом заходе (нет шага и пользователь не аутентифицирован)
    if not is_authenticated() and "just_reset" not in st.session_state:
        state.reset()
        st.session_state["just_reset"] = True
    
    # Initialize app
    init_app()
    
    # Show auth form in sidebar if not authenticated
    with st.sidebar:
        if not is_authenticated():
            with st.expander("🔐 Аутентификация", expanded=False):
                show_auth_form()
        else:
            show_logout_button()
    
    # Восстанавливаем состояние из Supabase только для аутентифицированных пользователей
    if is_authenticated():
        state.restore_from_supabase()
    
    sidebar_navigation()
    st.title("📊 Анализ временных рядов")
    navigation_buttons()
    
    if state.get('step') == 1:
        st.subheader("Шаг 1. Загрузка данных")
        step_upload.run_step()
    elif state.get('step') == 2:
        step_preprocessing.run_step()
    elif state.get('step') == 3:
        step_transformation.run_step()
    elif state.get('step') == 4:
        step_forecasting.run_step()
    
    # Сохраняем состояние в Supabase только для аутентифицированных пользователей
    if is_authenticated():
        state.save_to_supabase()

if __name__ == "__main__":
    main()