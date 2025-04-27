"""
Основной файл приложения
"""
import streamlit as st
from config import APP_CONFIG
from state.session import state
from components.navigation import navigation_buttons, sidebar_navigation
from steps import step_upload, step_preprocessing, step_transformation

def init_app():
    """Инициализация приложения"""
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon=APP_CONFIG['icon'],
        layout=APP_CONFIG['layout'],
        initial_sidebar_state=APP_CONFIG['initial_sidebar_state']
    )

def main():
    """Основная функция приложения"""
    init_app()
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

if __name__ == "__main__":
    main()