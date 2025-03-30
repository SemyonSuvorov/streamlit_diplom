# app.py
from steps import step_upload, step_preprocessing
import streamlit as st

def navigation_buttons():
    """Кнопки навигации с обработкой фильтрации данных"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.session_state.step > 1:
            if st.button("← Назад", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.step == 1:
            valid_selection = (
                st.session_state.date_col and 
                st.session_state.target_col and 
                st.session_state.date_col != st.session_state.target_col
            )
            allow_next = valid_selection and st.session_state.raw_df is not None

            if allow_next:
                if st.button("Далее →", type="primary", use_container_width=True):
                    # Фильтрация данных перед переходом
                    try:
                        st.session_state.filtered_df = st.session_state.processed_df[
                            [st.session_state.date_col, st.session_state.target_col]
                        ].copy()
                        st.session_state.step += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка фильтрации данных: {str(e)}")
            else:
                help_msg = ("Выберите разные столбцы даты и целевой переменной" 
                           if st.session_state.date_col == st.session_state.target_col 
                           else "Загрузите данные и выберите столбцы")
                st.button(
                    "Далее →", 
                    disabled=True, 
                    help=help_msg,
                    use_container_width=True
                )

def init_session_state():
    """Инициализация session state"""
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
        'current_file': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_navigation():
    """Навигация в сайдбаре"""
    with st.sidebar:
        st.title("📌 Навигация")
        
        steps = {
            1: "Загрузка данных",
            2: "Предобработка данных"
        }
        
        current_step = st.session_state.step
        allowed_steps = [1]
        
        if st.session_state.raw_df is not None and \
           st.session_state.date_col and \
           st.session_state.target_col:
            allowed_steps.append(2)
        
        for step_num, step_name in steps.items():
            status = "✅" if step_num < current_step else "➖"
            if step_num == current_step:
                status = "📍"
                
            disabled = step_num not in allowed_steps
            btn = st.button(
                f"{status} {step_name}",
                key=f"sidebar_step_{step_num}",
                disabled=disabled,
                use_container_width=True
            )
            
            if btn and not disabled and step_num != current_step:
                st.session_state.step = step_num
                st.rerun()

def main():
    st.set_page_config(
        page_title="Time-series analysis", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_session_state()
    
    sidebar_navigation()
    
    st.title("📊 Анализ временных рядов")
    navigation_buttons()
    
    if st.session_state.step == 1:
        st.subheader("Шаг 1. Загрузка данных")
        step_upload.run_step()
    
    elif st.session_state.step == 2:
        step_preprocessing.run_step()

if __name__ == "__main__":
    main()