"""
Компоненты навигации
"""
import streamlit as st
from state.session import state
import pandas as pd

def navigation_buttons():
    """Кнопки навигации"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if state.get('step', 1) > 1:
            if st.button("← Назад", use_container_width=True):
                state.set('step', state.get('step', 1) - 1)
                st.rerun()
    
    with col3:
        if state.get('step', 1) == 1:
            valid_selection = (
                state.get('date_col') and 
                state.get('target_col') and 
                state.get('date_col') != state.get('target_col')
            )
            allow_next = valid_selection and state.get('raw_df') is not None

            if allow_next:
                if st.button("Далее →", type="primary", use_container_width=True):
                    try:
                        # Сначала создаем filtered_df
                        filtered_df = state.get('processed_df')[
                            [state.get('date_col'), state.get('target_col')]
                        ].copy()
                        
                        # Затем обновляем состояние
                        state.update({
                            'filtered_df': filtered_df,
                            'original_filtered_df': filtered_df.copy(),
                            'original_missing': state.get('processed_df')[
                                state.get('target_col')
                            ].isnull().copy()
                        })
                        
                        # Преобразуем даты
                        state.get('filtered_df')[state.get('date_col')] = pd.to_datetime(
                            state.get('filtered_df')[state.get('date_col')]
                        )
                        
                        # Проверяем и сбрасываем индекс если нужно
                        if state.get('date_col') not in state.get('filtered_df').columns:
                            state.set('filtered_df', state.get('filtered_df').reset_index())
                            
                        state.set('step', state.get('step', 1) + 1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка фильтрации данных: {str(e)}")
            else:
                help_msg = ("Выберите разные столбцы даты и целевой переменной" 
                           if state.get('date_col') == state.get('target_col') 
                           else "Загрузите данные и выберите столбцы")
                st.button("Далее →", disabled=True, help=help_msg, use_container_width=True)
        
        elif state.get('step', 1) in [2, 3]:
            if st.button("Далее →", type="primary", use_container_width=True):
                state.set('step', state.get('step', 1) + 1)
                st.rerun()

def sidebar_navigation():
    """Навигация в сайдбаре"""
    with st.sidebar:
        st.title("📌 Навигация")
        steps = {
            1: "Загрузка данных",
            2: "Предобработка данных",
            3: "Преобразование данных и выбросы"
        }
        current_step = state.get('step', 1)  # Default to 1 if None
        allowed_steps = [1]
        
        # Проверка готовности данных для активации шагов
        if state.get('raw_df') is not None and state.get('date_col') and state.get('target_col'):
            allowed_steps.extend([2, 3])
        
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
                state.set('step', step_num)
                st.rerun() 