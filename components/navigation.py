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
        
        elif state.get('step', 1) == 2:
            # Проверяем наличие пропущенных значений в целевой переменной и дате
            if state.get('filtered_df') is not None:
                has_missing_target = state.get('filtered_df')[state.get('target_col')].isnull().any()
                has_missing_date = state.get('filtered_df')[state.get('date_col')].isnull().any()
                
                # Проверяем наличие пропусков во временных метках
                df = state.get('filtered_df').copy()
                df = df.sort_values(state.get('date_col'))
                date_diff = df[state.get('date_col')].diff()
                has_time_gaps = (date_diff > pd.Timedelta(days=1)).any()
                
                if has_missing_target or has_missing_date:
                    help_msg = "Устраните пропущенные значения в целевой переменной и дате перед переходом к следующему шагу"
                    st.button("Далее →", disabled=True, help=help_msg, use_container_width=True)
                elif has_time_gaps:
                    help_msg = "Обнаружены пропуски во временных метках. Устраните пропуски в датах перед переходом к следующему шагу"
                    st.button("Далее →", disabled=True, help=help_msg, use_container_width=True)
                else:
                    if st.button("Далее →", type="primary", use_container_width=True):
                        state.set('step', state.get('step', 1) + 1)
                        st.rerun()
            else:
                st.button("Далее →", disabled=True, help="Данные не загружены", use_container_width=True)
        
        elif state.get('step', 1) == 3:
            if st.button("Далее →", type="primary", use_container_width=True):
                state.set('step', state.get('step', 1) + 1)
                st.rerun()
        
        elif state.get('step', 1) == 4:
            st.button("Далее →", disabled=True, help="Это последний шаг", use_container_width=True)

def sidebar_navigation():
    """Навигация в сайдбаре"""
    with st.sidebar:
        st.title("📌 Навигация")
        steps = {
            1: "Загрузка данных",
            2: "Предобработка данных",
            3: "Преобразование данных и выбросы",
            4: "Прогнозирование"
        }
        current_step = state.get('step', 1)
        allowed_steps = [1]
        
        # Проверка готовности данных для активации шагов
        if state.get('raw_df') is not None and state.get('date_col') and state.get('target_col'):
            allowed_steps.extend([2, 3, 4])
            
        # Создаем контейнер для кнопок
        button_container = st.container()
        
        for step_num, step_name in steps.items():
            status = "✅" if step_num < current_step else "➖"
            if step_num == current_step:
                status = "📍"
            disabled = step_num not in allowed_steps
            
            with button_container:
                # Создаем кнопку без сохранения состояния
                if st.button(
                    f"{status} {step_name}",
                    disabled=disabled,
                    use_container_width=True
                ):
                    if not disabled and step_num != current_step:
                        state.set('step', step_num)
                        st.rerun()
                        
        # UX: Подсказка, если шаги неактивны
        if state.get('raw_df') is None:
            st.info("Загрузите данные, чтобы перейти к следующим шагам") 