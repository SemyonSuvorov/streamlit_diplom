# steps/step_preprocessing.py
import streamlit as st
import pandas as pd

def run_step():
    st.subheader("Шаг 2. Предобработка данных")
    
    if st.session_state.filtered_df is not None:
        st.write("### Текущие данные (только выбранные столбцы)")
        
        # Показываем отфильтрованные данные
        st.dataframe(
            st.session_state.filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Инструменты предобработки
        st.write("### Инструменты предобработки")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Обработка пропусков**")
            if st.button("Удалить строки с пропусками"):
                st.session_state.filtered_df = st.session_state.filtered_df.dropna()
                st.rerun()
                
            if st.button("Заполнить средним значением"):
                st.session_state.filtered_df = st.session_state.filtered_df.fillna(
                    st.session_state.filtered_df.mean(numeric_only=True)
                )
                st.rerun()
        
        with col2:
            st.write("**Типы данных**")
            if st.button("Преобразовать дату"):
                try:
                    date_col = st.session_state.date_col
                    st.session_state.filtered_df[date_col] = pd.to_datetime(
                        st.session_state.filtered_df[date_col]
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка преобразования даты: {str(e)}")
            
            if st.button("Нормализовать данные"):
                st.session_state.filtered_df = (
                    st.session_state.filtered_df - 
                    st.session_state.filtered_df.mean()
                ) / st.session_state.filtered_df.std()
                st.rerun()
    else:
        st.warning("Пожалуйста, загрузите данные на первом шаге")