import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def load_data(uploaded_file):
    """Загрузка и инициализация данных"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Сбрасываем выбор только если файл действительно новый
    if (not st.session_state.current_file or 
        uploaded_file.file_id != st.session_state.current_file.file_id):
        date_col = None
        target_col = None
    else:
        date_col = st.session_state.date_col
        target_col = st.session_state.target_col

    
    st.session_state.update({
        'raw_df': df.copy(),
        'processed_df': df.copy(),
        'original_columns': df.columns.tolist(),
        'current_columns': df.columns.tolist().copy(),
        'temp_columns': df.columns.tolist().copy(),
        'current_file': uploaded_file,
        'date_col': date_col,
        'target_col': target_col
    })

def show_data_preview():
    """Отображение предпросмотра данных"""
    st.subheader("📝 Предпросмотр данных")
    
    # Создаем копию DataFrame с актуальными названиями колонок
    preview_df = st.session_state.processed_df.copy()[:1000]
    preview_df.columns = st.session_state.current_columns
    
    st.dataframe(
        preview_df,
        use_container_width=True
    )

def show_select_cols_tab():
    """Вкладка выбора даты и зависимой переменной с визуализацией"""
    st.subheader("📌 Выбор переменных")
    if st.session_state.processed_df is not None:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Всегда использовать актуальные значения из session_state
            current_date_col = st.session_state.date_col or st.session_state.current_columns[0]
            current_target_col = st.session_state.target_col or st.session_state.current_columns[-1]

            # Выбор столбца с датой
            new_date_col = st.selectbox(
                "Выберите столбец с датой",
                options=st.session_state.current_columns,
                index=st.session_state.current_columns.index(current_date_col),
                key="date_col_selector"
            )
            
            # Выбор целевой переменной
            new_target_col = st.selectbox(
                "Выберите столбец с зависимой переменной",
                options=[c for c in st.session_state.current_columns if c != new_date_col],
                index=0,
                key="target_col_selector"
            )
            
            # Немедленное обновление состояния
            if new_date_col != st.session_state.date_col:
                st.session_state.date_col = new_date_col
                st.session_state.original_missing = None  # Сбрасываем при изменении даты
                # Сброс целевой переменной при изменении даты
                if new_target_col == new_date_col:
                    st.session_state.target_col = None
                    st.session_state.original_missing = st.session_state.processed_df[new_target_col].isnull().to_numpy().copy()
            
            if new_target_col != st.session_state.target_col:
                st.session_state.target_col = new_target_col
                st.session_state.original_missing = st.session_state.processed_df[new_target_col].isnull().to_numpy().copy()
                
        with col2:
            # Проверяем что оба столбца выбраны
            if st.session_state.date_col and st.session_state.target_col:
                try:
                    # Создаем копию данных для безопасности
                    plot_df = st.session_state.processed_df.copy()
                    
                    # Преобразуем дату
                    plot_df[st.session_state.date_col] = pd.to_datetime(
                        plot_df[st.session_state.date_col]
                       , dayfirst=True
                       , format='mixed'
                    )
                    
                    # Сортируем по дате
                    plot_df = plot_df.sort_values(st.session_state.date_col)
                    
                    fig = px.line(
                        plot_df,
                        x=st.session_state.date_col,
                        y=st.session_state.target_col,
                        title=f"Динамика {st.session_state.target_col}",
                        labels={
                            st.session_state.date_col: "Дата",
                            st.session_state.target_col: "Значение"
                        }
                    )
                    
                    fig.update_layout(
                        hovermode="x unified",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"🚨 Ошибка построения графика: {str(e)}")
            else:
                st.info("👉 Выберите оба столбца для отображения графика")
    else:
        st.warning("Данные не загружены")

def manual_rename_interface():
    """Интерфейс для ручного переименования столбцов"""
    st.subheader("✏️ Редактор названий столбцов")
    
    # Инициализация временных значений
    if not st.session_state.temp_columns:
        st.session_state.temp_columns = st.session_state.current_columns.copy()

    # Заголовки столбцов
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Исходное название**")
    with col2:
        st.markdown("**Новое название**")

    # Создаем строки для каждой пары названий
    temp_names = []
    for i, orig_col in enumerate(st.session_state.current_columns):
        row_cols = st.columns(2)
        with row_cols[0]:
            st.code(orig_col)
        with row_cols[1]:
            new_name = st.text_input(
                label=f"Редактирование {orig_col}",
                value=st.session_state.temp_columns[i],
                key=f"col_rename_{i}",
                label_visibility="collapsed"
            )
            temp_names.append(new_name.strip())
    
    st.session_state.temp_columns = temp_names

    # Кнопки управления
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        reset_btn = st.button("🔄 Сбросить к исходным", use_container_width=True)

    with btn_col3:
        apply_btn = st.button("✅ Применить все изменения", use_container_width=True)

    # Обработка применения изменений
    if apply_btn:
        handle_apply_changes()

    # Обработка сброса
    if reset_btn:
        handle_reset_columns()
        # Показываем сохраненные сообщения
        
    if 'rename_messages' in st.session_state:
        msg = st.session_state.rename_messages
        if msg['type'] == 'error':
            for m in msg['content']:
                st.error(m)
        elif msg['type'] == 'success':
            st.success(msg['content'])
        elif msg['type'] == 'info':
            st.info(msg['content'])
        
        # Удаляем сообщение после показа
        del st.session_state.rename_messages

def handle_apply_changes():
    """Обработчик применения изменений"""
    error_messages = []
    new_columns = st.session_state.temp_columns
    
    # Проверки
    if any(name == "" for name in new_columns):
        error_messages.append("🚫 Названия не могут быть пустыми!")
    if len(set(new_columns)) != len(new_columns):
        error_messages.append("🚫 Названия должны быть уникальными!")

    # Сохраняем сообщения в session_state
    if error_messages:
        st.session_state.rename_messages = {'type': 'error', 'content': error_messages}
    else:
        if new_columns == st.session_state.current_columns:
            st.session_state.rename_messages = {'type': 'info', 'content': "ℹ️ Нет новых изменений для применения"}
        else:
            st.session_state.current_columns = new_columns
            st.session_state.processed_df.columns = new_columns
            st.session_state.rename_messages = {'type': 'success', 'content': "✅ Изменения успешно применены!"}
        
        st.session_state.temp_columns = new_columns.copy()
    
    # Принудительное обновление с сохранением сообщений
    st.rerun()


def handle_reset_columns():
    """Обработчик сброса настроек"""
    st.session_state.temp_columns = st.session_state.original_columns.copy()
    st.session_state.current_columns = st.session_state.original_columns.copy()
    st.session_state.processed_df.columns = st.session_state.original_columns
    st.success("🔄 Названия сброшены к исходным!")



def run_step():
    # Используем сохраненный файл если есть
    uploaded_file = st.file_uploader(
        "Загрузите файл данных (CSV/Excel)", 
        type=["csv", "xlsx"],
        key="file_uploader"
    ) or st.session_state.get('current_file') 
    
    if uploaded_file:
        try:
            # Загружаем данные только если они еще не загружены или файл изменен
            if (st.session_state.raw_df is None or 
                uploaded_file.file_id != st.session_state.current_file.file_id):
                load_data(uploaded_file)
                st.rerun()
            
            # Основные вкладки
            tab1, tab2, tab3 = st.tabs(["Данные", "Переименование", "Выбор переменных"])
            
            with tab1:
                show_data_preview()
            
            with tab2:
                manual_rename_interface()
        
            with tab3:
                show_select_cols_tab()

        except Exception as e:
            st.error(f"🚨 Ошибка обработки файла: {str(e)}")
    else:
        if st.session_state.raw_df is not None:
            st.session_state.raw_df = None
            st.session_state.processed_df = None
            st.session_state.current_file = None
            st.rerun()
        
        st.info("👆 Пожалуйста, загрузите файл для начала работы")