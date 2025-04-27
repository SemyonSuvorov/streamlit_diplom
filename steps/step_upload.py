# steps/step_upload.py
import streamlit as st
import pandas as pd
from state.session import state
from services.data_service import DataService
from components.charts import create_time_series_plot

def load_data(uploaded_file):
    """Загрузка данных из файла"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Compare files based on their content hash instead of file_id
    current_file = state.get('current_file')
    is_new_file = (
        current_file is None or 
        (isinstance(current_file, dict) and current_file.get('name') != uploaded_file.name) or
        (not isinstance(current_file, dict) and current_file.name != uploaded_file.name)
    )
    
    if is_new_file:
        date_col = None
        target_col = None
    else:
        date_col = state.get('date_col')
        target_col = state.get('target_col')

    state.update({
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
    preview_df = state.get('processed_df').copy()[:1000]
    preview_df.columns = state.get('current_columns')
    st.dataframe(preview_df, use_container_width=True)

def show_select_cols_tab():
    """Отображение вкладки выбора переменных"""
    st.subheader("📌 Выбор переменных")
    if state.get('processed_df') is not None:
        # Валидация текущих значений
        current_columns = state.get('current_columns')
        
        # Проверка и коррекция date_col
        if state.get('date_col') not in current_columns:
            state.set('date_col', current_columns[0] if current_columns else None)
        
        # Проверка и коррекция target_col
        available_targets = [c for c in current_columns if c != state.get('date_col')]
        if state.get('target_col') not in available_targets:
            state.set('target_col', available_targets[0] if available_targets else None)

        col1, col2 = st.columns([1, 3])
        with col1:
            new_date_col = st.selectbox(
                "Выберите столбец с датой",
                options=current_columns,
                index=current_columns.index(state.get('date_col')) if state.get('date_col') in current_columns else 0,
                key="date_col_selector"
            )
            
            available_targets = [c for c in current_columns if c != new_date_col]
            current_target = state.get('target_col')
            target_index = 0
            if current_target in available_targets:
                target_index = available_targets.index(current_target)
            
            new_target_col = st.selectbox(
                "Выберите столбец с зависимой переменной",
                options=available_targets,
                index=target_index,
                key="target_col_selector"
            )
            
            # Обновляем значения только при изменении
            if new_date_col != state.get('date_col'):
                state.set('date_col', new_date_col)
                state.set('target_col', None)  # Сбрасываем целевую при смене даты
            
            if new_target_col != state.get('target_col'):
                state.set('target_col', new_target_col)
        with col2:
            if state.get('date_col') and state.get('target_col'):
                try:
                    plot_df = state.get('processed_df').copy()
                    plot_df[state.get('date_col')] = pd.to_datetime(plot_df[state.get('date_col')], dayfirst=True, format='mixed')
                    plot_df = plot_df.sort_values(state.get('date_col'))
                    fig = create_time_series_plot(
                        plot_df,
                        x_col=state.get('date_col'),
                        y_col=state.get('target_col'),
                        title=f"Динамика {state.get('target_col')}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"🚨 Ошибка построения графика: {str(e)}")
            else:
                st.info("👉 Выберите оба столбца для отображения графика")

def manual_rename_interface():
    """Интерфейс ручного переименования столбцов"""
    st.subheader("✏️ Редактор названий столбцов")
    if not state.get('temp_columns'):
        state.set('temp_columns', state.get('current_columns').copy())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Исходное название**")
    with col2:
        st.markdown("**Новое название**")

    temp_names = []
    for i, orig_col in enumerate(state.get('current_columns')):
        row_cols = st.columns(2)
        with row_cols[0]:
            st.code(orig_col)
        with row_cols[1]:
            # Get the current value from temp_columns
            current_value = state.get('temp_columns')[i]
            # Create the text input without setting value directly
            new_name = st.text_input(
                label=f"Редактирование {orig_col}",
                key=f"col_rename_{i}",
                label_visibility="collapsed"
            )
            # If the input is empty, use the current value
            temp_names.append(new_name.strip() if new_name.strip() else current_value)
    
    state.set('temp_columns', temp_names)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        reset_btn = st.button("🔄 Сбросить к исходным", use_container_width=True)
    with btn_col3:
        apply_btn = st.button("✅ Применить все изменения", use_container_width=True)

    if apply_btn:
        error_messages = []
        new_columns = state.get('temp_columns')
        if any(name == "" for name in new_columns):
            error_messages.append("🚫 Названия не могут быть пустыми!")
        if len(set(new_columns)) != len(new_columns):
            error_messages.append("🚫 Названия должны быть уникальными!")

        if error_messages:
            st.error("\n".join(error_messages))
        else:
            # Обновляем названия для date_col и target_col
            rename_mapping = dict(zip(state.get('original_columns'), new_columns))
            if state.get('date_col') in rename_mapping:
                state.set('date_col', rename_mapping[state.get('date_col')])
            if state.get('target_col') in rename_mapping:
                state.set('target_col', rename_mapping[state.get('target_col')])
            
            state.set('current_columns', new_columns)
            state.get('processed_df').columns = new_columns
            st.success("✅ Изменения успешно применены!")
            st.rerun()
            
    if reset_btn:
        state.set('temp_columns', state.get('original_columns').copy())
        state.set('current_columns', state.get('original_columns').copy())
        state.get('processed_df').columns = state.get('original_columns')
        st.success("🔄 Названия сброшены к исходным!")
        st.rerun()

def run_step():
    """Запуск шага загрузки данных"""
    uploaded_file = st.file_uploader(
        "Загрузите файл данных (CSV/Excel)", 
        type=["csv", "xlsx"]
    )

    # Отладочная информация
    if uploaded_file:
        st.write(f"Загружен файл: {uploaded_file.name}")

    tab1, tab2, tab3 = st.tabs(["Данные", "Переименование", "Выбор переменных"])
    
    with tab1:
        if uploaded_file:
            try:
                # Проверяем, нужно ли загружать новый файл
                current_file = state.get('current_file')
                should_load = (
                    state.get('raw_df') is None or 
                    current_file is None or
                    (isinstance(current_file, dict) and current_file.get('name') != uploaded_file.name) or
                    (not isinstance(current_file, dict) and current_file.name != uploaded_file.name)
                )
                
                if should_load:
                    load_data(uploaded_file)
                    st.rerun()
                show_data_preview()
            except Exception as e:
                st.error(f"🚨 Ошибка обработки файла: {str(e)}")
                st.write("Детали ошибки:", str(e))
        else:
            st.info("👆 Загрузите файл для предпросмотра данных")
    with tab2:
        if uploaded_file:
            manual_rename_interface()
        else:
            st.info("✏️ Загрузите файл для редактирования названий столбцов")
    with tab3:
        if uploaded_file:
            show_select_cols_tab()
        else:
            st.info("📌 Загрузите файл для выбора переменных")