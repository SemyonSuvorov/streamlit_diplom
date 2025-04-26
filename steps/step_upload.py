# steps/step_upload.py
import streamlit as st
import pandas as pd
import plotly.express as px

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
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
    st.subheader("📝 Предпросмотр данных")
    preview_df = st.session_state.processed_df.copy()[:1000]
    preview_df.columns = st.session_state.current_columns
    st.dataframe(preview_df, use_container_width=True)

def show_select_cols_tab():
    st.subheader("📌 Выбор переменных")
    if st.session_state.processed_df is not None:
        # Валидация текущих значений
        current_columns = st.session_state.current_columns
        
        # Проверка и коррекция date_col
        if st.session_state.date_col not in current_columns:
            st.session_state.date_col = current_columns[0] if current_columns else None
        
        # Проверка и коррекция target_col
        available_targets = [c for c in current_columns if c != st.session_state.date_col]
        if st.session_state.target_col not in available_targets:
            st.session_state.target_col = available_targets[0] if available_targets else None

        col1, col2 = st.columns([1, 3])
        with col1:
            new_date_col = st.selectbox(
                "Выберите столбец с датой",
                options=current_columns,
                index=current_columns.index(st.session_state.date_col),
                key="date_col_selector"
            )
            
            available_targets = [c for c in current_columns if c != new_date_col]
            new_target_col = st.selectbox(
                "Выберите столбец с зависимой переменной",
                options=available_targets,
                index=available_targets.index(st.session_state.target_col) if st.session_state.target_col in available_targets else 0,
                key="target_col_selector"
            )
            
            # Обновляем значения только при изменении
            if new_date_col != st.session_state.date_col:
                st.session_state.date_col = new_date_col
                st.session_state.target_col = None  # Сбрасываем целевую при смене даты
            
            if new_target_col != st.session_state.target_col:
                st.session_state.target_col = new_target_col
        with col2:
            if st.session_state.date_col and st.session_state.target_col:
                try:
                    plot_df = st.session_state.processed_df.copy()
                    plot_df[st.session_state.date_col] = pd.to_datetime(plot_df[st.session_state.date_col], dayfirst=True, format='mixed')
                    plot_df = plot_df.sort_values(st.session_state.date_col)
                    fig = px.line(
                        plot_df,
                        x=st.session_state.date_col,
                        y=st.session_state.target_col,
                        title=f"Динамика {st.session_state.target_col}",
                        labels={st.session_state.date_col: "Дата", st.session_state.target_col: "Значение"}
                    )
                    fig.update_layout(hovermode="x unified", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"🚨 Ошибка построения графика: {str(e)}")
            else:
                st.info("👉 Выберите оба столбца для отображения графика")

def manual_rename_interface():
    st.subheader("✏️ Редактор названий столбцов")
    if not st.session_state.temp_columns:
        st.session_state.temp_columns = st.session_state.current_columns.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Исходное название**")
    with col2:
        st.markdown("**Новое название**")

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

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col1:
        reset_btn = st.button("🔄 Сбросить к исходным", use_container_width=True)
    with btn_col3:
        apply_btn = st.button("✅ Применить все изменения", use_container_width=True)

    if apply_btn:
        error_messages = []
        new_columns = st.session_state.temp_columns
        if any(name == "" for name in new_columns):
            error_messages.append("🚫 Названия не могут быть пустыми!")
        if len(set(new_columns)) != len(new_columns):
            error_messages.append("🚫 Названия должны быть уникальными!")

        if error_messages:
            st.error("\n".join(error_messages))
        else:
            # Обновляем названия для date_col и target_col
            rename_mapping = dict(zip(st.session_state.original_columns, new_columns))
            if st.session_state.date_col in rename_mapping:
                st.session_state.date_col = rename_mapping[st.session_state.date_col]
            if st.session_state.target_col in rename_mapping:
                st.session_state.target_col = rename_mapping[st.session_state.target_col]
            
            st.session_state.current_columns = new_columns
            st.session_state.processed_df.columns = new_columns
            st.success("✅ Изменения успешно применены!")
            st.rerun()
            
    if reset_btn:
        st.session_state.temp_columns = st.session_state.original_columns.copy()
        st.session_state.current_columns = st.session_state.original_columns.copy()
        st.session_state.processed_df.columns = st.session_state.original_columns
        st.success("🔄 Названия сброшены к исходным!")
        st.rerun()

def run_step():
    uploaded_file = st.file_uploader(
        "Загрузите файл данных (CSV/Excel)", 
        type=["csv", "xlsx"],
        key="file_uploader"
    ) or st.session_state.get('current_file') 
    
    if uploaded_file:
        try:
            if (st.session_state.raw_df is None or 
                uploaded_file.file_id != st.session_state.current_file.file_id):
                load_data(uploaded_file)
                st.rerun()
            
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
        st.info("👆 Пожалуйста, загрузите файл для начала работы")