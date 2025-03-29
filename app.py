import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import hashlib

# Конфигурация модели
# MODEL_NAME = "cointegrated/rut5-base"
# CACHE_DIR = "models"

def init_session_state():
    """Инициализация всех необходимых переменных в session state"""
    session_vars = {
        'raw_df': None,
        'processed_df': None,
        'original_columns': [],
        'current_columns': [],
        'temp_columns': [],
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

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
    for i, (orig_col, current_col) in enumerate(zip(
        st.session_state.original_columns,
        st.session_state.current_columns
    )):
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

def main():
    st.set_page_config(page_title="Data Assistant", page_icon="📊", layout="wide")
    init_session_state()
    
    st.title("📊 Ассистент для работы с данными")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите файл данных (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if st.session_state.raw_df is None:
                load_data(uploaded_file)
            
            # Основные вкладки
            tab1, tab2, tab3 = st.tabs(["Данные", "Переименование", "Анализ"])
            
            with tab1:
                show_data_preview()
            
            with tab2:
                manual_rename_interface()
        
            with tab3:
                #show_analysis_tab()
                date_col = st.selectbox(
                    "Выберите столбец с датой",
                    options=st.session_state.current_columns
                )


        except Exception as e:
            st.error(f"🚨 Ошибка обработки файла: {str(e)}")
    else:
        st.info("👆 Пожалуйста, загрузите файл для начала работы")

def load_data(uploaded_file):
    """Загрузка и инициализация данных"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.session_state.update({
        'raw_df': df.copy(),
        'processed_df': df.copy(),
        'original_columns': df.columns.tolist(),
        'current_columns': df.columns.tolist().copy(),
        'temp_columns': df.columns.tolist().copy(),
    })

def show_data_preview():
    """Отображение предпросмотра данных"""
    st.subheader("Просмотр данных")
    
    # Создаем копию DataFrame с актуальными названиями колонок
    preview_df = st.session_state.processed_df.copy()
    preview_df.columns = st.session_state.current_columns
    
    st.dataframe(
        preview_df,
        use_container_width=True
    )

def show_analysis_tab():
    """Вкладка анализа данных"""
    st.subheader("Анализ данных")
    if st.session_state.processed_df is not None:
        date_col = st.selectbox(
            "Выберите столбец с датой",
            options=st.session_state.current_columns
        )
        
        try:
            st.session_state.processed_df[date_col] = pd.to_datetime(
                st.session_state.processed_df[date_col]
            )
            st.line_chart(st.session_state.processed_df.set_index(date_col))
        except Exception as e:
            st.error(f"⏰ Ошибка преобразования даты: {str(e)}")


if __name__ == "__main__":
    main()

# # Загрузка модели
# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
#     model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
#     return tokenizer, model

# # Генерация новых названий
# def generate_new_names(columns, context=""):
#     try:
#         tokenizer, model = load_model()
#         prompt = f"""
#         Переименуй названия колонок на русском языке в понятные, сохранив порядок.
#         Контекст: {context}. Исходные названия: {', '.join(columns)}.
#         Ответ должен содержать только новые названия через запятую.
#         Новые названия:"""
        
#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             max_length=512,
#             truncation=True,
#             padding="max_length"
#         )
        
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=50,
#             num_beams=5,
#             early_stopping=True
#         )
        
#         new_names = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         new_names = re.sub(r"[^а-яА-Я0-9,\-_\s]", "", new_names)
#         new_names = [name.strip().replace(' ', '_') for name in new_names.split(",")]
        
#         if len(new_names) != len(columns):
#             return columns
#         return new_names
    
#     except Exception as e:
#         st.error(f"Ошибка генерации: {str(e)}")
#         return columns