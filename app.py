from steps import step_upload
import streamlit as st

# Конфигурация модели
# MODEL_NAME = "cointegrated/rut5-base"
# CACHE_DIR = "models"

def navigation_buttons():
    """Функция для отображения кнопок навигации в верхней части страницы"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        inner_col1, inner_col2 = st.columns([1, 2])
        with inner_col1:
            if st.session_state.step > 1:
                if st.button("←", type="secondary", use_container_width=True):
                    st.session_state.step -= 1
                    st.rerun()
    
    with col3:
        allow_next = False
        if st.session_state.step == 1:
            # Проверяем актуальный статус загрузки файла
            allow_next = st.session_state.date_col and st.session_state.target_col

        inner_col1, inner_col2 = st.columns([2, 1])
        with inner_col2:
            if st.session_state.step == 1:
                if allow_next:
                    if st.button("→", type="primary", use_container_width=True):
                        st.session_state.step += 1
                        st.rerun()
                else:
                    st.button(
                        "→", 
                        disabled=True, 
                        help="Сначала загрузите файл и выберите столбец с датой и зависимой переменной",
                        use_container_width=True
                    )
            
            elif st.session_state.step == 2:
                if allow_next:
                    if st.button("→", type="primary", use_container_width=True):
                        st.success("Переход к анализу данных")
                        # Логика перехода
                else:
                    st.button(
                        "→", 
                        disabled=True, 
                        help="Выберите обязательные столбцы",
                        use_container_width=True
                    )


def init_session_state():
    """Инициализация всех необходимых переменных в session state"""
    session_vars = {
        'step': 1,
        'raw_df': None,
        'processed_df': None,
        'original_columns': [],
        'current_columns': [],
        'temp_columns': [],
        'file_uploaded': False,
        'date_col': None,
        'target_col': None,
        'current_file': None  
    }
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default


def main():
    st.set_page_config(page_title="Time-series analysis", page_icon="📊", layout="wide")
    init_session_state()
    
    st.title("📊 Анализ временных рядов")
    navigation_buttons()
    
    if st.session_state.step == 1:
        st.subheader("Шаг 1. Загрузка данных")
        step_upload.run_step()
    
    elif st.session_state.step == 2:
        st.session_state.date_col

   


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