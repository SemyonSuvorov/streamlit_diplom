import streamlit as st
from state.session import state

def get_available_weights(model_name):
    # Заглушка: список доступных весов для каждой модели
    weights_dict = {
        "LSTM": ["LSTM_energy_2023.h5", "LSTM_sales_2022.h5"],
        "Prophet": ["Prophet_retail_2021.pkl"],
        "Transformer": ["Transformer_finance_2024.ckpt"],
        "GRU": [],
        "ARIMA": [],
        "Random Forest": []
    }
    return weights_dict.get(model_name, [])

def show_forecasting_step():
    st.subheader("Шаг 4. Прогнозирование временного ряда")

    if state.get('filtered_df') is None or state.get('filtered_df').empty:
        st.warning("Данные не загружены!")
        return

    # 1. Выбор целевой переменной и горизонта прогноза
    numeric_cols = state.get('filtered_df').select_dtypes(include='number').columns.tolist()
    target_col = st.selectbox(
        "Целевая переменная",
        numeric_cols,
        index=numeric_cols.index(state.get('target_col')) if state.get('target_col') in numeric_cols else 0,
        key="forecast_target_col"
    )
    forecast_horizon = st.number_input(
        "Горизонт прогноза (шагов вперед)",
        min_value=1, max_value=365, value=30,
        key="forecast_horizon"
    )

    st.markdown("---")

    # 2. Выбор модели
    model_options = [
        "ARIMA", "Prophet", "Random Forest", "LSTM", "GRU", "Transformer"
    ]
    selected_model = st.selectbox(
        "Модель прогнозирования",
        model_options,
        key="forecast_model"
    )

    # 3. Настройка параметров и выбор весов
    st.subheader("Параметры модели")
    if selected_model in ["LSTM", "GRU", "Transformer", "Prophet"]:
        pretrained_option = st.radio(
            "Использовать предобученные веса?",
            ["Обучить с нуля", "Загрузить предобученные веса"],
            key="forecast_pretrained_option"
        )
        if pretrained_option == "Загрузить предобученные веса":
            available_weights = get_available_weights(selected_model)
            if available_weights:
                selected_weights = st.selectbox(
                    "Выберите предобученные веса",
                    available_weights,
                    key="forecast_selected_weights"
                )
            else:
                st.info("Нет доступных предобученных весов для этой модели.")
            uploaded_weights = st.file_uploader(
                "Или загрузите свой файл весов",
                type=["h5", "pth", "ckpt", "pkl"],
                key="forecast_uploaded_weights"
            )
    else:
        st.caption("Для выбранной модели предобученные веса не поддерживаются.")

    # Пример параметров для разных моделей (заглушки)
    if selected_model == "ARIMA":
        p = st.number_input("AR (p)", 0, 5, 1, key="arima_p")
        d = st.number_input("I (d)", 0, 2, 1, key="arima_d")
        q = st.number_input("MA (q)", 0, 5, 1, key="arima_q")
    elif selected_model == "Prophet":
        seasonality = st.selectbox(
            "Seasonality",
            ["auto", "yearly", "monthly", "weekly"],
            key="prophet_seasonality"
        )
    elif selected_model in ["LSTM", "GRU", "Transformer"]:
        window_size = st.number_input("Размер окна", 1, 60, 14, key="nn_window_size")
        epochs = st.number_input("Эпохи", 1, 200, 20, key="nn_epochs")
        batch_size = st.number_input("Batch size", 8, 512, 32, key="nn_batch_size")

    st.markdown("---")
    st.button("Построить прогноз", type="primary", key="forecast_run")

    # Заглушка для визуализации и метрик
    st.info("Здесь появится график прогноза и метрики после запуска.")

def run_step():
    show_forecasting_step() 