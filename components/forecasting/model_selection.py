import streamlit as st
from state.session import state
from components.forecasting.model_registry import ModelType, ModelConfig, ModelRegistry
from components.forecasting.model_weights import get_available_weights

def show_model_selection_tab() -> tuple[ModelConfig, ModelType]:
    """Display model selection and parameter configuration tab"""
    st.markdown("### Выбор модели и параметров")
    
    # 1. Select target variable
    numeric_cols = state.get('filtered_df').select_dtypes(include='number').columns.tolist()
    target_col = st.selectbox(
        "Целевая переменная",
        numeric_cols,
        index=numeric_cols.index(state.get('target_col')) if state.get('target_col') in numeric_cols else 0,
        key="forecast_target_col"
    )

    st.markdown("---")

    # 2. Model selection
    model_options = [model.value for model in ModelType]
    selected_model = st.selectbox(
        "Модель прогнозирования",
        model_options,
        key="forecast_model"
    )

    # 3. Model parameters configuration
    st.markdown("### Параметры модели")
    model_type = ModelType(selected_model)
    default_config = ModelRegistry.get_default_config(model_type)
    
    # Create configuration based on model type
    config = ModelConfig(
        target_col=target_col
    )
    
    # Add model-specific parameters
    if model_type == ModelType.ARIMA:
        col1, col2 = st.columns(2)
        with col1:
            p_values = st.multiselect(
                "AR (p) значения для поиска",
                options=list(range(6)),
                default=default_config.get('p_values', [0, 1, 2]),
                key="arima_p_values"
            )
            d_values = st.multiselect(
                "I (d) значения для поиска",
                options=list(range(3)),
                default=default_config.get('d_values', [0, 1]),
                key="arima_d_values"
            )
        with col2:
            q_values = st.multiselect(
                "MA (q) значения для поиска",
                options=list(range(6)),
                default=default_config.get('q_values', [0, 1, 2]),
                key="arima_q_values"
            )
            config.train_size = st.slider(
                "Размер обучающей выборки",
                min_value=0.5,
                max_value=0.9,
                value=default_config.get('train_size', 0.8),
                step=0.1,
                key="arima_train_size"
            )
    elif model_type in [ModelType.XGBOOST, ModelType.CATBOOST]:
        col1, col2 = st.columns(2)
        with col1:
            config.window_size = st.number_input(
                "Размер окна",
                min_value=1,
                max_value=60,
                value=default_config.get('window_size', 14),
                key=f"{model_type.value.lower()}_window_size"
            )
            n_estimators = st.number_input(
                "Количество деревьев" if model_type == ModelType.XGBOOST else "Количество итераций",
                min_value=10,
                max_value=1000,
                value=default_config.get('n_estimators', 100),
                key=f"{model_type.value.lower()}_n_estimators"
            )
        with col2:
            max_depth = st.number_input(
                "Максимальная глубина" if model_type == ModelType.XGBOOST else "Глубина деревьев",
                min_value=1,
                max_value=20,
                value=default_config.get('max_depth', 6),
                key=f"{model_type.value.lower()}_depth"
            )
            learning_rate = st.number_input(
                "Скорость обучения",
                min_value=0.001,
                max_value=1.0,
                value=default_config.get('learning_rate', 0.1),
                step=0.01,
                key=f"{model_type.value.lower()}_learning_rate"
            )
        config.n_splits = st.number_input(
            "Количество фолдов для кросс-валидации",
            min_value=2,
            max_value=10,
            value=default_config.get('n_splits', 5),
            key=f"{model_type.value.lower()}_n_splits"
        )
    elif model_type in [ModelType.LSTM, ModelType.GRU, ModelType.TRANSFORMER]:
        config.window_size = st.number_input(
            "Размер окна",
            min_value=1,
            max_value=60,
            value=default_config.get('window_size', 14),
            key="nn_window_size"
        )
        epochs = st.number_input(
            "Эпохи",
            min_value=1,
            max_value=200,
            value=default_config.get('epochs', 20),
            key="nn_epochs"
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=8,
            max_value=512,
            value=default_config.get('batch_size', 32),
            key="nn_batch_size"
        )
        pretrained_option = st.radio(
            "Использовать предобученные веса?",
            ["Обучить с нуля", "Загрузить предобученные веса"],
            key="forecast_pretrained_option"
        )
        if pretrained_option == "Загрузить предобученные веса":
            available_weights = get_available_weights(model_type.value)
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
    
    return config, model_type 