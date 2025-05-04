import streamlit as st
from state.session import state
from components.forecasting.model_registry import ModelType, ModelConfig, ModelRegistry
from components.forecasting.model_weights import get_available_weights

def show_model_selection_tab() -> tuple[ModelConfig, ModelType]:
    """UI выбора модели и параметров. Возвращает config и тип модели."""
    st.markdown("### Выбор модели и параметров")
    
    # 1. Целевая переменная
    numeric_cols = state.get('filtered_df').select_dtypes(include='number').columns.tolist()
    target_col = st.selectbox(
        "Целевая переменная",
        numeric_cols,
        index=numeric_cols.index(state.get('target_col')) if state.get('target_col') in numeric_cols else 0,
        key="forecast_target_col"
    )

    # 2. Размер обучающей выборки (универсально)
    train_size = st.slider(
        "Размер обучающей выборки",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.1,
        key="training_train_size"
    )

    st.markdown("---")

    # 3. Выбор модели
    model_options = [model.value for model in ModelType]
    selected_model = st.selectbox(
        "Модель прогнозирования",
        model_options,
        key="forecast_model"
    )
    model_type = ModelType(selected_model)
    default_config = ModelRegistry.get_default_config(model_type)

    # 4. Параметры модели
    st.markdown("### Параметры модели")
    config = ModelConfig(
        target_col=target_col,
        train_size=train_size
    )

    # SARIMA
    if model_type == ModelType.SARIMA:
        st.markdown("#### Параметры SARIMA (auto_arima)")
        col1, col2 = st.columns(2)
        with col1:
            config.n_splits = st.number_input(
                "Количество фолдов для кросс-валидации",
                min_value=2,
                max_value=10,
                value=default_config.get('n_splits', 5),
                key="sarimax_n_splits"
            )
        with col2:
            default_seasonal = default_config.get('seasonal_period', 12)
            if default_seasonal is None or default_seasonal < 1:
                default_seasonal = 12
            config.seasonal_period = st.number_input(
                "Период сезонности (s)",
                min_value=1,
                max_value=365,
                value=default_seasonal,
                key="sarimax_seasonal_period"
            )

    elif model_type in [ModelType.XGBOOST, ModelType.CATBOOST]:
        st.markdown("#### Параметры градиентного бустинга")
        forecast_approach = st.radio(
            "Тип прогноза:",
            ["Рекурсивный (многошаговый)", "Прямой (отдельная модель на каждый шаг)"],
            index=0,
            key=f"{model_type.value.lower()}_forecast_approach"
        )
        config.forecast_approach = 'recursive' if forecast_approach.startswith('Рекурсивный') else 'direct'
        col1, col2 = st.columns(2)
        with col1:
            config.n_estimators = st.number_input(
                "Количество деревьев" if model_type == ModelType.XGBOOST else "Количество итераций",
                min_value=10,
                max_value=1000,
                value=default_config.get('n_estimators', 100),
                key=f"{model_type.value.lower()}_n_estimators"
            )
            config.n_splits = st.number_input(
                "Количество фолдов для кросс-валидации",
                min_value=2,
                max_value=10,
                value=default_config.get('n_splits', 5),
                key=f"{model_type.value.lower()}_n_splits"
            )
        with col2:
            config.max_depth = st.number_input(
                "Максимальная глубина" if model_type == ModelType.XGBOOST else "Глубина деревьев",
                min_value=1,
                max_value=20,
                value=default_config.get('max_depth', 6),
                key=f"{model_type.value.lower()}_depth"
            )
            config.learning_rate = st.number_input(
                "Скорость обучения",
                min_value=0.001,
                max_value=1.0,
                value=default_config.get('learning_rate', 0.1),
                step=0.01,
                key=f"{model_type.value.lower()}_learning_rate"
            )
    # LSTM, GRU, TRANSFORMER
    elif model_type in [ModelType.LSTM, ModelType.GRU, ModelType.TRANSFORMER]:
        st.markdown("#### Параметры нейронной сети")
        config.epochs = st.number_input(
            "Эпохи",
            min_value=1,
            max_value=200,
            value=default_config.get('epochs', 20),
            key="nn_epochs"
        )
        config.batch_size = st.number_input(
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
        config.pretrained_option = pretrained_option
        if pretrained_option == "Загрузить предобученные веса":
            available_weights = get_available_weights(model_type.value)
            if available_weights:
                config.selected_weights = st.selectbox(
                    "Выберите предобученные веса",
                    available_weights,
                    key="forecast_selected_weights"
                )
            else:
                st.info("Нет доступных предобученных весов для этой модели.")
            config.uploaded_weights = st.file_uploader(
                "Или загрузите свой файл весов",
                type=["h5", "pth", "ckpt", "pkl"],
                key="forecast_uploaded_weights"
            )
    # Другие модели (например, Prophet, Random Forest и т.д.) можно добавить аналогично

    return config, model_type 