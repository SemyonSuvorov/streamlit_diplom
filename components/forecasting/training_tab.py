import streamlit as st
from state.session import state
import pandas as pd
from components.forecasting.model_registry import ModelType, ModelConfig, ModelFactory, ModelRegistry
from components.forecasting.model_weights import get_available_weights
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def get_model_description(model_type: ModelType) -> str:
    """Get description for each model type"""
    descriptions = {
        ModelType.ARIMA: """
        ARIMA (AutoRegressive Integrated Moving Average) - классическая модель для прогнозирования временных рядов.
        Состоит из трех компонентов:
        - AR (p) - авторегрессия, учитывает зависимость от предыдущих значений
        - I (d) - интегрирование, делает ряд стационарным
        - MA (q) - скользящее среднее, учитывает ошибки предыдущих прогнозов
        """,
        ModelType.XGBOOST: """
        XGBoost - градиентный бустинг на деревьях решений.
        Особенности:
        - Высокая точность прогнозирования
        - Устойчивость к выбросам
        - Возможность работы с пропущенными значениями
        """,
        ModelType.CATBOOST: """
        CatBoost - градиентный бустинг с категориальными признаками.
        Преимущества:
        - Автоматическая обработка категориальных признаков
        - Устойчивость к переобучению
        - Высокая скорость обучения
        """,
        ModelType.LSTM: """
        LSTM (Long Short-Term Memory) - рекуррентная нейронная сеть.
        Особенности:
        - Способность запоминать долгосрочные зависимости
        - Хорошо работает с нелинейными паттернами
        - Требует большого количества данных
        """,
        ModelType.GRU: """
        GRU (Gated Recurrent Unit) - упрощенная версия LSTM.
        Преимущества:
        - Меньше параметров, чем у LSTM
        - Быстрее обучается
        - Хорошо работает с короткими последовательностями
        """,
        ModelType.TRANSFORMER: """
        Transformer - архитектура на основе механизма внимания.
        Особенности:
        - Параллельная обработка последовательностей
        - Учет глобальных зависимостей
        - Высокая точность на больших наборах данных
        """,
        ModelType.PROPHET: """
        Prophet - модель для прогнозирования временных рядов от Facebook.
        Преимущества:
        - Учет сезонности и трендов
        - Устойчивость к выбросам
        - Простота настройки
        """,
        ModelType.RANDOM_FOREST: """
        Random Forest - ансамбль деревьев решений.
        Особенности:
        - Устойчивость к переобучению
        - Работа с нелинейными зависимостями
        - Интерпретируемость результатов
        """
    }
    return descriptions.get(model_type, "Описание модели отсутствует")

def get_parameter_description(param_name: str) -> str:
    """Get description for model parameters"""
    descriptions = {
        'window_size': """
        Размер окна - количество предыдущих значений, используемых для прогнозирования следующего значения.
        Большее значение может помочь уловить долгосрочные зависимости, но требует больше данных.
        """,
        'n_estimators': """
        Количество деревьев/итераций - чем больше, тем лучше модель может улавливать сложные зависимости,
        но увеличивает время обучения и риск переобучения.
        """,
        'max_depth': """
        Максимальная глубина дерева - ограничивает сложность модели.
        Большая глубина может привести к переобучению, маленькая - к недообучению.
        """,
        'learning_rate': """
        Скорость обучения - определяет, насколько сильно каждое дерево влияет на итоговый прогноз.
        Меньшее значение требует больше деревьев, но может дать лучший результат.
        """,
        'n_splits': """
        Количество фолдов для кросс-валидации - определяет, на сколько частей разбиваются данные
        для оценки качества модели. Большее значение дает более надежную оценку.
        """,
        'p_values': """
        AR (p) - порядок авторегрессии, определяет, сколько предыдущих значений использовать
        для прогнозирования следующего значения.
        """,
        'd_values': """
        I (d) - порядок интегрирования, определяет, сколько раз нужно взять разность ряда,
        чтобы сделать его стационарным.
        """,
        'q_values': """
        MA (q) - порядок скользящего среднего, определяет, сколько предыдущих ошибок использовать
        для прогнозирования следующего значения.
        """,
        'epochs': """
        Количество эпох - сколько раз модель увидит весь обучающий набор данных.
        Большее значение может улучшить качество, но увеличивает время обучения.
        """,
        'batch_size': """
        Размер батча - количество примеров, обрабатываемых за один проход.
        Больший размер может ускорить обучение, но требует больше памяти.
        """
    }
    return descriptions.get(param_name, "Описание параметра отсутствует")

def show_training_tab():
    """Display model training tab"""
    st.markdown("### Обучение модели")
    
    col1, col2 = st.columns(2)
    with col1:
        numeric_cols = state.get('filtered_df').select_dtypes(include='number').columns.tolist()
        target_col = st.selectbox(
            "Целевая переменная",
            numeric_cols,
            index=numeric_cols.index(state.get('target_col')) if state.get('target_col') in numeric_cols else 0,
            key="forecast_target_col"
        )
    with col2:
        train_size = st.slider(
            "Размер обучающей выборки",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.1,
            key="training_train_size"
        )

    st.markdown("---")

    # 2. Model selection with description
    model_options = [model.value for model in ModelType]
    selected_model = st.selectbox(
        "Модель прогнозирования",
        model_options,
        key="forecast_model"
    )
    
    # Show model description
    model_type = ModelType(selected_model)
    with st.expander("Описание модели", expanded=True):
        st.markdown(get_model_description(model_type))

    # 3. Model parameters configuration
    st.markdown("### Параметры модели")
    default_config = ModelRegistry.get_default_config(model_type)
    
    # Create configuration based on model type
    config = ModelConfig(
        target_col=target_col,
        train_size=train_size
    )
    
    # Add model-specific parameters
    if model_type == ModelType.ARIMA:
        st.markdown("#### Параметры ARIMA")
        with st.expander("AR (p) - порядок авторегрессии", expanded=False):
            st.markdown("""
            Определяет, сколько предыдущих значений использовать для прогнозирования следующего значения.
            - Большее значение может уловить более сложные зависимости
            - Меньшее значение делает модель проще и быстрее
            """)
        p_values = st.multiselect(
            "AR (p) значения для поиска",
            options=list(range(6)),
            default=default_config.get('p_values', [0, 1, 2]),
            key="arima_p_values"
        )
        
        with st.expander("I (d) - порядок интегрирования", expanded=False):
            st.markdown("""
            Определяет, сколько раз нужно взять разность ряда, чтобы сделать его стационарным.
            - Помогает устранить тренд в данных
            - Обычно используется значение 0, 1 или 2
            """)
        d_values = st.multiselect(
            "I (d) значения для поиска",
            options=list(range(3)),
            default=default_config.get('d_values', [0, 1]),
            key="arima_d_values"
        )
        
        with st.expander("MA (q) - порядок скользящего среднего", expanded=False):
            st.markdown("""
            Определяет, сколько предыдущих ошибок использовать для прогнозирования следующего значения.
            - Помогает учесть случайные колебания
            - Влияет на сглаживание прогноза
            """)
        q_values = st.multiselect(
            "MA (q) значения для поиска",
            options=list(range(6)),
            default=default_config.get('q_values', [0, 1, 2]),
            key="arima_q_values"
        )
    elif model_type in [ModelType.XGBOOST, ModelType.CATBOOST]:
        st.markdown("#### Параметры градиентного бустинга")
        # Добавляем выбор типа прогноза
        forecast_approach = st.radio(
            "Тип прогноза:",
            ["Рекурсивный (многошаговый)", "Прямой (отдельная модель на каждый шаг)",],
            index=0,
            key=f"{model_type.value.lower()}_forecast_approach"
        )
        # Сохраняем в конфиг ('recursive' или 'direct')
        config.forecast_approach = 'recursive' if forecast_approach.startswith('Рекурсивный') else 'direct'
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Количество деревьев/итераций", expanded=False):
                st.markdown("""
                Количество деревьев в ансамбле (XGBoost) или итераций обучения (CatBoost).
                - Больше деревьев = лучше качество, но дольше обучение
                - Меньше деревьев = быстрее обучение, но хуже качество
                """)
            n_estimators = st.number_input(
                "Количество деревьев" if model_type == ModelType.XGBOOST else "Количество итераций",
                min_value=10,
                max_value=1000,
                value=default_config.get('n_estimators', 100),
                key=f"{model_type.value.lower()}_n_estimators"
            )
            with st.expander("Количество фолдов для кросс-валидации", expanded=False):
                st.markdown("""
                Количество частей, на которые разбиваются данные для оценки качества модели.
                - Больше фолдов = более надежная оценка, но дольше обучение
                - Меньше фолдов = быстрее обучение, но менее надежная оценка
                """)
            config.n_splits = st.number_input(
                "Количество фолдов для кросс-валидации",
                min_value=2,
                max_value=10,
                value=default_config.get('n_splits', 5),
                key=f"{model_type.value.lower()}_n_splits"
            )
        with col2:
            with st.expander("Максимальная глубина", expanded=False):
                st.markdown("""
                Максимальная глубина каждого дерева в ансамбле.
                - Большая глубина = сложнее модель, риск переобучения
                - Малая глубина = проще модель, риск недообучения
                """)
            max_depth = st.number_input(
                "Максимальная глубина" if model_type == ModelType.XGBOOST else "Глубина деревьев",
                min_value=1,
                max_value=20,
                value=default_config.get('max_depth', 6),
                key=f"{model_type.value.lower()}_depth"
            )
            with st.expander("Скорость обучения", expanded=False):
                st.markdown("""
                Определяет, насколько сильно каждое дерево влияет на итоговый прогноз.
                - Меньшее значение = более плавное обучение, требует больше деревьев
                - Большее значение = более быстрое обучение, но может быть менее стабильным
                """)
            learning_rate = st.number_input(
                "Скорость обучения",
                min_value=0.001,
                max_value=1.0,
                value=default_config.get('learning_rate', 0.1),
                step=0.01,
                key=f"{model_type.value.lower()}_learning_rate"
            )
    
    elif model_type in [ModelType.LSTM, ModelType.GRU, ModelType.TRANSFORMER]:
        st.markdown("#### Параметры нейронной сети")
        with st.expander("Размер окна", expanded=False):
            st.markdown("""
            Количество предыдущих значений, используемых для прогнозирования следующего значения.
            - Большее значение может помочь уловить долгосрочные зависимости
            - Требует больше данных для обучения
            """)
        config.window_size = st.number_input(
            "Размер окна",
            min_value=1,
            max_value=60,
            value=default_config.get('window_size', 14),
            key="nn_window_size"
        )
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Эпохи", expanded=False):
                st.markdown("""
                Количество полных проходов по обучающему набору данных.
                - Больше эпох = лучше качество, но дольше обучение
                - Меньше эпох = быстрее обучение, но хуже качество
                """)
            epochs = st.number_input(
                "Эпохи",
                min_value=1,
                max_value=200,
                value=default_config.get('epochs', 20),
                key="nn_epochs"
            )
        with col2:
            with st.expander("Batch size", expanded=False):
                st.markdown("""
                Количество примеров, обрабатываемых за один проход.
                - Больший размер = быстрее обучение, но требует больше памяти
                - Меньший размер = медленнее обучение, но может дать лучший результат
                """)
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
    
    if st.button("Обучить модель", type="primary", key="training_train_model"):
        try:
            X_df = state.get('filtered_df').copy()
            
            # Check if 'feature_df' is in the state (features created in previous steps)
            if state.get('feature_df') is not None:
                X_df = state.get('feature_df').copy()
            
            # Ensure target column exists
            if config.target_col not in X_df.columns:
                st.error(f"Целевая переменная '{config.target_col}' не найдена в данных.")
                return
            
            # Check if we have a proper datetime index, if not try to find and set it
            if not isinstance(X_df.index, pd.DatetimeIndex):
                # Try to find date column
                date_cols = [col for col in X_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'дата' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    try:
                        X_df[date_col] = pd.to_datetime(X_df[date_col])
                        X_df = X_df.set_index(date_col)
                    except:
                        st.warning(f"Не удалось преобразовать колонку '{date_col}' в формат даты")
                else:
                    # Create a synthetic datetime index if none exists
                    st.warning("Дата не найдена в данных. Создаю синтетический временной индекс.")
                    X_df.index = pd.date_range(start='2023-01-01', periods=len(X_df), freq='D')

            # Check if index is sorted
            if not X_df.index.is_monotonic_increasing:
                X_df = X_df.sort_index()

            # Handle non-numeric features (especially important for tree-based models)
            for col in X_df.columns:
                if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                    st.warning(f"Столбец '{col}' содержит категориальные данные. Выполняется преобразование.")
                    X_df[col] = pd.Categorical(X_df[col]).codes
            
            # Replace NaN values which could cause issues
            if X_df.isna().any().any():
                st.warning("Обнаружены пропуски в данных. Выполняется заполнение.")
                X_df = X_df.fillna(X_df.mean(numeric_only=True))
            
            # Ensure all data is in proper format but exclude the target column from filtering
            target_value = X_df[config.target_col].copy()
            X_df_filtered = X_df.select_dtypes(include='number')
            
            # Make sure the target column is preserved
            if config.target_col not in X_df_filtered.columns:
                X_df_filtered[config.target_col] = target_value
            
            X_df = X_df_filtered

            # Добавляем базовые временные признаки всегда, если есть DatetimeIndex
            if isinstance(X_df.index, pd.DatetimeIndex):
                X_df['day'] = X_df.index.day
                X_df['month'] = X_df.index.month  
                X_df['year'] = X_df.index.year
                X_df['day_of_week'] = X_df.index.dayofweek
            else:
                # Если нет временного индекса, добавляем просто порядковый номер
                X_df['row_index'] = range(len(X_df))

            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            if X_df.isna().any().any():
                st.warning("В данных есть пропуски или бесконечные значения. Заполняю средними значениями.")
                X_df = X_df.fillna(X_df.mean(numeric_only=True))

            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Прогресс: {int(progress * 100)}%")
            model = ModelFactory.create_model(model_type, config)
            cv_metrics = model.cross_validate(X_df, config.target_col, progress_callback=update_progress)
            # УБРАНО: st.success("Кросс-валидация завершена!")
            st.markdown("### Основные метрики")
            if isinstance(cv_metrics, dict) and all(isinstance(v, list) for v in cv_metrics.values()):
                avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
                col1, col2, col3 = st.columns(3)
                for i, (metric_name, value) in enumerate(avg_metrics.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(str.upper(metric_name), f"{value:.4f}")
            else:
                st.write(cv_metrics)
            st.markdown("### График на тестовой выборке (последний фолд)")
            try:
                if hasattr(model, 'plot_test_predictions'):
                    fig = model.plot_test_predictions()
                    st.plotly_chart(fig, use_container_width=True)
                # УБРАНО: else: st.info("Для этой модели не реализован график тестовых предсказаний.")
            except Exception as e:
                st.error(f"Ошибка при построении графика: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            
            # Final training on all data
            try:
                # УБРАНО: st.info("Обучение финальной модели на всех данных...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Prepare data for final training
                if hasattr(model, 'fit'):
                    # Fit the model on the entire dataset
                    result = model.fit(X_df, config.target_col, 
                              progress_callback=update_progress)
                    # Check if the fit was successful
                    if result is not None:
                        # Set the fitted flag
                        if hasattr(model, '_is_fitted'):
                            model._is_fitted = True
                        # УБРАНО: st.success("Обучение финальной модели успешно завершено!")
                    else:
                        st.error("Ошибка при обучении финальной модели. Возможно, модель не сошлась или данные некорректны.")
                        return
                else:
                    # УБРАНО: st.warning("Модель не имеет метода fit(). Используется модель из кросс-валидации.")
                    pass
            except Exception as e:
                st.error(f"Ошибка при обучении финальной модели: {str(e)}")
                # УБРАНО: st.warning("Будет использована последняя модель из кросс-валидации.")
            finally:
                progress_bar.empty()
                status_text.empty()
            
            # Save model to session state
            if hasattr(model, '_is_fitted') and model._is_fitted:
                st.session_state[f"{model_type.value.lower()}_model"] = model
                # Save the feature dataframe to use for future forecasting
                state.set('feature_df', X_df.reset_index() if isinstance(X_df.index, pd.DatetimeIndex) else X_df)
                # УБРАНО: st.success(f"Модель {model_type.value} успешно обучена и сохранена!")
            else:
                # Final attempt to mark model as fitted if it has the required attributes
                if hasattr(model, 'best_model') and model.best_model is not None:
                    model._is_fitted = True
                    st.session_state[f"{model_type.value.lower()}_model"] = model
                    # Save the feature dataframe to use for future forecasting
                    state.set('feature_df', X_df.reset_index() if isinstance(X_df.index, pd.DatetimeIndex) else X_df)
                    # УБРАНО: st.success(f"Модель {model_type.value} успешно обучена и сохранена!")
                else:
                    st.error("Модель не была корректно обучена. Повторите попытку с другими параметрами.")
            return
        except Exception as e:
            st.error(f"Ошибка при кросс-валидации: {str(e)}") 