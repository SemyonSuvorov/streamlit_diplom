import streamlit as st
from state.session import state
import pandas as pd
from components.forecasting.model_registry import ModelType, ModelConfig, ModelFactory, ModelRegistry
from components.forecasting.model_weights import get_available_weights
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from components.forecasting.model_selection import show_model_selection_tab

def get_model_description(model_type: ModelType) -> str:
    """Get description for each model type"""
    descriptions = {
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
        """,
        ModelType.SARIMA: """
        SARIMAX (Seasonal ARIMA with eXogenous regressors) - расширение ARIMA, учитывающее сезонность.
        Состоит из:
        - AR (p) - авторегрессия
        - I (d) - интегрирование
        - MA (q) - скользящее среднее
        - Seasonal (P, D, Q, s) - сезонные компоненты
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

    # Получаем параметры и тип модели через UI выбора
    config, model_type = show_model_selection_tab()

    # Описание модели (можно добавить expander при желании)
    with st.expander("Описание модели", expanded=True):
        st.markdown(get_model_description(model_type))

    if st.button("Обучить модель", type="primary", key="training_train_model"):
        try:
            X_df = state.get('filtered_df').copy()
            if state.get('feature_df') is not None:
                X_df = state.get('feature_df').copy()
            if config.target_col not in X_df.columns:
                st.error(f"Целевая переменная '{config.target_col}' не найдена в данных.")
                return
            if not isinstance(X_df.index, pd.DatetimeIndex):
                date_cols = [col for col in X_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'дата' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    try:
                        X_df[date_col] = pd.to_datetime(X_df[date_col])
                        X_df = X_df.set_index(date_col)
                    except:
                        st.warning(f"Не удалось преобразовать колонку '{date_col}' в формат даты")
                else:
                    st.warning("Дата не найдена в данных. Создаю синтетический временной индекс.")
                    X_df.index = pd.date_range(start='2023-01-01', periods=len(X_df), freq='D')
            if not X_df.index.is_monotonic_increasing:
                X_df = X_df.sort_index()
            for col in X_df.columns:
                if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                    st.warning(f"Столбец '{col}' содержит категориальные данные. Выполняется преобразование.")
                    X_df[col] = pd.Categorical(X_df[col]).codes
            if X_df.isna().any().any():
                st.warning("Обнаружены пропуски в данных. Выполняется заполнение.")
                X_df = X_df.fillna(X_df.mean(numeric_only=True))
            target_value = X_df[config.target_col].copy()
            X_df_filtered = X_df.select_dtypes(include='number')
            if config.target_col not in X_df_filtered.columns:
                X_df_filtered[config.target_col] = target_value
            X_df = X_df_filtered
            if isinstance(X_df.index, pd.DatetimeIndex):
                X_df['day'] = X_df.index.day
                X_df['month'] = X_df.index.month  
                X_df['year'] = X_df.index.year
                X_df['day_of_week'] = X_df.index.dayofweek
                X_df['quarter'] = X_df.index.quarter
                X_df['month_sin'] = np.sin(2 * np.pi * X_df['month'] / 12)
                X_df['month_cos'] = np.cos(2 * np.pi * X_df['month'] / 12)
                X_df['quarter_sin'] = np.sin(2 * np.pi * X_df['quarter'] / 4)
                X_df['quarter_cos'] = np.cos(2 * np.pi * X_df['quarter'] / 4)
                X_df['dayofweek_sin'] = np.sin(2 * np.pi * X_df['day_of_week'] / 7)
                X_df['dayofweek_cos'] = np.cos(2 * np.pi * X_df['day_of_week'] / 7)
            else:
                X_df['row_index'] = range(len(X_df))
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            if X_df.isna().any().any():
                st.warning("В данных есть пропуски или бесконечные значения. Заполняю средними значениями.")
                X_df = X_df.fillna(X_df.mean(numeric_only=True))

            if model_type == ModelType.SARIMA:
                with st.spinner("Идет подбор параметров SARIMA..."):
                    model = ModelFactory.create_model(model_type, config)
                    model.fit(X_df[config.target_col])
                    cv_metrics = model.cross_validate(X_df[[config.target_col]], config.target_col)
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Прогресс: {int(progress * 100)}%")
                model = ModelFactory.create_model(model_type, config)
                # Сначала обучаем модель
                model.fit(X_df, config.target_col, progress_callback=update_progress)
                # Затем делаем кросс-валидацию
                cv_metrics = model.cross_validate(X_df, config.target_col, progress_callback=update_progress)

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
            except Exception as e:
                st.error(f"Ошибка при построении графика: {str(e)}")
            if model_type != ModelType.SARIMA:
                progress_bar.empty()
                status_text.empty()
            
            if hasattr(model, '_is_fitted'):
                if model._is_fitted:
                    st.session_state[f"{model_type.value.lower()}_model"] = model
                    state.set('feature_df', X_df.reset_index() if isinstance(X_df.index, pd.DatetimeIndex) else X_df)
                    st.success("Модель успешно обучена и сохранена!")
                else:
                    st.error("Модель не была корректно обучена. Повторите попытку с другими параметрами.")
            else:
                st.error("Ошибка инициализации модели. Проверьте параметры и данные.")
            return
        except Exception as e:
            import traceback
            st.error(f"Ошибка при обучении модели: {str(e)}") 