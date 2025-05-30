import streamlit as st
from state.session import state
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.forecasting.model_registry import ModelType
import numpy as np
from services.transformation_service import TransformationService

def inverse_transform_forecast(forecast_series, conf_int=None):
    """
    Выполняет обратное преобразование прогноза на основе
    сохраненных трансформаций стационарности
    """
    # Если преобразований не было, возвращаем исходные данные
    if 'stationarity_transformations' not in st.session_state or not st.session_state.stationarity_transformations:
        return forecast_series, conf_int
    
    # Применяем преобразования в обратном порядке
    transformed_forecast = forecast_series.copy()
    transformed_conf_int = None if conf_int is None else conf_int.copy()
    
    # Поскольку преобразования применялись последовательно,
    # мы применяем обратные преобразования в обратном порядке
    for transform_info in reversed(st.session_state.stationarity_transformations):
        transform_type = transform_info.get('type')
        params = transform_info.get('params', {})
        
        if transform_type == 'diff':
            # Для обратного дифференцирования нам нужно последнее значение исходного ряда
            order = params.get('order', 1)
            if state.get('stationarity_initial') is not None:
                original_ts = state.get('stationarity_initial').set_index(state.get('date_col'))[state.get('target_col')]
                last_values = original_ts.iloc[-order:].values
                
                # Накопительная сумма (cumsum) с начальным значением
                forecast_values = transformed_forecast.values
                for i in range(order):
                    # Используем последнее значение оригинального ряда как начальное
                    cumsum_values = np.concatenate([[last_values[-(i+1)]], forecast_values])
                    forecast_values = np.cumsum(cumsum_values)[1:]
                
                transformed_forecast = pd.Series(forecast_values, index=forecast_series.index)
                
                # Если есть доверительные интервалы, преобразуем их тоже
                if transformed_conf_int is not None:
                    for col in ['lower', 'upper']:
                        interval_values = transformed_conf_int[col].values
                        for i in range(order):
                            # Используем последнее значение как начальное
                            cumsum_interval = np.concatenate([[last_values[-(i+1)]], interval_values])
                            interval_values = np.cumsum(cumsum_interval)[1:]
                        transformed_conf_int[col] = interval_values
                
        elif transform_type == 'log':
            # Для обратного логарифмирования применяем экспоненту
            transformed_forecast = np.exp(transformed_forecast)
            
            # Если есть доверительные интервалы, преобразуем их тоже
            if transformed_conf_int is not None:
                transformed_conf_int['lower'] = np.exp(transformed_conf_int['lower'])
                transformed_conf_int['upper'] = np.exp(transformed_conf_int['upper'])
                
        elif transform_type == 'seasonal_diff':
            # Для обратного сезонного дифференцирования также нужны исходные значения
            seasonal_period = params.get('seasonal_period', 12)
            if state.get('stationarity_initial') is not None:
                original_ts = state.get('stationarity_initial').set_index(state.get('date_col'))[state.get('target_col')]
                
                # Получаем сезонные значения
                seasonal_values = original_ts.iloc[-seasonal_period:].values
                
                # Выполняем обратное сезонное дифференцирование
                forecast_values = transformed_forecast.values
                result_values = np.zeros_like(forecast_values)
                
                for i in range(len(forecast_values)):
                    if i < seasonal_period:
                        # Используем исходные сезонные значения
                        result_values[i] = forecast_values[i] + seasonal_values[i]
                    else:
                        # Используем уже рассчитанные значения
                        result_values[i] = forecast_values[i] + result_values[i - seasonal_period]
                
                transformed_forecast = pd.Series(result_values, index=forecast_series.index)
                
                # Если есть доверительные интервалы, преобразуем их тоже
                if transformed_conf_int is not None:
                    for col in ['lower', 'upper']:
                        interval_values = transformed_conf_int[col].values
                        result_interval = np.zeros_like(interval_values)
                        
                        for i in range(len(interval_values)):
                            if i < seasonal_period:
                                result_interval[i] = interval_values[i] + seasonal_values[i]
                            else:
                                result_interval[i] = interval_values[i] + result_interval[i - seasonal_period]
                                
                        transformed_conf_int[col] = result_interval
    
    return transformed_forecast, transformed_conf_int

def show_feature_importance_tab(model, model_type):
    """
    Отображает вкладку с информацией о важности признаков для регрессионных моделей
    """
    st.subheader("Важность признаков")
    
    # Специальная обработка для DMEN
    if model_type == ModelType.DMEN:
        try:
            # Отображаем динамические веса моделей
            st.markdown("### Динамические веса моделей")
            weights = model.get_model_weights()
            
            # График весов
            fig_weights = model.plot_model_weights()
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Таблица весов
            weights_df = pd.DataFrame({
                'Модель': list(weights.keys()),
                'Вес': list(weights.values())
            })
            st.dataframe(weights_df)
            
            # Матрица взаимного усиления
            st.markdown("### Матрица взаимного усиления (Θ)")
            theta_matrix = model.get_theta_matrix()
            
            # Создаем DataFrame для лучшего отображения
            theta_df = pd.DataFrame(
                theta_matrix,
                index=['SARIMA', 'XGBoost', 'CatBoost', 'LSTM'],
                columns=['SARIMA', 'XGBoost', 'CatBoost', 'LSTM']
            )
            
            # Тепловая карта матрицы
            fig_theta = go.Figure(data=go.Heatmap(
                z=theta_matrix,
                x=['SARIMA', 'XGBoost', 'CatBoost', 'LSTM'],
                y=['SARIMA', 'XGBoost', 'CatBoost', 'LSTM'],
                colorscale='RdBu',
                zmid=0,
                text=np.round(theta_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig_theta.update_layout(
                title="Матрица взаимного усиления между моделями",
                xaxis_title="Модель-источник остатков",
                yaxis_title="Модель-получатель усиления"
            )
            
            st.plotly_chart(fig_theta, use_container_width=True)
            
            # Объяснение матрицы
            st.markdown("""
            ### Интерпретация матрицы взаимного усиления
            
            Матрица показывает, как остатки одной модели влияют на прогнозы другой:
            
            - **Положительные значения**: Остатки модели-источника улучшают прогнозы модели-получателя
            - **Отрицательные значения**: Остатки модели-источника ухудшают прогнозы модели-получателя
            - **Диагональ всегда равна 0**: Модель не может усиливать саму себя
            
            Значения обновляются динамически в процессе обучения на основе градиентного спуска.
            """)
            
        except Exception as e:
            st.error(f"Ошибка при отображении информации о DMEN: {str(e)}")
        
        return
    
    # Проверяем, что модель является регрессионной (XGBoost или CatBoost)
    if model_type in [ModelType.XGBOOST, ModelType.CATBOOST]:
        try:
            # Получаем важность признаков
            if not hasattr(model, 'feature_names') or not model.feature_names:
                st.warning("Информация о признаках недоступна для данной модели")
                return
                
            if model_type == ModelType.XGBOOST:
                # Для XGBoost используем встроенную функцию get_score()
                feature_importance = model.model.get_booster().get_score(importance_type='weight')
                # Преобразуем в DataFrame для отображения
                importance_df = pd.DataFrame({
                    'Признак': list(feature_importance.keys()),
                    'Важность': list(feature_importance.values())
                })
                importance_df = importance_df.sort_values('Важность', ascending=False)
                
            elif model_type == ModelType.CATBOOST:
                # Для CatBoost получаем важность признаков
                importance = model.model.get_feature_importance()
                importance_df = pd.DataFrame({
                    'Признак': model.feature_names,
                    'Важность': importance
                })
                importance_df = importance_df.sort_values('Важность', ascending=False)
                
            fig = go.Figure()
            
            # Ограничиваем количество отображаемых признаков (топ-10)
            top_n = min(10, len(importance_df))
            top_features = importance_df.head(top_n)
            
            fig.add_trace(go.Bar(
                x=top_features['Признак'],
                y=top_features['Важность'],
                marker_color='royalblue'
            ))
            
            fig.update_layout(
                title=f"Топ-{top_n} важных признаков",
                xaxis_title="Признак",
                yaxis_title="Важность",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Добавляем описание влияния признаков
            st.markdown("### Интерпретация важности признаков")
            st.markdown("""
            Важность признаков показывает, насколько сильно каждый признак влияет на прогноз модели:
            
            - **Высокая важность**: Признаки с высокой важностью оказывают существенное влияние на результат прогнозирования.
            - **Низкая важность**: Признаки с низкой важностью вносят меньший вклад в прогноз.
            
            При разработке новых моделей рекомендуется фокусироваться на признаках с высокой важностью.
            """)
            
        except Exception as e:
            st.error(f"Ошибка при отображении важности признаков: {str(e)}")
    else:
        st.info("Информация о важности признаков доступна только для регрессионных моделей (XGBoost, CatBoost) и DMEN")

def show_forecast_tab():
    """Display forecast tab"""
    st.markdown("### Прогнозирование")
    
    # Get available models
    available_models = []
    model_type = None
    for model in ModelType:
        model_key = f"{model.value.lower()}_model"
        if model_key in st.session_state and st.session_state[model_key].is_fitted():
            available_models.append(model.value)
            if model_type is None:  # Use the first available model as default
                model_type = model
    
    if not available_models:
        st.warning("Сначала обучите модель на вкладке 'Обучение'")
        return
    
    col1,col2,col3 = st.columns(3)
    with col1:
        st.success(f"Доступные модели: {', '.join(available_models)}")
    
    # Create columns for parameters
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if len(available_models) > 1:
            selected_model_name = st.selectbox(
                "Выберите модель для прогнозирования",
                available_models,
                key="forecast_selected_model"
            )
            # Update model_type based on selection
            for model in ModelType:
                if model.value == selected_model_name:
                    model_type = model
                    break
    
    # Get trained model
    model = st.session_state[f"{model_type.value.lower()}_model"]
    
    with col1:
        # Get forecast horizon
        forecast_horizon = st.number_input(
            "Горизонт прогнозирования (дни)",
            min_value=1,
            max_value=365,
            value=30,
            key="forecast_tab_horizon"
        )

        
    
    # Добавляем чекбокс для обратного преобразования только если были применены преобразования стационарности
    inverse_transform = False
    if 'stationarity_transformations' in st.session_state and st.session_state.stationarity_transformations:
        inverse_transform = st.checkbox(
            "Выполнить обратное преобразование стационарности",
            value=True,
            help="Применить обратное преобразование к прогнозу, чтобы вернуть его к исходному масштабу данных",
            key="forecast_inverse_transform"
        )
    show_confidence = False
    if model_type == ModelType.SARIMA:
        show_confidence = st.checkbox(
            "Показывать доверительные интервалы",
            value=True,
            key="forecast_show_confidence"
        )
    if st.button("Сделать прогноз", type="primary", key="forecast_make_forecast"):
        progress_bar = None
        status_text = None
        try:
            # Check if model is trained
            if hasattr(model, 'is_fitted') and callable(model.is_fitted) and not model.is_fitted():
                st.error("Модель не обучена. Пожалуйста, вернитесь на вкладку 'Обучение' и убедитесь, что модель обучена корректно.")
                return
            
            # Prepare data
            date_col = state.get('date_col')
            feature_df = state.get('feature_df')
            if feature_df is not None:
                ts = feature_df.copy()
                # Преобразование date_col в datetimeindex, если он задан и есть в ts
                if date_col is not None and date_col in ts.columns:
                    ts[date_col] = pd.to_datetime(ts[date_col], errors='coerce')
                    ts = ts.set_index(date_col)
                elif 'index' in ts.columns:
                    ts = ts.set_index('index')
            else:
                st.error("Не найден feature_df для прогноза. Переобучите модель на вкладке 'Обучение'.")
                return

            if model_type != ModelType.SARIMA:
                # Define target_col at the beginning
                target_col = model.config.target_col if hasattr(model, 'config') and hasattr(model.config, 'target_col') else state.get('target_col')
                
                model_features = set(getattr(model, 'feature_names', []))
                df_features = set([col for col in ts.columns if col != target_col])
                missing_in_df = model_features - df_features
                extra_in_df = df_features - model_features
                
                # If using LSTM model and missing lag features, create them
                if model_type == ModelType.LSTM:
                    # Get the features that were used during training
                    lstm_features = state.get('lstm_features')
                    if lstm_features is not None:
                        # Only check for missing features that were used in training
                        model_features = set(lstm_features)
                        missing_in_df = model_features - df_features
                        extra_in_df = set()  # Ignore extra features for LSTM
                        
                        # Create any missing lag features
                        for feature in list(missing_in_df):
                            if '_lag' in feature:
                                parts = feature.split('_lag')
                                base_feature = '_'.join(parts[0].split('_'))
                                lag_num = int(parts[1])
                                
                                if base_feature in ts.columns:
                                    ts[feature] = ts[base_feature].shift(lag_num)
                                    missing_in_df.remove(feature)
                        
                        # Fill NaN values created by the lag
                        ts = ts.fillna(method='bfill').fillna(method='ffill')
                        
                        # Update feature lists after creating lag features
                        df_features = set([col for col in ts.columns if col != target_col])
                        missing_in_df = model_features - df_features
                    else:
                        if missing_in_df or (extra_in_df and not all(col in ['dayofweek_sin', 'dayofweek_cos', 'day_of_week', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'] for col in extra_in_df)):
                            st.error(f"Набор признаков в модели не совпадает с текущими признаками в данных!\n"
                                     f"\nОтсутствуют в данных: {', '.join(missing_in_df) if missing_in_df else 'нет'}"
                                     f"\nЛишние в данных: {', '.join(extra_in_df) if extra_in_df else 'нет'}\n"
                                     "Переобучите модель или вернитесь к шагу трансформации.")
                            return
                else:
                    if missing_in_df or (extra_in_df and not all(col in ['dayofweek_sin', 'dayofweek_cos', 'day_of_week', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'] for col in extra_in_df)):
                        st.error(f"Набор признаков в модели не совпадает с текущими признаками в данных!\n"
                                 f"\nОтсутствуют в данных: {', '.join(missing_in_df) if missing_in_df else 'нет'}"
                                 f"\nЛишние в данных: {', '.join(extra_in_df) if extra_in_df else 'нет'}\n"
                                 "Переобучите модель или вернитесь к шагу трансформации.")
                        return

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Прогресс прогнозирования: {int(progress * 100)}%")
            
            # Make forecast
            try:
                if model_type == ModelType.SARIMA:
                    forecast, conf_int = model.forecast(ts[model.config.target_col], forecast_horizon)
                elif model_type == ModelType.LSTM:
                    # For LSTM, use the features that were selected during training
                    lstm_features = state.get('lstm_features')
                    
                    # Ensure target column is available
                    target_col = model.config.target_col if hasattr(model, 'config') and hasattr(model.config, 'target_col') else state.get('target_col')
                    if target_col not in ts.columns:
                        st.error(f"Целевая переменная '{target_col}' не найдена в данных для прогнозирования. Доступные колонки: {', '.join(ts.columns)}")
                        if progress_bar is not None:
                            progress_bar.empty()
                        if status_text is not None:
                            status_text.empty()
                        return
                    
                    if lstm_features is not None:
                        # Create any missing lag features
                        for feature in lstm_features:
                            if '_lag' in feature and feature not in ts.columns:
                                parts = feature.split('_lag')
                                base_feature = '_'.join(parts[0].split('_'))
                                lag_num = int(parts[1])
                                if base_feature in ts.columns:
                                    ts[feature] = ts[base_feature].shift(lag_num)
                        
                        # Fill NaN values from the lag features
                        ts = ts.fillna(method='bfill').fillna(method='ffill')
                    
                
                    
                    forecast, conf_int = model.forecast(ts, forecast_horizon, progress_callback=update_progress)
                else:
                    forecast, conf_int = model.forecast(ts, forecast_horizon, progress_callback=update_progress)

            except Exception as e:
                error_msg = str(e)
                if "Model has not been trained" in error_msg or "Call fit() first" in error_msg:
                    st.error("Модель не была обучена. Пожалуйста, вернитесь на вкладку 'Обучение' и корректно обучите модель.")
                    if progress_bar is not None:
                        progress_bar.empty()
                    if status_text is not None:
                        status_text.empty()
                    return
                else:
                    # Re-raise if it's another error
                    raise
                    
            # Выполняем обратное преобразование, если это запрошено и были трансформации
            original_forecast = forecast.copy()
            original_conf_int = None if conf_int is None else conf_int.copy()
            
            if inverse_transform:
                forecast, conf_int = inverse_transform_forecast(forecast, conf_int)
                    
            # Store forecast in session state
            st.session_state[f"{model_type.value.lower()}_forecast"] = forecast
            st.session_state[f"{model_type.value.lower()}_conf_int"] = conf_int
            
            col1,col2,col3 = st.columns(3)
            with col1:
                st.success("Прогноз успешно создан!")

            
            # Создаем вкладки для отображения результатов
            forecast_tab, importance_tab = st.tabs(["Результаты прогноза", "Важность признаков"])
            
            with forecast_tab:
                # Create interactive plot with Plotly
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=("Прогноз временного ряда", "Детали прогноза"),
                                  vertical_spacing=0.2)
                
                # Add actual values
                if hasattr(model, 'config') and hasattr(model.config, 'target_col'):
                    target_col = model.config.target_col
                else:
                    target_col = state.get('target_col')
                    
                # Если выполнено обратное преобразование и есть исходный ряд, используем его вместо текущего
                historical_data = ts[target_col]
                if inverse_transform and state.get('stationarity_initial') is not None:
                    original_ts = state.get('stationarity_initial').set_index(state.get('date_col'))[state.get('target_col')]
                    fig.add_trace(
                        go.Scatter(x=original_ts.index, y=original_ts.values, name="Исторические данные"),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(x=ts.index, y=ts[target_col], name="Исторические данные"),
                        row=1, col=1
                    )
                
                # Add forecast
                fig.add_trace(
                    go.Scatter(x=forecast.index, y=forecast.values, name="Прогноз",
                              line=dict(color='red')),
                    row=1, col=1
                )
                
                # Add confidence intervals if available and requested (только для SARIMA)
                if model_type == ModelType.SARIMA and conf_int is not None and show_confidence:
                    fig.add_trace(
                        go.Scatter(x=forecast.index, y=conf_int['upper'], 
                                  name="Верхняя граница", line=dict(color='rgba(255,0,0,0.2)')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=forecast.index, y=conf_int['lower'], 
                                  name="Нижняя граница", line=dict(color='rgba(255,0,0,0.2)'),
                                  fill='tonexty'),
                        row=1, col=1
                    )
                
                # Add forecast details
                # 1. Trend analysis
                try:
                    forecast_values = np.array(forecast.values, dtype=np.float64)
                    if np.isnan(forecast_values).any():
                        st.warning("В прогнозе есть некорректные значения. Анализ тренда может быть неточным.")
                        forecast_values = np.nan_to_num(forecast_values)
                    
                    trend = np.polyfit(range(len(forecast)), forecast_values, 1)[0]
                    fig.add_trace(
                        go.Scatter(x=forecast.index, y=forecast_values, name="Тренд",
                                  line=dict(color='green', dash='dash')),
                        row=2, col=1
                    )
                    
                    # 2. Seasonal decomposition (if applicable)
                    if len(forecast) > 7:  # Only if we have enough data for weekly seasonality
                        weekly_avg = pd.Series(forecast_values, index=forecast.index).groupby(forecast.index.dayofweek).mean()
                        seasonal = weekly_avg.reindex(forecast.index.dayofweek).values
                        fig.add_trace(
                            go.Scatter(x=forecast.index, y=seasonal, name="Сезонность",
                                      line=dict(color='purple', dash='dot')),
                            row=2, col=1
                        )
                except Exception as e:
                    st.warning(f"Не удалось выполнить анализ тренда и сезонности: {str(e)}")
                    # Продолжаем без анализа тренда и сезонности
                    pass
                
                # Update layout
                transformed_text = " (обратно преобразованный)" if inverse_transform else ""
                fig.update_layout(
                    height=800, 
                    showlegend=True,
                    title=f"Прогноз{transformed_text}"
                )
                fig.update_xaxes(title_text="Дата", row=1, col=1)
                fig.update_xaxes(title_text="Дата", row=2, col=1)
                fig.update_yaxes(title_text="Значение", row=1, col=1)
                fig.update_yaxes(title_text="Значение", row=2, col=1)
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast analysis
                st.subheader("Анализ прогноза")
                
                # Create columns for different metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Основные характеристики:**")
                    forecast_stats = pd.DataFrame({
                        'Метрика': ['Среднее значение', 'Стандартное отклонение', 
                                  'Минимум', 'Максимум'],
                        'Значение': [forecast.mean(), forecast.std(), 
                                   forecast.min(), forecast.max()]
                    })
                    st.dataframe(forecast_stats)
                
                with col2:
                    st.markdown("**Тренд:**")
                    trend_stats = pd.DataFrame({
                        'Метрика': ['Наклон тренда', 'Изменение за период'],
                        'Значение': [trend, trend * len(forecast)]
                    })
                    st.dataframe(trend_stats)
                
                with col3:
                    # Показываем информацию о доверительных интервалах только для SARIMA
                    if model_type == ModelType.SARIMA:
                        st.markdown("**Доверительные интервалы:**")
                        if conf_int is not None:
                            conf_stats = pd.DataFrame({
                                'Метрика': ['Средняя ширина', 'Максимальная ширина'],
                                'Значение': [(conf_int['upper'] - conf_int['lower']).mean(),
                                           (conf_int['upper'] - conf_int['lower']).max()]
                            })
                            st.dataframe(conf_stats)
                        else:
                            st.info("Доверительные интервалы недоступны")
                    else:
                        st.markdown("**Дополнительно:**")
                        st.info("Для получения дополнительной информации о модели перейдите на вкладку \"Важность признаков\"")
                
                # Display forecast values in a table
                st.subheader("Таблица прогноза")
                forecast_df = pd.DataFrame({
                    'Date': forecast.index,
                    'Value': forecast.values
                })
                # Добавляем доверительные интервалы в таблицу только для SARIMA
                if model_type == ModelType.SARIMA and conf_int is not None:
                    forecast_df['Нижняя граница'] = conf_int['lower']
                    forecast_df['Верхняя граница'] = conf_int['upper']
                st.dataframe(forecast_df)
                
                # Add download button for forecast data
                csv = forecast_df.to_csv(index=False, encoding="cp1251")
                st.download_button(
                    label="Скачать прогноз",
                    data=csv,
                    file_name=f"forecast_{model_type.value.lower()}.csv",
                    mime="text/csv"
                )
            
            # Показываем вкладку с важностью признаков только для поддерживаемых моделей
            with importance_tab:
                show_feature_importance_tab(model, model_type)
                
        except Exception as e:
            st.write(e)
            st.error(f"Ошибка при создании прогноза: {str(e)}")
        finally:
            if progress_bar is not None:
                progress_bar.empty()
            if status_text is not None:
                status_text.empty() 