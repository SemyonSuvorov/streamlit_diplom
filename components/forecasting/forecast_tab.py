import streamlit as st
from state.session import state
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.forecasting.model_registry import ModelType
import numpy as np
from services.transformation_service import TransformationService

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
    
    if available_models:
        st.success(f"Доступные модели: {', '.join(available_models)}")
        
        # Add model selection if multiple models are available
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
    else:
        st.warning("Сначала обучите модель на вкладке 'Обучение'")
        return
    
    # Get trained model
    model = st.session_state[f"{model_type.value.lower()}_model"]
    
    # Get forecast horizon
    forecast_horizon = st.number_input(
        "Горизонт прогнозирования (дни)",
        min_value=1,
        max_value=365,
        value=30,
        key="forecast_tab_horizon"
    )
    
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

            # === Проверка совпадения признаков ===
            model_features = set(getattr(model, 'feature_names', []))
            df_features = set([col for col in ts.columns if col != (model.config.target_col if hasattr(model, 'config') and hasattr(model.config, 'target_col') else state.get('target_col'))])
            missing_in_df = model_features - df_features
            extra_in_df = df_features - model_features
            if missing_in_df or extra_in_df:
                st.error(f"Набор признаков в модели не совпадает с текущими признаками в данных!\n"
                         f"\nОтсутствуют в данных: {', '.join(missing_in_df) if missing_in_df else 'нет'}"
                         f"\nЛишние в данных: {', '.join(extra_in_df) if extra_in_df else 'нет'}\n"
                         "Переобучите модель или вернитесь к шагу трансформации.")
                return
            # === Конец проверки совпадения признаков ===

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Прогресс прогнозирования: {int(progress * 100)}%")
            
            # Make forecast
            try:
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
            
            # Store forecast in session state
            st.session_state[f"{model_type.value.lower()}_forecast"] = forecast
            st.session_state[f"{model_type.value.lower()}_conf_int"] = conf_int
            
            # Show forecast results
            st.success("Прогноз успешно создан!")
            
            # Create interactive plot with Plotly
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=("Прогноз временного ряда", "Детали прогноза"),
                              vertical_spacing=0.2)
            
            # Add actual values
            if hasattr(model, 'config') and hasattr(model.config, 'target_col'):
                target_col = model.config.target_col
            else:
                target_col = state.get('target_col')
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
            
            # Add confidence intervals if available and requested
            if conf_int is not None and show_confidence:
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
            trend = np.polyfit(range(len(forecast)), forecast.values, 1)[0]
            fig.add_trace(
                go.Scatter(x=forecast.index, y=forecast.values, name="Тренд",
                          line=dict(color='green', dash='dash')),
                row=2, col=1
            )
            
            # 2. Seasonal decomposition (if applicable)
            if len(forecast) > 7:  # Only if we have enough data for weekly seasonality
                weekly_avg = forecast.groupby(forecast.index.dayofweek).mean()
                seasonal = weekly_avg.reindex(forecast.index.dayofweek).values
                fig.add_trace(
                    go.Scatter(x=forecast.index, y=seasonal, name="Сезонность",
                              line=dict(color='purple', dash='dot')),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(height=800, showlegend=True)
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
            
            # Display forecast values in a table
            st.subheader("Таблица прогноза")
            forecast_df = pd.DataFrame({
                'Дата': forecast.index,
                'Прогноз': forecast.values
            })
            if conf_int is not None:
                forecast_df['Нижняя граница'] = conf_int['lower']
                forecast_df['Верхняя граница'] = conf_int['upper']
            st.dataframe(forecast_df)
            
            # Add download button for forecast data
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать прогноз",
                data=csv,
                file_name=f"forecast_{model_type.value.lower()}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.write(e)
            st.error(f"Ошибка при создании прогноза: {str(e)}")
        finally:
            if progress_bar is not None:
                progress_bar.empty()
            if status_text is not None:
                status_text.empty() 