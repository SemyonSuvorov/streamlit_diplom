import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def run_step():
    st.subheader("Шаг 2. Предобработка данных")
    
    if st.session_state.filtered_df is not None:
        tab1, tab2, tab3 = st.tabs(["Описание данных", "Обработка пропусков", "бла бла бла"])
        
        with tab1:
            
            st.write("### 📊 Описательная статистика")
            # Общая информация о данных
            st.markdown("**Основные характеристики данных:**")
            info_container = st.container(border=True)
            with info_container:
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Столбцов", len(st.session_state.filtered_df.columns))
                with cols[1]:
                    st.metric("Наблюдений", len(st.session_state.filtered_df))
                with cols[2]:
                    st.metric("Пропусков", st.session_state.filtered_df.isnull().sum().sum())
                with cols[3]:
                    st.metric("Дубликаты", st.session_state.filtered_df.duplicated().sum())

            # Расширенная статистика для числовых столбцов
            if st.session_state.target_col in st.session_state.filtered_df.select_dtypes(include='number'):
                st.markdown("---")
                
                stats = st.session_state.filtered_df[st.session_state.target_col].agg([
                    'mean', 'median', 'std', 'min', 'max', 'skew'
                ]).reset_index()
                
                stats.columns = ['Метрика', 'Значение']
                stats['Метрика'] = [
                    'Среднее', 'Медиана', 'Станд. отклонение', 
                    'Минимум', 'Максимум', 'Асимметрия'
                ]
                
                # Форматирование чисел
                stats['Значение'] = stats['Значение'].apply(
                    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
                )
                
                # Отображение в двух колонках
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown("**Статистика по целевой переменной:**")
                    st.dataframe(
                        stats,
                        use_container_width=True,
                        hide_index=True,
                        height=250
                    )
                
                with cols[1]:
                    try:
                        st.markdown("**Распределение целевой переменной:**")
                        fig = px.histogram(
                            st.session_state.filtered_df,
                            x=st.session_state.target_col,
                            nbins=50,
                            height=320
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Ошибка построения гистограммы: {str(e)}")

            # Анализ временной шкалы
            st.markdown("---")
            st.markdown("**⏰ Анализ временных меток**")
            time_container = st.container(border=True)
            with time_container:
                if pd.api.types.is_datetime64_any_dtype(st.session_state.filtered_df[st.session_state.date_col]):
                    dates = st.session_state.filtered_df[st.session_state.date_col]
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Первая дата", dates.min().strftime('%d.%m.%Y'))
                    with cols[1]:
                        st.metric("Последняя дата", dates.max().strftime('%d.%m.%Y'))
                    with cols[2]:
                        delta = dates.max() - dates.min()
                        st.metric("Период покрытия", f"{delta.days} дней")
                    with cols[3]:
                        freq = pd.infer_freq(dates)
                        freq_map = {
                            'D': 'Дневная',
                            'M': 'Месячная',
                            'Y': 'Годовая',
                            'H': 'Почасовая',
                            None: 'Не определена'
                        }
                        st.metric("Частота", freq_map.get(freq, freq))
                    
                    # Проверка пропусков дат
                    try:
                        full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
                        missing = full_range.difference(dates)
                        st.warning(f"Обнаружено {len(missing)} пропущенных временных меток") if len(missing) > 0 else st.success("Пропущенные даты отсутствуют")
                    except:
                        st.error("Ошибка при проверке временного ряда")
                      
        with tab2:
            st.write("### 📈 Визуализация данных в реальном времени")

            if st.session_state.original_missing is not None:
                plot_df = st.session_state.filtered_df.copy()
                date_col = st.session_state.date_col
                target_col = st.session_state.target_col
                
                try:
                    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                    plot_df = plot_df.sort_values(date_col)
                    
                    # Определяем заполненные пропуски
                    plot_df['filled'] = np.where(
                        st.session_state.original_missing & ~plot_df[target_col].isna(),
                        plot_df[target_col],
                        np.nan
                    )
                    
                    # Создаем основной график
                    fig = px.line(
                        plot_df,
                        x=date_col,
                        y=target_col,
                        title=f"Динамика {target_col}",
                        labels={date_col: "Дата", target_col: "Значение"},
                        line_shape='linear',
                    )
                    
                    # Добавляем красные маркеры для заполненных значений
                    if not plot_df['filled'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=plot_df[date_col],
                            y=plot_df['filled'],
                            mode='markers',
                            marker=dict(
                                color='red',
                                size=6,
                                line=dict(width=1, color='darkred')
                            ),
                            name='Заполненные пропуски',
                            hoverinfo='y'
                        ))
                    
                    # Настройки легенды
                    fig.update_layout(
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка построения графика: {str(e)}")

            # Общие элементы управления
            total_missing = st.session_state.filtered_df[st.session_state.target_col].isnull().sum()
            st.markdown(f"**Текущие пропуски:** `{total_missing}`")
            
            # Кнопка отмены
            if st.session_state.preprocessing_history:
                if st.button("⏪ Сбросить все изменения"):
                    st.session_state.filtered_df = st.session_state.original_filtered_df.copy()
                    st.session_state.preprocessing_history = [st.session_state.original_filtered_df.copy()]
                    st.rerun()

            # Методы обработки
            st.write("### 🧩 Методы обработки пропусков")
            target_column = st.session_state.target_col
            
            def apply_method(method_func):
                """Универсальный обработчик методов"""
                try:
                    # Сброс к исходному состоянию
                    current_df = st.session_state.original_filtered_df.copy()
                    
                    # Применение метода
                    processed = method_func(current_df[target_column])
                    
                    # Обновление состояния
                    st.session_state.filtered_df = current_df
                    st.session_state.filtered_df[target_column] = processed
                    st.session_state.preprocessing_history = [st.session_state.original_filtered_df.copy()]
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

            # Интерполяция
            with st.expander("📈 Методы интерполяции"):
                method = st.selectbox("Тип интерполяции", 
                    options=['linear', 'time', 'spline', 'nearest'],
                    format_func=lambda x: {
                        'linear': 'Линейная',
                        'time': 'Временная',
                        'spline': 'Сплайновая',
                        'nearest': 'Ближайшая'
                    }[x])
                
                if st.button("Применить интерполяцию"):
                    def interpolate(series):
                        return series.interpolate(method=method, order=3 if method == 'spline' else None)
                    
                    apply_method(interpolate)

            # Статистические методы
            with st.expander("📊 Статистические методы"):
                stat_method = st.radio("Метод заполнения",
                    options=['mean', 'median', 'ffill', 'bfill', 'zero'],
                    format_func=lambda x: {
                        'mean': 'Среднее значение',
                        'median': 'Медиана',
                        'ffill': 'Последнее известное',
                        'bfill': 'Следующее известное',
                        'zero': 'Нулевое значение'
                    }[x])
                
                if st.button("Применить выбранный метод"):
                    def fill_na(series):
                        if stat_method in ['mean', 'median']:
                            return series.fillna(series.__getattribute__(stat_method)())
                        elif stat_method == 'zero':
                            return series.fillna(0)
                        return series.fillna(method=stat_method)
                    
                    apply_method(fill_na)

            #TODO: Машинное обучение
            with st.expander("🤖 Прогнозирование (ARIMA)"):
                if st.button("Прогнозировать пропуски с ARIMA"):
                    def arima_fill(series):
                        from statsmodels.tsa.arima.model import ARIMA
                        model = ARIMA(series.dropna(), order=(1,1,1))
                        model_fit = model.fit()
                        return model_fit.predict(start=series.first_valid_index(), 
                                               end=series.last_valid_index())
                    
                    apply_method(arima_fill)
    else:
        st.warning("Пожалуйста, загрузите данные на первом шаге")