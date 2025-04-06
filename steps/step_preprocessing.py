import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
from plotly.subplots import make_subplots
import numpy as np

def run_step():
    st.subheader("Шаг 2. Предобработка данных")
    
    if st.session_state.filtered_df is not None:
        tab1, tab2, tab3 = st.tabs(["Описание данных", "Обработка пропусков", "Декомпозиция временного ряда"])
        
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
                dates = pd.to_datetime(st.session_state.filtered_df[st.session_state.date_col])
                #dates = st.session_state.filtered_df[st.session_state.date_col]
                
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
                    st.session_state.freq = freq
                    st.metric("Частота", freq_map.get(freq, freq))
                
                # Проверка пропусков дат
                try:
                    full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
                    missing = full_range.difference(dates)
                    st.warning(f"Обнаружено {len(missing)} пропущенных временных меток") if len(missing) > 0 else st.success("Пропущенные даты отсутствуют")
                except:
                    st.error("Ошибка при проверке временного ряда")
                      
        with tab2:
            st.write("### 📈 Визуализация данных")
            
            # Инициализация original_missing
            if 'original_missing' not in st.session_state:
                st.session_state.original_missing = None

            if 'filtered_df' in st.session_state and st.session_state.filtered_df is not None:
                plot_df = st.session_state.filtered_df.copy()
                date_col = st.session_state.date_col
                target_col = st.session_state.target_col

                try:
                    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                    plot_df = plot_df.sort_values(date_col)
                    
                    # Сохраняем информацию о первоначальных пропусках
                    if st.session_state.original_missing is None:
                        st.session_state.original_missing = plot_df[target_col].isna().copy()
                    
                    # Создание графика с подсветкой заполненных пропусков
                    fig = px.line(
                        plot_df,
                        x=date_col,
                        y=target_col,
                        title=f"Динамика {target_col}",
                        labels={date_col: "Дата", target_col: "Значение"},
                        line_shape='linear',
                    )

                    # Определение заполненных пропусков
                    filled_mask = st.session_state.original_missing & ~plot_df[target_col].isna()
                    current_missing = plot_df[target_col].isna()

                    # Добавление маркеров для заполненных значений
                    if filled_mask.any():
                        fig.add_trace(go.Scatter(
                            x=plot_df[date_col][filled_mask],
                            y=plot_df[target_col][filled_mask],
                            mode='markers',
                            marker=dict(
                                color='red',
                                size=6,
                                line=dict(width=1, color='darkred')
                            ),
                            name='Заполненные пропуски',
                            hoverinfo='y'
                        ))

                    fig.update_layout(
                        hovermode="x unified",
                        height=500,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ошибка построения графика: {str(e)}")

            # Блок управления временными метками
            st.write("### 🕒 Управление временными метками")
            
            with st.expander("⚙️ Обработка дубликатов временных меток", expanded=True):
                if st.session_state.filtered_df[st.session_state.date_col].duplicated().any():
                    st.warning("Обнаружены дубликаты дат!")
                    agg_strategy = st.radio(
                        "Стратегия объединения:",
                        options=['mean', 'last', 'first', 'sum', 'max', 'min'],
                        format_func=lambda x: {
                            'mean': 'Среднее значение',
                            'last': 'Последнее значение', 
                            'first': 'Первое значение',
                            'sum': 'Сумма значений',
                            'max': 'Максимальное значение',
                            'min': 'Минимальное значение'
                        }[x],
                        horizontal=True
                    )
                    
                    if st.button("Устранить дубликаты"):
                        dedup_df = st.session_state.filtered_df.groupby(
                            st.session_state.date_col, 
                            as_index=False
                        ).agg({
                            st.session_state.target_col: agg_strategy
                        })
                        
                        keep_cols = [st.session_state.date_col, st.session_state.target_col]
                        st.session_state.filtered_df = dedup_df[keep_cols]
                        st.rerun()
                else:
                    st.info("Дубликаты временных меток не обнаружены")

            with st.expander("🔄 Восстановление временного ряда"):
                try:
                    # Конвертируем даты в обоих DataFrame
                    st.session_state.filtered_df[st.session_state.date_col] = pd.to_datetime(
                        st.session_state.filtered_df[st.session_state.date_col]
                    )
                    
                    dates = st.session_state.filtered_df[st.session_state.date_col]
                    full_range = pd.date_range(
                        start=dates.min(), 
                        end=dates.max(), 
                        freq=st.session_state.freq
                    )
                    missing_dates = full_range.difference(dates)
                    
                    if len(missing_dates) > 0:
                        st.markdown(f"**Обнаружено пропущенных дат:** {len(missing_dates)}")                
                        if st.button("Добавить недостающие даты"):
                            # Создаем DataFrame с правильным типом даты
                            new_index_df = pd.DataFrame({
                                st.session_state.date_col: pd.to_datetime(full_range)
                            })

                            # Конвертируем даты в исходном DataFrame
                            filtered_df = st.session_state.filtered_df.copy()
                            filtered_df[st.session_state.date_col] = pd.to_datetime(
                                filtered_df[st.session_state.date_col]
                            )

                            # Выполняем объединение
                            merged_df = pd.merge(
                                new_index_df,
                                filtered_df,
                                on=st.session_state.date_col,
                                how='left'
                            )

                            # Восстанавливаем порядок колонок
                            st.session_state.filtered_df = merged_df[filtered_df.columns.tolist()]
                            st.rerun()
                    else:
                        st.info("Пропущенные временные метки не обнаружены")
                        
                except Exception as e:
                    st.error(f"Ошибка обработки временного ряда: {str(e)}")

            # Обновленный блок обработки пропусков значений
            st.write("### 🧩 Методы обработки пропусков значений")
            
            if st.session_state.filtered_df[st.session_state.target_col].isna().sum() > 0:
                current_missing = st.session_state.filtered_df[st.session_state.target_col].isna()
                st.markdown(f"**Текущие пропуски:** `{current_missing.sum()}`")
                
                def apply_fill_method(method):
                    try:
                        filled = st.session_state.filtered_df.copy()
                        target_col = st.session_state.target_col
                        
                        if method == 'time':
                            # Убеждаемся, что индекс DatetimeIndex
                            temp_df = filled.set_index(st.session_state.date_col)
                            temp_df[target_col] = temp_df[target_col].interpolate(method='time')
                            filled = temp_df.reset_index()
                        elif method == 'linear':
                            filled[target_col] = filled[target_col].interpolate(method='linear')
                        elif method in ['ffill', 'bfill']:
                            filled[target_col] = filled[target_col].fillna(method=method)
                        elif method == 'mean':
                            filled[target_col] = filled[target_col].fillna(filled[target_col].mean())
                        elif method == 'zero':
                            filled[target_col] = filled[target_col].fillna(0)
                        
                        st.session_state.filtered_df = filled
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Ошибка заполнения: {str(e)}")

                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Автоматические методы:**")
                    auto_method = st.selectbox(
                        "Выберите метод:",
                        options=['linear', 'time', 'ffill', 'bfill', 'mean'],
                        format_func=lambda x: {
                            'linear': 'Линейная интерполяция',
                            'time': 'Временная интерполяция',
                            'ffill': 'Заполнение предыдущим',
                            'bfill': 'Заполнение следующим',
                            'mean': 'Среднее значение'
                        }[x]
                    )
                    if st.button("Применить автоматическое заполнение"):
                        apply_fill_method(auto_method)

                with cols[1]:
                    st.markdown("**Ручное заполнение:**")
                    manual_value = st.number_input("Значение для заполнения", value=0.0)
                    if st.button("Заполнить выбранным значением"):
                        apply_fill_method('zero')
                        
            else:
                st.success("Пропуски в значениях отсутствуют")

            # Улучшенная кнопка сброса
            if st.button("⏪ Сбросить все изменения к исходным данным"):
                st.session_state.filtered_df = st.session_state.original_filtered_df.copy()
                st.session_state.original_missing = st.session_state.original_filtered_df[st.session_state.target_col].isna().copy()
                st.rerun()

            # #TODO: Машинное обучение
            # with st.expander("🤖 Прогнозирование (SARIMAX)"):
            #     st.markdown("**Параметры модели**")
            #     cols = st.columns(3)
            #     with cols[0]:
            #         use_seasonality = st.checkbox(
            #             "Использовать сезонность",
            #             value=hasattr(st.session_state, 'seasonality_type')
            #         )
            #     with cols[1]:
            #         max_order = st.number_input("Макс. порядок", 1, 5, 3)
            #     with cols[2]:
            #         max_iter = st.number_input("Макс. итераций", 50, 200, 100)
                
            #     if st.button("Прогнозировать пропуски с SARIMAX"):
            #         def sarimax_fill(series):
            #             # Определение параметров сезонности
            #             seasonal = False
            #             m = 1
            #             if use_seasonality and hasattr(st.session_state, 'seasonal_period'):
            #                 seasonal = True
            #                 m = st.session_state.seasonal_period
                        
            #             # Построение модели
            #             model = auto_arima(
            #                 series.dropna(),
            #                 seasonal=seasonal,
            #                 m=m,
            #                 max_order=max_order,
            #                 max_iter=max_iter,
            #                 suppress_warnings=True,
            #                 trace=True
            #             )
                        
            #             # Прогнозирование всех значений
            #             pred = model.predict_in_sample()
            #             return pd.Series(pred, index=series.index)
            #         apply_method(sarimax_fill)
        with tab3:
            st.write("### 📉 STL-Декомпозиция временного ряда")
            
            if st.session_state.filtered_df is None:
                st.warning("Данные не загружены!")
                st.stop()

            # Создаем рабочую копию данных
            df = st.session_state.filtered_df.copy()
            date_col = st.session_state.date_col
            target_col = st.session_state.target_col

            try:
                # Проверка наличия колонок
                if date_col not in df.columns or target_col not in df.columns:
                    raise KeyError(f"Колонки {date_col} или {target_col} не найдены")

                # Преобразование даты и установка индекса
                df[date_col] = pd.to_datetime(df[date_col])
                temp_df = df.set_index(date_col).copy()
                
                # Словарь для автоматического определения периода сезонности
                FREQ_TO_PERIOD = {
                    'D': 7,    # daily -> weekly seasonality
                    'W': 52,   # weekly -> yearly seasonality
                    'M': 12,   # monthly -> yearly seasonality
                    'Q': 4,    # quarterly -> yearly seasonality
                    'Y': 1,    # yearly (no seasonality)
                    'H': 24,   # hourly -> daily seasonality
                    None: None # fallback
                }

                # Определение частоты данных
                if 'freq' not in st.session_state:
                    try:
                        inferred_freq = pd.infer_freq(temp_df.index)
                        st.session_state.freq = inferred_freq[0] if inferred_freq else None
                    except:
                        st.session_state.freq = None

                # Ручная настройка частоты и периода
                if not st.session_state.freq or st.session_state.freq not in FREQ_TO_PERIOD:
                    with st.expander("⚙️ Настройка сезонности", expanded=True):
                        cols = st.columns(2)
                        with cols[0]:
                            new_freq = st.selectbox(
                                "Частота данных:",
                                options=['D', 'W', 'M', 'Q', 'Y', 'H'],
                                index=0
                            )
                        with cols[1]:
                            # Показываем текущий период для выбранной частоты
                            default_period = FREQ_TO_PERIOD[new_freq]
                            custom_period = st.number_input(
                                "Период сезонности:",
                                value=default_period if default_period else 1,
                                min_value=1,
                                step=1,
                                help="Количество наблюдений в одном сезонном цикле"
                            )
                        
                        # Обновляем параметры
                        if st.button("Установить параметры"):
                            st.session_state.freq = new_freq
                            FREQ_TO_PERIOD[new_freq] = custom_period  # Обновляем словарь
                            st.rerun()
                    
                    st.stop()  # Не продолжаем без подтверждения параметров

                # Автоматическое определение периода
                seasonal_period = FREQ_TO_PERIOD[st.session_state.freq]
                if not seasonal_period:
                    st.error("Не удалось определить период сезонности для выбранной частоты")
                    st.stop()

                # Проверка пропусков
                if temp_df[target_col].isnull().sum() > 0:
                    st.warning("Обнаружены пропуски! Заполните их перед декомпозицией.")
                    if st.button("🔄 Заполнить пропуски линейной интерполяцией"):
                        temp_df[target_col] = temp_df[target_col].interpolate(method='linear')
                        st.session_state.filtered_df = temp_df.reset_index()
                        st.success("Пропуски заполнены!")
                        st.rerun()
                    st.stop()

                # Настройки декомпозиции
                with st.expander("⚙️ Параметры STL", expanded=True):
                    cols = st.columns(3)
                    with cols[0]:
                        seasonal_smoothing = st.number_input(
                            "Сезонное сглаживание (seasonal)",
                            value=7,
                            min_value=3,
                            step=2,
                            help="Нечетное число ≥3"
                        )
                    with cols[1]:
                        trend_smoothing = st.number_input(
                            "Трендовое сглаживание (trend)",
                            value=13,
                            min_value=3,
                            step=2,
                            help="Нечетное число ≥3"
                        )
                    with cols[2]:
                        # Динамический расчет low_pass
                        min_low_pass = seasonal_period + (1 if seasonal_period%2 == 0 else 2)
                        low_pass_smoothing = st.number_input(
                            "Низкочастотное сглаживание (low_pass)",
                            value=min_low_pass,
                            min_value=min_low_pass,
                            step=2,
                            help=f"Нечетное число > периода сезонности ({seasonal_period})"
                        )

                # Валидация параметров
                if low_pass_smoothing <= seasonal_period:
                    st.error(f"low_pass ({low_pass_smoothing}) должен быть > периода ({seasonal_period})")
                    st.stop()
                if any([v%2 == 0 for v in [seasonal_smoothing, trend_smoothing, low_pass_smoothing]]):
                    st.error("Все параметры сглаживания должны быть нечетными")
                    st.stop()

                # Выбор типа декомпозиции
                decomposition_type = st.radio(
                    "Тип декомпозиции:",
                    ["additive", "multiplicative"],
                    horizontal=True,
                    help="Для мультипликативной модели используйте логарифмирование"
                )

                # Подготовка временного ряда
                ts = temp_df[target_col].ffill().dropna()
                if decomposition_type == "multiplicative":
                    if (ts <= 0).any():
                        st.error("Мультипликативная модель требует положительных значений")
                        st.stop()
                    ts = np.log(ts)

                # Выполнение STL-декомпозиции
                stl = STL(
                    ts,
                    period=seasonal_period,
                    seasonal=seasonal_smoothing,
                    trend=trend_smoothing,
                    low_pass=low_pass_smoothing,
                    robust=True
                ).fit()

                # Восстановление компонент
                if decomposition_type == "multiplicative":
                    trend = np.exp(stl.trend)
                    seasonal = np.exp(stl.seasonal) - 1
                    resid = np.exp(stl.resid) - 1
                    original = np.exp(ts)
                else:
                    trend = stl.trend
                    seasonal = stl.seasonal
                    resid = stl.resid
                    original = ts

                # Визуализация
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        "Исходный ряд",
                        "Тренд",
                        "Сезонная компонента",
                        "Остатки"
                    )
                )

                components = [
                    (original, 'Исходный ряд', '#1f77b4'),
                    (trend, 'Тренд', '#ff7f0e'),
                    (seasonal, 'Сезонность', '#2ca02c'),
                    (resid, 'Остатки', '#d62728')
                ]

                for i, (data, name, color) in enumerate(components, 1):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data,
                            name=name,
                            line=dict(color=color),
                            showlegend=False
                        ),
                        row=i, col=1
                    )

                fig.update_layout(
                    height=800,
                    margin=dict(l=50, r=50, b=50, t=50),
                    hovermode="x unified"
                )

                # Настройка подписей осей
                yaxis_titles = ["Значение", "Тренд", "Сезонность", "Остатки"]
                for i, title in enumerate(yaxis_titles, 1):
                    fig.update_yaxes(title_text=title, row=i, col=1)
                fig.update_xaxes(title_text="Дата", row=4, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # Дополнительная информация
                with st.expander("📌 Интерпретация компонент"):
                    st.markdown("""
                    - **Тренд**: Долгосрочная направленная динамика ряда
                    - **Сезонность**: Периодические колебания с фиксированной частотой
                    - **Остатки**: Случайная составляющая после удаления тренда и сезонности
                    """)

            except Exception as e:
                st.error(f"Ошибка декомпозиции: {str(e)}")
                st.error("Проверьте параметры сезонности и заполненность данных")
            
    else:
        st.warning("Пожалуйста, загрузите данные на первом шаге")