# steps/step_preprocessing.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from plotly.subplots import make_subplots
import numpy as np

def run_step():
    st.subheader("Шаг 2. Предобработка данных")
    
    if 'initial_preprocessing_state' not in st.session_state:
        st.session_state.initial_preprocessing_state = st.session_state.filtered_df.copy()
    
    if st.session_state.filtered_df is None:
        st.warning("Пожалуйста, загрузите данные на первом шаге")
        return

    tab1, tab2, tab3 = st.tabs(["Описание данных", "Обработка пропусков", "Декомпозиция временного ряда"])
    
    with tab1:
        st.write("### 📊 Описательная статистика")
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

        if st.session_state.target_col in st.session_state.filtered_df.select_dtypes(include='number'):
            stats = st.session_state.filtered_df[st.session_state.target_col].agg([
                'mean', 'median', 'std', 'min', 'max', 'skew'
            ]).reset_index()
            stats.columns = ['Метрика', 'Значение']
            stats['Метрика'] = [
                'Среднее', 'Медиана', 'Станд. отклонение', 
                'Минимум', 'Максимум', 'Асимметрия'
            ]
            stats['Значение'] = stats['Значение'].apply(
                lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
            )
            
            cols = st.columns([1, 2])
            with cols[0]:
                st.markdown("**Статистика по целевой переменной:**")
                st.dataframe(stats, use_container_width=True, height=250)
            with cols[1]:
                try:
                    fig = px.histogram(
                        st.session_state.filtered_df,
                        x=st.session_state.target_col,
                        nbins=50,
                        height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка построения гистограммы: {str(e)}")

        st.markdown("---")
        st.markdown("**⏰ Анализ временных меток**")
        time_container = st.container(border=True)
        with time_container:
            dates = pd.to_datetime(st.session_state.filtered_df[st.session_state.date_col])
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
                freq_map = {'D': 'Дневная', 'M': 'Месячная', 'Y': 'Годовая', 'H': 'Почасовая', None: 'Не определена'}
                st.session_state.freq = freq
                st.metric("Частота", freq_map.get(freq, freq))
                
            try:
                full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
                missing = full_range.difference(dates)
                st.warning(f"Обнаружено {len(missing)} пропущенных временных меток") if len(missing) > 0 else st.success("Пропущенные даты отсутствуют")
            except:
                st.error("Ошибка при проверке временного ряда")

    with tab2:
        st.write("### 📈 Визуализация данных")
        if 'original_missing' not in st.session_state:
            st.session_state.original_missing = None

        if st.session_state.filtered_df is not None:
            plot_df = st.session_state.filtered_df.copy()
            date_col = st.session_state.date_col
            target_col = st.session_state.target_col

            try:
                plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                plot_df = plot_df.sort_values(date_col)
                
                if st.session_state.original_missing is None:
                    st.session_state.original_missing = plot_df[target_col].isna().copy()
                
                fig = px.line(plot_df, x=date_col, y=target_col, title=f"Динамика {target_col}", line_shape='linear')
                filled_mask = st.session_state.original_missing & ~plot_df[target_col].isna()
                current_missing = plot_df[target_col].isna()

                if filled_mask.any():
                    fig.add_trace(go.Scatter(
                        x=plot_df[date_col][filled_mask],
                        y=plot_df[target_col][filled_mask],
                        mode='markers',
                        marker=dict(color='red', size=6, line=dict(width=1, color='darkred')),
                        name='Заполненные пропуски',
                        hoverinfo='y'
                    ))

                fig.update_layout(hovermode="x unified", height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка построения графика: {str(e)}")

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
                    ).agg({st.session_state.target_col: agg_strategy})
                    keep_cols = [st.session_state.date_col, st.session_state.target_col]
                    st.session_state.filtered_df = dedup_df[keep_cols]
                    st.rerun()
            else:
                st.info("Дубликаты временных меток не обнаружены")

        with st.expander("🔄 Восстановление временного ряда", expanded=True):
            try:
                dates = pd.to_datetime(st.session_state.filtered_df[st.session_state.date_col])
                full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=st.session_state.freq)
                missing_dates = full_range.difference(dates)
                
                if len(missing_dates) > 0:
                    st.markdown(f"**Обнаружено пропущенных дат:** {len(missing_dates)}")                
                    if st.button("Добавить недостающие даты"):
                        new_index_df = pd.DataFrame({st.session_state.date_col: pd.to_datetime(full_range)})
                        filtered_df = st.session_state.filtered_df.copy()
                        filtered_df[st.session_state.date_col] = pd.to_datetime(filtered_df[st.session_state.date_col])
                        merged_df = pd.merge(new_index_df, filtered_df, on=st.session_state.date_col, how='left')
                        st.session_state.filtered_df = merged_df[filtered_df.columns.tolist()]
                        st.rerun()
                else:
                    st.info("Пропущенные временные метки не обнаружены")
            except Exception as e:
                st.error(f"Ошибка обработки временного ряда: {str(e)}")

        st.write("### 🧩 Методы обработки пропусков значений")
        if st.session_state.filtered_df[st.session_state.target_col].isna().sum() > 0:
            def apply_fill_method(method):
                filled = st.session_state.filtered_df.copy()
                target_col = st.session_state.target_col
                if method == 'time':
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

        if st.button("⏪ Сбросить все изменения к исходным данным"):
            st.session_state.filtered_df = st.session_state.initial_preprocessing_state.copy()
            st.session_state.original_missing = st.session_state.initial_preprocessing_state[st.session_state.target_col].isna().copy()
            st.rerun()

    with tab3:
        st.write("### 📉 STL-Декомпозиция временного ряда")
        try:
            df = st.session_state.filtered_df.copy()
            date_col = st.session_state.date_col
            target_col = st.session_state.target_col

            df[date_col] = pd.to_datetime(df[date_col])
            temp_df = df.set_index(date_col).copy()
            
            FREQ_TO_PERIOD = {'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1, 'H': 24, None: None}
            if 'freq' not in st.session_state:
                try:
                    inferred_freq = pd.infer_freq(temp_df.index)
                    st.session_state.freq = inferred_freq[0] if inferred_freq else None
                except:
                    st.session_state.freq = None

            if not st.session_state.freq or st.session_state.freq not in FREQ_TO_PERIOD:
                with st.expander("⚙️ Настройка сезонности", expanded=True):
                    cols = st.columns(2)
                    with cols[0]:
                        new_freq = st.selectbox("Частота данных:", options=['D', 'W', 'M', 'Q', 'Y', 'H'], index=0)
                    with cols[1]:
                        default_period = FREQ_TO_PERIOD[new_freq]
                        custom_period = st.number_input(
                            "Период сезонности:",
                            value=default_period if default_period else 1,
                            min_value=1,
                            step=1
                        )
                    if st.button("Установить параметры"):
                        st.session_state.freq = new_freq
                        FREQ_TO_PERIOD[new_freq] = custom_period
                        st.rerun()
                st.stop()

            seasonal_period = FREQ_TO_PERIOD[st.session_state.freq]
            if not seasonal_period:
                st.error("Не удалось определить период сезонности")
                st.stop()

            with st.expander("⚙️ Параметры STL", expanded=True):
                cols = st.columns(3)
                with cols[0]:
                    seasonal_smoothing = st.number_input("Сезонное сглаживание", value=7, min_value=3, step=2)
                with cols[1]:
                    trend_smoothing = st.number_input("Трендовое сглаживание", value=13, min_value=3, step=2)
                with cols[2]:
                    min_low_pass = seasonal_period + (1 if seasonal_period%2 == 0 else 2)
                    low_pass_smoothing = st.number_input("Низкочастотное сглаживание", value=min_low_pass, min_value=min_low_pass, step=2)

            decomposition_type = st.radio("Тип декомпозиции:", ["additive", "multiplicative"], horizontal=True)

            ts = temp_df[target_col].ffill().dropna()
            if decomposition_type == "multiplicative":
                if (ts <= 0).any():
                    st.error("Мультипликативная модель требует положительных значений")
                    st.stop()
                ts = np.log(ts)

            stl = STL(
                ts,
                period=seasonal_period,
                seasonal=seasonal_smoothing,
                trend=trend_smoothing,
                low_pass=low_pass_smoothing,
                robust=True
            ).fit()

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

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                              subplot_titles=("Исходный ряд", "Тренд", "Сезонная компонента", "Остатки"))

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


            fig.update_layout(height=800, margin=dict(l=50, r=50, b=50, t=50), hovermode="x unified")
            yaxis_titles = ["Значение", "Тренд", "Сезонность", "Остатки"]
            for i, title in enumerate(yaxis_titles, 1):
                fig.update_yaxes(title_text=title, row=i, col=1)
            fig.update_xaxes(title_text="Дата", row=4, col=1)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка декомпозиции: {str(e)}")