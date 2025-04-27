# steps/step_preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from state.session import state
from services.preprocessing_service import PreprocessingService
from services.data_service import DataService
from components.charts import create_time_series_plot, create_stl_plot, create_box_plot, create_histogram
from config import PREPROCESSING_CONFIG

def show_data_description():
    """Отображение описательной статистики"""
    st.write("### 📊 Описательная статистика")
    info_container = st.container(border=True)
    with info_container:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Столбцов", len(state.get('filtered_df').columns))
        with cols[1]:
            st.metric("Наблюдений", len(state.get('filtered_df')))
        with cols[2]:
            st.metric("Пропусков", state.get('filtered_df').isnull().sum().sum())
        with cols[3]:
            st.metric("Дубликаты", state.get('filtered_df').duplicated().sum())

    if state.get('target_col') in state.get('filtered_df').select_dtypes(include='number'):
        stats = state.get('filtered_df')[state.get('target_col')].agg([
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
                fig = create_histogram(
                    state.get('filtered_df'),
                    x_col=state.get('target_col'),
                    title=f"Распределение {state.get('target_col')}",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка построения гистограммы: {str(e)}")

    st.markdown("---")
    st.markdown("**⏰ Анализ временных меток**")
    time_container = st.container(border=True)
    with time_container:
        dates = pd.to_datetime(state.get('filtered_df')[state.get('date_col')])
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
            freq_map = {'D': 'Дневная', 'ME': 'Месячная', 'Y': 'Годовая', 'H': 'Почасовая', None: 'Не определена'}
            state.set('freq', freq)
            st.metric("Частота", freq_map.get(freq, freq))
            
        try:
            full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
            missing = full_range.difference(dates)
            st.warning(f"Обнаружено {len(missing)} пропущенных временных меток") if len(missing) > 0 else st.success("Пропущенные даты отсутствуют")
        except:
            st.error("Ошибка при проверке временного ряда")

def show_missing_values_tab():
    """Отображение вкладки обработки пропусков"""
    st.write("### 📈 Визуализация данных")
    if state.get('original_missing') is None:
        state.set('original_missing', None)

    if state.get('filtered_df') is not None:
        plot_df = state.get('filtered_df').copy()
        date_col = state.get('date_col')
        target_col = state.get('target_col')

        try:
            plot_df[date_col] = pd.to_datetime(plot_df[date_col])
            plot_df = plot_df.sort_values(date_col)
            
            if state.get('original_missing') is None:
                state.set('original_missing', plot_df[target_col].isna().copy())
            
            fig = create_time_series_plot(
                plot_df,
                x_col=date_col,
                y_col=target_col,
                title=f"Динамика {target_col}"
            )
            
            filled_mask = state.get('original_missing') & ~plot_df[target_col].isna()
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
        if state.get('filtered_df')[state.get('date_col')].duplicated().any():
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
                dedup_df = PreprocessingService.handle_duplicates(
                    state.get('filtered_df'),
                    state.get('date_col'),
                    state.get('target_col'),
                    strategy=agg_strategy
                )
                keep_cols = [state.get('date_col'), state.get('target_col')]
                state.set('filtered_df', dedup_df[keep_cols])
                st.rerun()
        else:
            st.info("Дубликаты временных меток не обнаружены")

    with st.expander("🔄 Восстановление временного ряда", expanded=True):
        try:
            dates = pd.to_datetime(state.get('filtered_df')[state.get('date_col')])
            full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=state.get('freq'))
            missing_dates = full_range.difference(dates)
            
            if len(missing_dates) > 0:
                st.markdown(f"**Обнаружено пропущенных дат:** {len(missing_dates)}")                
                if st.button("Добавить недостающие даты"):
                    state.set('filtered_df', PreprocessingService.add_missing_dates(
                        state.get('filtered_df'),
                        state.get('date_col'),
                        state.get('target_col'),
                        freq=state.get('freq')
                    ))
                    st.rerun()
            else:
                st.info("Пропущенные временные метки не обнаружены")
        except Exception as e:
            st.error(f"Ошибка обработки временного ряда: {str(e)}")

    st.write("### 🧩 Методы обработки пропусков значений")
    missing_count = state.get('filtered_df')[state.get('target_col')].isna().sum()
    if missing_count > 0:
        st.warning(f"Обнаружено {missing_count} пропущенных значений в целевой переменной")
        def apply_fill_method(method):
            if state.get('initial_preprocessing_state') is None:
                state.set('initial_preprocessing_state', state.get('filtered_df').copy())
            state.set('filtered_df', PreprocessingService.handle_missing_values(
                state.get('filtered_df'),
                state.get('target_col'),
                method=method
            ))
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
        state.set('filtered_df', state.get('initial_preprocessing_state').copy())
        state.set('original_missing', state.get('initial_preprocessing_state')[state.get('target_col')].isna().copy())
        st.rerun()

def show_stl_decomposition():
    """Отображение STL-декомпозиции"""
    st.write("### 📉 STL-Декомпозиция временного ряда")
    try:
        df = state.get('filtered_df').copy()
        date_col = state.get('date_col')
        target_col = state.get('target_col')

        df[date_col] = pd.to_datetime(df[date_col])
        temp_df = df.set_index(date_col).copy()
        
        FREQ_TO_PERIOD = {'D': 7, 'W': 52, 'ME': 12, 'Q': 4, 'Y': 1, 'H': 24, None: None}
        if state.get('freq') is None:
            try:
                inferred_freq = pd.infer_freq(temp_df.index)
                state.set('freq', inferred_freq[0] if inferred_freq else None)
            except:
                state.set('freq', None)

        if not state.get('freq') or state.get('freq') not in FREQ_TO_PERIOD:
            with st.expander("⚙️ Настройка сезонности", expanded=True):
                cols = st.columns(2)
                with cols[0]:
                    new_freq = st.selectbox("Частота данных:", options=['D', 'W', 'ME', 'Q', 'Y', 'H'], index=0)
                with cols[1]:
                    default_period = FREQ_TO_PERIOD[new_freq]
                    custom_period = st.number_input(
                        "Период сезонности:",
                        value=default_period if default_period else 1,
                        min_value=1,
                        step=1
                    )
                if st.button("Установить параметры"):
                    state.set('freq', new_freq)
                    FREQ_TO_PERIOD[new_freq] = custom_period
                    st.rerun()
            st.stop()

        seasonal_period = FREQ_TO_PERIOD[state.get('freq')]
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

        components = PreprocessingService.perform_stl_decomposition(
            df,
            target_col,
            seasonal_period,
            seasonal_smoothing=seasonal_smoothing,
            trend_smoothing=trend_smoothing,
            low_pass_smoothing=low_pass_smoothing,
            decomposition_type=decomposition_type
        )

        fig = create_stl_plot(components)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка декомпозиции: {str(e)}")

def run_step():
    """Запуск шага предобработки данных"""
    st.subheader("Шаг 2. Предобработка данных")
    
    if state.get('filtered_df') is None:
        st.warning("Пожалуйста, загрузите данные на первом шаге")
        return
        
    if state.get('initial_preprocessing_state') is None:
        state.set('initial_preprocessing_state', state.get('filtered_df').copy())

    tab1, tab2, tab3 = st.tabs(["Описание данных", "Обработка пропусков", "Декомпозиция временного ряда"])
    
    with tab1:
        show_data_description()
    with tab2:
        show_missing_values_tab()
    with tab3:
        show_stl_decomposition()