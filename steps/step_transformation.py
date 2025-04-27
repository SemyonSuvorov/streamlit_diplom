# steps/step_transformation.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import median_abs_deviation, mstats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from state.session import state
from services.transformation_service import TransformationService
from services.data_service import DataService
from components.charts import create_time_series_plot, create_box_plot
from config import TRANSFORMATION_CONFIG

@st.cache_data
def perform_stationarity_tests(series):
    """Выполнение тестов на стационарность"""
    adf_result = adfuller(series.dropna())
    kpss_result = kpss(series.dropna(), regression='c')
    return {
        'ADF Statistic': adf_result[0],
        'ADF p-value': adf_result[1],
        'KPSS Statistic': kpss_result[0],
        'KPSS p-value': kpss_result[1]
    }

def handle_outliers(data, method, replacement_method='median', window_size=5, 
                   interpolation_method='linear', **kwargs):
    """Обработка выбросов с продвинутой заменой"""
    outliers_mask = pd.Series(False, index=data.index)
    
    # Обнаружение выбросов
    if method == 'IQR':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        threshold = kwargs.get('threshold', 1.5)
        outliers_mask = (data < (q1 - threshold*iqr)) | (data > (q3 + threshold*iqr))
    elif method == 'Z-score':
        threshold = kwargs.get('threshold', 3)
        z = np.abs((data - data.mean()) / data.std())
        outliers_mask = z > threshold
    elif method == 'Isolation Forest':
        model = IsolationForest(
            contamination=kwargs.get('contamination', 0.1),
            random_state=42
        )
        preds = model.fit_predict(data.values.reshape(-1, 1))
        outliers_mask = preds == -1
    elif method == 'DBSCAN':
        model = DBSCAN(
            eps=kwargs.get('eps', 0.5),
            min_samples=kwargs.get('min_samples', 5)
        )
        clusters = model.fit_predict(data.values.reshape(-1, 1))
        outliers_mask = clusters == -1
    elif method == 'LOF':
        lof = LocalOutlierFactor(
            n_neighbors=kwargs.get('n_neighbors', 20),
            contamination=kwargs.get('contamination', 0.1)
        )
        preds = lof.fit_predict(data.values.reshape(-1, 1))
        outliers_mask = preds == -1
    elif method == 'Robust Z-score':
        median = data.median()
        mad = median_abs_deviation(data, scale='normal')
        z_scores = np.abs((data - median) / mad)
        outliers_mask = z_scores > kwargs.get('threshold', 3)
    
    # Замена выбросов
    if replacement_method == 'median':
        return data.mask(outliers_mask, data.median())
    elif replacement_method == 'moving_average':
        rolling_mean = data.rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).mean().ffill().bfill()
        return data.mask(outliers_mask, rolling_mean)
    elif replacement_method == 'interpolation':
        temp = data.mask(outliers_mask, np.nan)
        return temp.interpolate(
            method=interpolation_method,
            limit_direction='both',
            limit_area='inside'
        ).ffill().bfill()
    return data

def show_stationarity_tab():
    """Отображение вкладки стационарности"""
    col_original, col_transformed = st.columns([1, 1], gap="large")
    
    with col_original:
        st.markdown("### Исходный временной ряд")
        fig = create_time_series_plot(
            state.get('filtered_df'),
            x_col=state.get('date_col'),
            y_col=state.get('target_col'),
            title="Оригинальные данные"
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        try:
            # Кэшируем результаты тестов
            if 'stationarity_tests' not in st.session_state:
                st.session_state.stationarity_tests = perform_stationarity_tests(
                    state.get('filtered_df').set_index(state.get('date_col'))[state.get('target_col')]
                )
            
            st.markdown("**Результаты тестов стационарности:**")
            adf_col, kpss_col = st.columns(2)
            with adf_col:
                with st.container(border=True):
                    st.markdown("##### Тест Дики-Фуллера (ADF)")
                    st.metric(
                        label="p-value",
                        value=f"{st.session_state.stationarity_tests['ADF p-value']:.4f}",
                        delta="Стационарен" if st.session_state.stationarity_tests['ADF p-value'] < 0.05 else "Нестационарен",
                        delta_color="normal" if st.session_state.stationarity_tests['ADF p-value'] < 0.05 else "off"
                    )
            with kpss_col:
                with st.container(border=True):
                    st.markdown("##### Тест КПСС (KPSS)")
                    st.metric(
                        label="p-value",
                        value=f"{st.session_state.stationarity_tests['KPSS p-value']:.4f}",
                        delta="Стационарен" if st.session_state.stationarity_tests['KPSS p-value'] > 0.05 else "Нестационарен",
                        delta_color="normal" if st.session_state.stationarity_tests['KPSS p-value'] > 0.05 else "off"
                    )
        except Exception as e:
            st.error(f"Ошибка анализа: {str(e)}")

    with col_transformed:
        st.markdown("### Преобразование ряда")
        transform_method = st.selectbox(
            "Выберите метод преобразования:",
            ["Дифференцирование", "Логарифмирование", "Скользящее среднее"],
            index=None,
            placeholder="Выберите метод...",
            key="transform_select"
        )
        
        if transform_method:
            params = {}
            with st.expander("Настройки преобразования", expanded=True):
                if transform_method == "Дифференцирование":
                    params['order'] = st.number_input(
                        "Порядок дифференцирования",
                        min_value=1, max_value=3, value=1,
                        help="Рекомендуется начинать с 1-го порядка",
                        key="diff_order"
                    )
                elif transform_method == "Скользящее среднее":
                    params['window'] = st.number_input(
                        "Размер окна",
                        min_value=2, max_value=90, value=7,
                        help="Выбирайте в зависимости от сезонности данных",
                        key="window_size"
                    )
            
            if st.button("Применить преобразование", type="primary", key="apply_transform"):
                try:
                    if state.get('stationarity_initial') is None:
                        state.set('stationarity_initial', state.get('filtered_df').copy())
                    
                    ts = state.get('filtered_df').set_index(state.get('date_col'))[state.get('target_col')].copy()
                    
                    if transform_method == "Дифференцирование":
                        transformed = ts.diff(params['order']).dropna()
                    elif transform_method == "Логарифмирование":
                        if (ts > 0).all():
                            transformed = np.log(ts)
                        else:
                            st.error("Логарифмирование невозможно: есть неположительные значения")
                            transformed = ts
                    elif transform_method == "Скользящее среднее":
                        transformed = ts.rolling(params['window']).mean().dropna()
                    
                    state.set('filtered_df', transformed.reset_index())
                    # Очищаем кэш тестов стационарности
                    if 'stationarity_tests' in st.session_state:
                        del st.session_state.stationarity_tests
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка преобразования: {str(e)}")

        if state.get('stationarity_initial') is not None:
            if st.button("Отменить все преобразования", type="secondary", key="revert_transform"):
                state.set('filtered_df', state.get('stationarity_initial').copy())
                state.reset('stationarity_initial')
                # Очищаем кэш тестов стационарности
                if 'stationarity_tests' in st.session_state:
                    del st.session_state.stationarity_tests
                st.rerun()

@st.cache_data
def create_time_features(df, date_col, feature_type):
    """Создание временных характеристик с кэшированием"""
    return TransformationService.create_features(df, date_col, feature_type)

def show_features_tab():
    """Отображение вкладки создания признаков"""
    st.markdown("## 🛠️ Создание признаков временного ряда")
    
    st.info("""
    **Доступные методы:**
    - Временные характеристики
    - Скользящие статистики
    - Динамические показатели
    """)
    
    with st.expander("➕ Добавить новые признаки", expanded=True):
        feature_type = st.selectbox(
            "Тип признака:",
            [
                "Скользящее среднее", 
                "Скользящее стандартное отклонение",
                "Разница", 
                "Процентное изменение",
                "Месяц",
                "Квартал",
                "День недели",
                "День месяца",
                "День года",
                "Неделя года"
            ],
            index=None,
            placeholder="Выберите тип признака...",
            key="feature_type"
        )
        
        if feature_type:
            try:
                ts = state.get('filtered_df').set_index(state.get('date_col'))[state.get('target_col')]
                new_feature = None
                params = {}
                
                if feature_type == "Скользящее среднее":
                    window = st.number_input("Размер окна", 2, 180, 7,
                                           help="Количество точек для усреднения",
                                           key="rolling_window")
                    min_periods = st.number_input("Минимальные периоды", 1, window, 1,
                                                key="min_periods")
                    new_feature = ts.rolling(window=window, min_periods=min_periods).mean()
                    params = {'window': window, 'min_periods': min_periods}
                    
                elif feature_type == "Скользящее стандартное отклонение":
                    window = st.number_input("Размер окна", 2, 180, 7,
                                           key="std_window")
                    min_periods = st.number_input("Минимальные периоды", 1, window, 2,
                                                key="std_min_periods")
                    new_feature = ts.rolling(window=window, min_periods=min_periods).std()
                    params = {'window': window, 'min_periods': min_periods}
                    
                elif feature_type == "Разница":
                    diff_order = st.number_input("Порядок разности", 1, 5, 1,
                                               key="diff_order_input")
                    new_feature = ts.diff(diff_order)
                    params = {'order': diff_order}
                    
                elif feature_type == "Процентное изменение":
                    periods = st.number_input("Периоды", 1, 30, 1,
                                            key="pct_change_periods")
                    new_feature = ts.pct_change(periods)
                    params = {'periods': periods}

                elif feature_type in ["Месяц", "Квартал", "День недели", "День месяца", "День года", "Неделя года"]:
                    try:
                        # Проверяем наличие столбца с датой
                        date_col = state.get('date_col')
                        if date_col not in state.get('filtered_df').columns:
                            st.error(f"Столбец с датой '{date_col}' не найден в данных")
                            return
                            
                        # Создаем копию датафрейма с переименованным столбцом даты
                        temp_df = state.get('filtered_df').copy()
                        
                        # Преобразуем даты в datetime
                        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                        
                        # Переименовываем столбец
                        temp_df = temp_df.rename(columns={date_col: 'date'})
                        
                        # Проверяем преобразование
                        new_feature = TransformationService.create_features(
                            temp_df,
                            state.get('target_col'),
                            feature_type,
                            date_col='date'
                        )
                        params = {'component': feature_type.lower().replace(' ', '_')}
                    except Exception as e:
                        st.error(f"Ошибка создания признака: {str(e)}")
                        st.error(f"Доступные столбцы: {state.get('filtered_df').columns.tolist()}")
                        st.error(f"Используемый столбец даты: {state.get('date_col')}")
                        st.error(f"Тип данных в столбце даты: {type(state.get('filtered_df')[state.get('date_col')].iloc[0])}")
                        return

                if new_feature is not None:
                    feature_name = f"{feature_type.lower().replace(' ', '_')}"
                    if params:
                        feature_name += "_" + "_".join(map(str, params.values()))
                    feature_name = feature_name.replace(" ", "_")
                    
                    if st.button("Добавить признак", key="add_feature"):
                        if state.get('features_initial') is None:
                            state.set('features_initial', state.get('filtered_df').copy())
                        
                        df = state.get('filtered_df').copy()
                        df[feature_name] = new_feature.values
                        state.set('filtered_df', df)
                        st.success(f"Признак {feature_name} успешно добавлен!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Ошибка создания признака: {str(e)}")

    st.markdown("---")
    st.markdown("### Визуализация новых признаков")
    
    if len(state.get('filtered_df').columns) > 2:
        selected_feature = st.selectbox(
            "Выберите признак для отображения:",
            [col for col in state.get('filtered_df').columns if col not in [state.get('date_col'), state.get('target_col')]],
            index=None,
            key="feature_visualization"
        )
        
        if selected_feature:
            fig = go.Figure()
            
            # Add original series with a distinct color
            fig.add_trace(go.Scatter(
                x=state.get('filtered_df')[state.get('date_col')],
                y=state.get('filtered_df')[state.get('target_col')],
                name='Исходный ряд',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add feature with a contrasting color and different line style
            fig.add_trace(go.Scatter(
                x=state.get('filtered_df')[state.get('date_col')],
                y=state.get('filtered_df')[selected_feature],
                name=selected_feature,
                line=dict(color='#ff7f0e', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title=f"Сравнение с исходным рядом",
                height=400,
                margin=dict(t=80, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Добавьте хотя бы один признак для визуализации")

    if state.get('features_initial') is not None:
        if st.button("⏪ Отменить все созданные признаки", key="reset_features"):
            state.set('filtered_df', state.get('features_initial').copy())
            state.reset('features_initial')
            st.rerun()

@st.cache_data
def detect_outliers(data, method, **kwargs):
    """Обнаружение выбросов с кэшированием"""
    return TransformationService.detect_outliers(data, method, **kwargs)

@st.cache_data
def replace_outliers(data, outliers_mask, method, **kwargs):
    """Замена выбросов с кэшированием"""
    return TransformationService.replace_outliers(data, outliers_mask, method, **kwargs)

def calculate_outlier_statistics(data, outliers_mask):
    """Расчет статистики по выбросам"""
    stats = {
        'outliers_count': outliers_mask.sum(),
        'outliers_percentage': (outliers_mask.sum() / len(data)) * 100,
        'std_before': data.std(),
        'mean_before': data.mean(),
        'std_after': data[~outliers_mask].std(),
        'mean_after': data[~outliers_mask].mean()
    }
    return stats

def create_statistics_table(before_stats, after_stats):
    """Создание таблицы со статистикой"""
    data = {
        'Метрика': ['Количество выбросов', 'Процент выбросов', 'Стандартное отклонение', 'Среднее значение'],
        'До обработки': [
            before_stats['outliers_count'],
            f"{before_stats['outliers_percentage']:.2f}%",
            f"{before_stats['std_before']:.2f}",
            f"{before_stats['mean_before']:.2f}"
        ],
        'После обработки': [
            after_stats['outliers_count'],
            f"{after_stats['outliers_percentage']:.2f}%",
            f"{after_stats['std_after']:.2f}",
            f"{after_stats['mean_after']:.2f}"
        ],
        'Изменение': [
            f"{after_stats['outliers_count'] - before_stats['outliers_count']}",
            f"{after_stats['outliers_percentage'] - before_stats['outliers_percentage']:.2f}%",
            f"{after_stats['std_after'] - before_stats['std_before']:.2f}",
            f"{after_stats['mean_after'] - before_stats['mean_before']:.2f}"
        ]
    }
    return pd.DataFrame(data)

def show_outliers_tab():
    """Отображение вкладки выбросов и скейлинга"""
    st.markdown("## Обработка выбросов и масштабирование")
    
    # Инициализация session_state для статистики
    if 'outlier_stats' not in st.session_state:
        st.session_state.outlier_stats = {}
    
    # Разделение на подтабы
    analysis_tab, processing_tab, scaling_tab = st.tabs(["Анализ выбросов", "Обработка выбросов", "Масштабирование"])
    
    with analysis_tab:
        st.markdown("### Анализ выбросов")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            numeric_cols = state.get('filtered_df').select_dtypes(include=np.number).columns.tolist()
            selected_column = st.selectbox(
                "Выберите столбец для анализа:",
                numeric_cols,
                index=numeric_cols.index(state.get('target_col')) if state.get('target_col') in numeric_cols else 0,
                key="analysis_column"
            )
            
            # Получаем данные для анализа
            data = state.get('filtered_df')[selected_column]

            method = st.selectbox(
                "Метод обнаружения:",
                ["IQR", "Z-score", "Isolation Forest", "DBSCAN", "LOF", "Robust Z-score"],
                key="analysis_method"
            )
            
            detection_params = {}
            if method in ['IQR', 'Z-score', 'Robust Z-score']:
                detection_params['threshold'] = st.slider(
                    "Пороговое значение",
                    1.0, 5.0, 3.0 if method == 'Z-score' else 1.5, 0.1,
                    help="Стандартные значения: IQR (1.5-3), Z-score (2.5-3)",
                    key="analysis_threshold"
                )
            elif method == 'Isolation Forest':
                detection_params['contamination'] = st.slider(
                    "Contamination",
                    0.01, 0.5, 0.1, 0.01,
                    help="Ожидаемая доля выбросов в данных",
                    key="analysis_contamination"
                )
            elif method == 'DBSCAN':
                detection_params['eps'] = st.slider(
                    "EPS",
                    0.1, 2.0, 0.5, 0.1,
                    help="Максимальное расстояние между соседями",
                    key="analysis_eps"
                )
                detection_params['min_samples'] = st.number_input(
                    "Min Samples",
                    1, 20, 5,
                    help="Минимальное количество точек для формирования кластера",
                    key="analysis_min_samples"
                )
            elif method == 'LOF':
                detection_params['n_neighbors'] = st.number_input(
                    "Количество соседей",
                    5, 50, 20,
                    help="Большие значения лучше для больших наборов данных",
                    key="analysis_n_neighbors"
                )
                detection_params['contamination'] = st.slider(
                    "Contamination",
                    0.01, 0.5, 0.1, 0.01,
                    key="analysis_lof_contamination"
                )
            
            if st.button("Проанализировать выбросы", key="analyze_outliers"):
                try:
                    outliers_mask = detect_outliers(data, method=method, **detection_params)
                    
                    # Сохраняем результаты анализа
                    st.session_state.outlier_stats[selected_column] = {
                        'method': method,
                        'params': detection_params,
                        'mask': outliers_mask,
                        'stats': calculate_outlier_statistics(data, outliers_mask)
                    }
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка анализа: {str(e)}")
        
        with col2:
            if selected_column in st.session_state.outlier_stats:
                stats = st.session_state.outlier_stats[selected_column]['stats']
                outliers_mask = st.session_state.outlier_stats[selected_column]['mask']
                
                # Отображаем статистику в виде метрик
                col_metrics = st.columns(2)
                
                with col_metrics[0]:
                    st.metric(
                        "Количество выбросов",
                        stats['outliers_count'],
                        help="Общее количество обнаруженных выбросов"
                    )
                    st.metric(
                        "Стандартное отклонение",
                        f"{stats['std_before']:.2f}",
                        help="Мера разброса данных"
                    )
                
                with col_metrics[1]:
                    st.metric(
                        "Процент выбросов",
                        f"{stats['outliers_percentage']:.2f}%",
                        help="Доля выбросов в общем количестве данных"
                    )
                    st.metric(
                        "Среднее значение",
                        f"{stats['mean_before']:.2f}",
                        help="Среднее значение данных"
                    )
                
                # Создаем вкладки для разных типов визуализации
                viz_tab1, viz_tab2 = st.tabs(["Распределение", "Временной ряд"])
                
                with viz_tab1:
                    # Визуализация распределения
                    fig = go.Figure()
                    
                    # Добавляем основной график
                    fig.add_trace(go.Histogram(
                        x=data,
                        name='Нормальные значения',
                        nbinsx=50,
                        opacity=0.8,
                        marker_color='#1f77b4'
                    ))
                    
                    # Добавляем выбросы другим цветом
                    fig.add_trace(go.Histogram(
                        x=data[outliers_mask],
                        name='Выбросы',
                        nbinsx=50,
                        opacity=0.8,
                        marker_color='#ff7f0e'
                    ))
                    
                    fig.update_layout(
                        title=f"Распределение данных в столбце {selected_column}",
                        xaxis_title="Значение",
                        yaxis_title="Количество",
                        barmode='overlay',
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab2:
                    # Линейный график с выделением выбросов
                    fig_line = go.Figure()
                    
                    # Создаем массив индексов для оси X
                    x_values = np.arange(len(data))
                    
                    # Добавляем основную линию
                    fig_line.add_trace(go.Scatter(
                        x=x_values,
                        y=data,
                        mode='lines',
                        name='Временной ряд',
                        line=dict(
                            color='#4c78a8',  # Более приглушенный синий
                            width=1
                        )
                    ))
                    
                    # Добавляем выбросы точками
                    fig_line.add_trace(go.Scatter(
                        x=x_values[outliers_mask],
                        y=data[outliers_mask],
                        mode='markers',
                        name='Выбросы',
                        marker=dict(
                            color='#e377c2',  # Более приглушенный розовый
                            size=6,  # Уменьшаем размер точек
                            opacity=0.7,  # Уменьшаем прозрачность
                            line=dict(
                                color='#d62728',
                                width=1
                            )
                        )
                    ))
                    
                    # Добавляем линии для среднего и стандартного отклонения
                    mean = data.mean()
                    std = data.std()
                    
                    fig_line.add_hline(
                        y=mean,
                        line_dash="dash",
                        line_color="#17becf",  # Более приглушенный бирюзовый
                        annotation_text="Среднее",
                        annotation_position="top right"
                    )
                    
                    fig_line.add_hline(
                        y=mean + 2*std,
                        line_dash="dot",
                        line_color="#bcbd22",  # Более приглушенный желто-зеленый
                        annotation_text="+2σ",
                        annotation_position="top right"
                    )
                    
                    fig_line.add_hline(
                        y=mean - 2*std,
                        line_dash="dot",
                        line_color="#bcbd22",  # Более приглушенный желто-зеленый
                        annotation_text="-2σ",
                        annotation_position="top right"
                    )
                    
                    fig_line.update_layout(
                        title=f"Временной ряд с выделением выбросов",
                        xaxis_title="Индекс",
                        yaxis_title="Значение",
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
    
    with processing_tab:
        st.markdown("### Обработка выбросов")
        
        if not st.session_state.outlier_stats:
            st.warning("Сначала выполните анализ выбросов на вкладке 'Анализ выбросов'")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_column = st.selectbox(
                "Выберите столбец для обработки:",
                list(st.session_state.outlier_stats.keys()),
                key="processing_column"
            )
            
            replacement_method = st.selectbox(
                "Способ замены:",
                ["median", "moving_average", "interpolation"],
                format_func=lambda x: {
                    "median": "Медиана",
                    "moving_average": "Скользящее среднее",
                    "interpolation": "Интерполяция"
                }[x],
                key="processing_method"
            )
            
            replacement_params = {}
            if replacement_method == 'moving_average':
                replacement_params['window_size'] = st.number_input(
                    "Размер окна",
                    min_value=3,
                    max_value=31,
                    value=7,
                    step=2,
                    help="Рекомендуется нечетное число для симметричного окна",
                    key="processing_window_size"
                )
            elif replacement_method == 'interpolation':
                replacement_params['interpolation_method'] = st.selectbox(
                    "Метод интерполяции",
                    ['linear', 'time', 'quadratic', 'cubic'],
                    format_func=lambda x: {
                        'linear': 'Линейная',
                        'time': 'Временная',
                        'quadratic': 'Квадратичная',
                        'cubic': 'Кубическая'
                    }[x],
                    key="processing_interpolation_method"
                )
            
            if st.button("Применить обработку", key="apply_processing"):
                try:
                    if state.get('outliers_initial') is None:
                        state.set('outliers_initial', state.get('filtered_df').copy())
                    
                    df = state.get('filtered_df').copy()
                    stats = st.session_state.outlier_stats[selected_column]
                    
                    # Обработка выбросов
                    df[selected_column] = replace_outliers(
                        df[selected_column],
                        stats['mask'],
                        method=replacement_method,
                        **replacement_params
                    )
                    
                    # Анализ после обработки
                    new_outliers_mask = detect_outliers(
                        df[selected_column],
                        method=stats['method'],
                        **stats['params']
                    )
                    new_stats = calculate_outlier_statistics(df[selected_column], new_outliers_mask)
                    
                    # Сохраняем результаты обработки
                    st.session_state.outlier_stats[selected_column]['processed_stats'] = new_stats
                    state.set('filtered_df', df)
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка обработки: {str(e)}")
            
            if state.get('outliers_initial') is not None:
                if st.button("Отменить обработку", type="secondary", key="revert_processing"):
                    state.set('filtered_df', state.get('outliers_initial').copy())
                    state.reset('outliers_initial')
                    st.session_state.outlier_stats = {}
                    st.rerun()
        
        with col2:
            if selected_column in st.session_state.outlier_stats and 'processed_stats' in st.session_state.outlier_stats[selected_column]:
                stats = st.session_state.outlier_stats[selected_column]['stats']
                new_stats = st.session_state.outlier_stats[selected_column]['processed_stats']
                
                # Создаем и отображаем таблицу со статистикой
                stats_df = create_statistics_table(stats, new_stats)
                st.markdown("### Сравнение статистики")
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Визуализация до и после
                fig = go.Figure()
                
                # До обработки
                fig.add_trace(go.Histogram(
                    x=state.get('outliers_initial')[selected_column],
                    name='До обработки',
                    nbinsx=50,
                    opacity=0.7
                ))
                
                # После обработки
                fig.add_trace(go.Histogram(
                    x=state.get('filtered_df')[selected_column],
                    name='После обработки',
                    nbinsx=50,
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title=f"Сравнение распределений до и после обработки",
                    xaxis_title="Значение",
                    yaxis_title="Количество",
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

    with scaling_tab:
        show_scaling_tab()

def show_scaling_tab():
    """Отображение вкладки масштабирования данных"""
    st.markdown("### Масштабирование данных")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        numeric_cols = state.get('filtered_df').select_dtypes(include=np.number).columns.tolist()
        scale_cols = st.multiselect(
            "Выберите столбцы для масштабирования:",
            numeric_cols,
            default=[state.get('target_col')] if state.get('target_col') in numeric_cols else [],
            key="scale_cols"
        )
        
        scale_method = st.selectbox(
            "Метод масштабирования:",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"],
            format_func=lambda x: {
                "StandardScaler": "StandardScaler",
                "MinMaxScaler": "MinMaxScaler",
                "RobustScaler": "RobustScaler"
            }[x],
            index=0,
            key="scale_method"
        )
        
        if st.button("Применить масштабирование", key="apply_scaling"):
            if scale_cols:
                try:
                    if state.get('scaling_initial') is None:
                        state.set('scaling_initial', state.get('filtered_df').copy())
                        
                        df = state.get('filtered_df').copy()
                        df = TransformationService.scale_data(
                            df,
                            scale_cols,
                            method=scale_method
                        )
                        state.set('filtered_df', df)
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка масштабирования: {str(e)}")

        if state.get('scaling_initial') is not None:
            if st.button("Отменить масштабирование", type="secondary", key="revert_scaling"):
                state.set('filtered_df', state.get('scaling_initial').copy())
                state.reset('scaling_initial')
                st.rerun()

    with col2:
        if scale_cols:
            try:
                # Создаем таблицу со статистикой до и после масштабирования
                stats_data = []
                for col in scale_cols:
                    if state.get('scaling_initial') is not None:
                        before_data = state.get('scaling_initial')[col]
                        after_data = state.get('filtered_df')[col]
                        
                        stats_data.append({
                            'Столбец': col,
                            'До масштабирования': {
                                'Среднее': f"{before_data.mean():.2f}",
                                'Стандартное отклонение': f"{before_data.std():.2f}",
                                'Минимум': f"{before_data.min():.2f}",
                                'Максимум': f"{before_data.max():.2f}"
                            },
                            'После масштабирования': {
                                'Среднее': f"{after_data.mean():.2f}",
                                'Стандартное отклонение': f"{after_data.std():.2f}",
                                'Минимум': f"{after_data.min():.2f}",
                                'Максимум': f"{after_data.max():.2f}"
                            }
                        })
                
                # Отображаем статистику
                for stat in stats_data:
                    st.markdown(f"#### {stat['Столбец']}")
                    col_before, col_after = st.columns(2)
                    
                    with col_before:
                        st.markdown("**До масштабирования:**")
                        for key, value in stat['До масштабирования'].items():
                            st.metric(key, value)
                    
                    with col_after:
                        st.markdown("**После масштабирования:**")
                        for key, value in stat['После масштабирования'].items():
                            st.metric(key, value)
                    
                    # Визуализация распределения
                    fig = go.Figure()
                    
                    # До масштабирования
                    fig.add_trace(go.Histogram(
                        x=state.get('scaling_initial')[stat['Столбец']],
                        name='До масштабирования',
                        nbinsx=50,
                        opacity=0.7
                    ))
                    
                    # После масштабирования
                    fig.add_trace(go.Histogram(
                        x=state.get('filtered_df')[stat['Столбец']],
                        name='После масштабирования',
                        nbinsx=50,
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title=f"Распределение {stat['Столбец']} до и после масштабирования",
                        xaxis_title="Значение",
                        yaxis_title="Количество",
                        barmode='overlay',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Ошибка визуализации: {str(e)}")

def run_step():
    """Запуск шага трансформации данных"""
    st.subheader("Шаг 3. Преобразование данных и анализ")
    
    if state.get('filtered_df') is None or state.get('filtered_df').empty:
        st.warning("Данные не загружены!")
        return

    if state.get('trans_initial') is None:
        state.set('trans_initial', state.get('filtered_df').copy())
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Стационарность",
        "Признаки", 
        "Выбросы",
        "Масштабирование"
    ])

    with tab1:
        show_stationarity_tab()
    with tab2:
        show_features_tab()
    with tab3:
        show_outliers_tab()
    with tab4:
        show_scaling_tab()