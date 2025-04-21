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
    """
    Обработка выбросов с продвинутой заменой
    """
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

def run_step():
    st.subheader("Шаг 3. Преобразование данных и анализ выбросов")
    
    if 'filtered_df' not in st.session_state or st.session_state.filtered_df.empty:
        st.warning("Данные не загружены!")
        return

    if 'trans_initial' not in st.session_state:
        st.session_state.trans_initial = st.session_state.filtered_df.copy()
    
    df = st.session_state.filtered_df.copy()
    date_col = st.session_state.date_col
    target_col = st.session_state.target_col
    
    tab1, tab2 = st.tabs(["📈 Стационарность", "📊 Выбросы"])
    
    with tab1:
        st.markdown("## 📈 Анализ и преобразование временного ряда")
        col_original, col_transformed = st.columns([1, 1], gap="large")
        
        with col_original:
            st.markdown("### Исходный временной ряд")
            fig = px.line(df, x=date_col, y=target_col, height=350,
                        title="Оригинальные данные",
                        color_discrete_sequence=['#1f77b4'])
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            try:
                original_tests = perform_stationarity_tests(df.set_index(date_col)[target_col])
                st.markdown("**Результаты тестов стационарности:**")
                adf_col, kpss_col = st.columns(2)
                with adf_col:
                    with st.container(border=True):
                        st.markdown("##### Тест Дики-Фуллера (ADF)")
                        st.metric(
                            label="p-value",
                            value=f"{original_tests['ADF p-value']:.4f}",
                            delta="Стационарен" if original_tests['ADF p-value'] < 0.05 else "Нестационарен",
                            delta_color="normal" if original_tests['ADF p-value'] < 0.05 else "off"
                        )
                with kpss_col:
                    with st.container(border=True):
                        st.markdown("##### Тест КПСС (KPSS)")
                        st.metric(
                            label="p-value",
                            value=f"{original_tests['KPSS p-value']:.4f}",
                            delta="Стационарен" if original_tests['KPSS p-value'] > 0.05 else "Нестационарен",
                            delta_color="normal" if original_tests['KPSS p-value'] > 0.05 else "off"
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
                            help="Рекомендуется начинать с 1-го порядка"
                        )
                    elif transform_method == "Скользящее среднее":
                        params['window'] = st.number_input(
                            "Размер окна",
                            min_value=2, max_value=90, value=7,
                            help="Выбирайте в зависимости от сезонности данных"
                        )
                
                if st.button("Применить преобразование", type="primary"):
                    try:
                        st.session_state.stationarity_initial = st.session_state.filtered_df.copy()
                        ts = st.session_state.stationarity_initial.set_index(date_col)[target_col].copy()
                        
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
                        
                        st.session_state.filtered_df = transformed.reset_index()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка преобразования: {str(e)}")

            if 'stationarity_initial' in st.session_state:
                if st.button("Отменить преобразование", type="secondary"):
                    st.session_state.filtered_df = st.session_state.stationarity_initial.copy()
                    del st.session_state.stationarity_initial
                    st.rerun()

    with tab2:
        st.markdown("## 📊 Расширенная обработка выбросов")
        
        st.info("""
        **Советы по настройке параметров:**
        - Для методов на основе расстояний (DBSCAN, LOF) масштабируйте данные
        - Contamination: доля ожидаемых выбросов в данных (0.01-0.5)
        - Размер окна: выбирайте нечетные значения для симметричного сглаживания
        - Интерполяция: используйте 'time' для временных рядов с неравными интервалами
        """)

        col_detection, col_replacement = st.columns([1, 1], gap="large")
        
        with col_detection:
            st.markdown("### Обнаружение выбросов")
            method = st.selectbox(
                "Метод обнаружения:",
                ["IQR", "Z-score", "Isolation Forest", "DBSCAN", "LOF", "Robust Z-score"],
                key="outlier_method"
            )
            
            detection_params = {}
            if method in ['IQR', 'Z-score', 'Robust Z-score']:
                detection_params['threshold'] = st.slider(
                    "Пороговое значение",
                    1.0, 5.0, 3.0 if method == 'Z-score' else 1.5, 0.1,
                    help="Стандартные значения: IQR (1.5-3), Z-score (2.5-3)"
                )
            elif method == 'Isolation Forest':
                detection_params['contamination'] = st.slider(
                    "Contamination",
                    0.01, 0.5, 0.1, 0.01,
                    help="Ожидаемая доля выбросов в данных"
                )
            elif method == 'DBSCAN':
                detection_params['eps'] = st.slider(
                    "EPS",
                    0.1, 2.0, 0.5, 0.1,
                    help="Максимальное расстояние между соседями"
                )
                detection_params['min_samples'] = st.number_input(
                    "Min Samples",
                    1, 20, 5,
                    help="Минимальное количество точек для формирования кластера"
                )
            elif method == 'LOF':
                detection_params['n_neighbors'] = st.number_input(
                    "Количество соседей",
                    5, 50, 20,
                    help="Большие значения лучше для больших наборов данных"
                )
                detection_params['contamination'] = st.slider(
                    "Contamination",
                    0.01, 0.5, 0.1, 0.01
                )

        with col_replacement:
            st.markdown("### Замена выбросов")
            replacement_method = st.selectbox(
                "Способ замены:",
                ["median", "moving_average", "interpolation"],
                format_func=lambda x: {
                    "median": "Медиана",
                    "moving_average": "Скользящее среднее",
                    "interpolation": "Интерполяция"
                }[x],
                key="replacement_method"
            )
            
            replacement_params = {}
            if replacement_method == 'moving_average':
                replacement_params['window_size'] = st.number_input(
                    "Размер окна",
                    min_value=3,
                    max_value=31,
                    value=7,
                    step=2,
                    help="Рекомендуется нечетное число для симметричного окна"
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
                    key="interpolation_method",
                    help="'time' учитывает временные интервалы между наблюдениями"
                )

            if st.button("Применить коррекцию", key="apply_outliers"):
                try:
                    st.session_state.outliers_initial = st.session_state.filtered_df.copy()
                    df = st.session_state.outliers_initial.copy()
                    
                    df[target_col] = handle_outliers(
                        df[target_col],
                        method=method,
                        replacement_method=replacement_method,
                        window_size=replacement_params.get('window_size', 5),
                        interpolation_method=replacement_params.get('interpolation_method', 'linear'),
                        **detection_params
                    )
                    
                    st.session_state.filtered_df = df.copy()
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка обработки: {str(e)}")

            if 'outliers_initial' in st.session_state:
                if st.button("Отменить обработку выбросов", type="secondary"):
                    st.session_state.filtered_df = st.session_state.outliers_initial.copy()
                    del st.session_state.outliers_initial
                    st.rerun()

        st.markdown("---")
        st.markdown("### Визуализация распределения")
        period = st.selectbox(
            "Период анализа:",
            ["День", "Неделя", "Месяц", "Год"],
            index=2,
            key="period_select"
        )
        
        try:
            freq_map = {"День": "D", "Неделя": "W", "Месяц": "M", "Год": "Y"}
            grouped = df.groupby(pd.Grouper(key=date_col, freq=freq_map[period]))
            
            fig = go.Figure()
            for name, group in grouped:
                if not group.empty:
                    fig.add_trace(go.Box(
                        y=group[target_col],
                        name=name.strftime('%Y-%m-%d'),
                        boxpoints='outliers'
                    ))
            
            fig.update_layout(
                height=400,
                title="Распределение значений по периодам",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка визуализации: {str(e)}")

    if st.button("⏪ Сбросить все изменения этого шага", key="reset_all"):
        st.session_state.filtered_df = st.session_state.trans_initial.copy()
        st.rerun()