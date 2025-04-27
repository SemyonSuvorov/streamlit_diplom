"""
Компоненты для визуализации
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import VISUALIZATION_CONFIG
import pandas as pd

def create_time_series_plot(
    df,
    x_col: str,
    y_col: str,
    title: str = None,
    height: int = None,
    show_legend: bool = True
):
    """
    Создание графика временного ряда
    
    Args:
        df: DataFrame с данными
        x_col: Название столбца с датой
        y_col: Название столбца с целевой переменной
        title: Заголовок графика
        height: Высота графика
        show_legend: Показывать ли легенду
    """
    if height is None:
        height = VISUALIZATION_CONFIG['chart_height']
    
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        height=height
    )
    
    fig.update_layout(
        margin=VISUALIZATION_CONFIG['chart_margin'],
        showlegend=show_legend,
        hovermode="x unified"
    )
    
    return fig

def create_stl_plot(
    components: dict,
    title: str = "STL-Декомпозиция временного ряда"
):
    """
    Создание графика STL-декомпозиции
    
    Args:
        components: Словарь с компонентами декомпозиции
        title: Заголовок графика
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Исходный ряд", "Тренд", "Сезонная компонента", "Остатки")
    )
    
    components_data = [
        (components['original'], 'Исходный ряд', '#1f77b4'),
        (components['trend'], 'Тренд', '#ff7f0e'),
        (components['seasonal'], 'Сезонность', '#2ca02c'),
        (components['resid'], 'Остатки', '#d62728')
    ]
    
    for i, (data, name, color) in enumerate(components_data, 1):
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
        hovermode="x unified",
        title=dict(text=title, y=0.95)
    )
    
    yaxis_titles = ["Значение", "Тренд", "Сезонность", "Остатки"]
    for i, title in enumerate(yaxis_titles, 1):
        fig.update_yaxes(title_text=title, row=i, col=1)
    fig.update_xaxes(title_text="Дата", row=4, col=1)
    
    return fig

def create_box_plot(
    df,
    x_col: str,
    y_col: str,
    title: str = None,
    height: int = None,
    period: str = None
):
    """
    Создание box-plot
    
    Args:
        df: DataFrame с данными
        x_col: Название столбца для группировки
        y_col: Название столбца с значениями
        title: Заголовок графика
        height: Высота графика
        period: Период группировки (День, Неделя, Месяц, Год)
    """
    if height is None:
        height = VISUALIZATION_CONFIG['chart_height']
    
    fig = go.Figure()
    
    # Convert to datetime if not already
    df[x_col] = pd.to_datetime(df[x_col])
    
    # Group by period if specified
    if period:
        freq_map = {"День": "D", "Неделя": "W", "Месяц": "M", "Год": "Y"}
        df['period'] = df[x_col].dt.to_period(freq_map[period])
        grouped = df.groupby('period')
    else:
        grouped = df.groupby(x_col)
    
    for name, group in grouped:
        if not group.empty and y_col in group.columns:
            fig.add_trace(go.Box(
                y=group[y_col],
                name=str(name),
                boxpoints='outliers'
            ))
    
    fig.update_layout(
        height=height,
        title=dict(text=title, y=0.95),
        margin=VISUALIZATION_CONFIG['chart_margin'],
        showlegend=False
    )
    
    return fig

def create_histogram(
    df,
    x_col: str,
    title: str = None,
    height: int = None,
    nbins: int = 50
):
    """
    Создание гистограммы
    
    Args:
        df: DataFrame с данными
        x_col: Название столбца для гистограммы
        title: Заголовок графика
        height: Высота графика
        nbins: Количество бинов
    """
    if height is None:
        height = VISUALIZATION_CONFIG['chart_height']
    
    fig = px.histogram(
        df,
        x=x_col,
        nbins=nbins,
        height=height,
        title=title
    )
    
    fig.update_layout(
        margin=VISUALIZATION_CONFIG['chart_margin']
    )
    
    return fig 