import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


def plot_price_distribution(df):
    """Plot distribution of house prices"""
    fig = px.histogram(
        df, x='SalePrice', nbins=50,
        title='Phân phối giá nhà',
        labels={'SalePrice': 'Giá nhà ($)', 'count': 'Số lượng'}
    )
    fig.update_layout(
        xaxis_title='Giá nhà ($)',
        yaxis_title='Số lượng',
        template='plotly_white'
    )
    return fig


def plot_price_vs_feature(df, feature, title=None):
    """Plot scatter plot of price vs. a feature"""
    if df[feature].dtype in ['int64', 'float64']:
        fig = px.scatter(
            df, x=feature, y='SalePrice',
            title=title or f'Giá nhà theo {feature}',
            labels={feature: feature, 'SalePrice': 'Giá nhà ($)'}
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title='Giá nhà ($)',
            template='plotly_white'
        )
    else:  # Categorical feature
        fig = px.box(
            df, x=feature, y='SalePrice',
            title=title or f'Giá nhà theo {feature}',
            labels={feature: feature, 'SalePrice': 'Giá nhà ($)'}
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title='Giá nhà ($)',
            template='plotly_white',
            xaxis={'categoryorder': 'median descending'}
        )
    return fig


def plot_correlation_matrix(df, features):
    """Plot correlation matrix of selected features"""
    corr_df = df[features + ['SalePrice']].corr()

    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Ma trận tương quan giữa các đặc trưng'
    )
    fig.update_layout(
        width=800,
        height=800
    )
    return fig


def plot_prediction_vs_actual(y_true, y_pred):
    """Plot actual vs predicted values"""
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={'x': 'Giá thực tế ($)', 'y': 'Giá dự đoán ($)'},
        title='Giá dự đoán so với giá thực tế'
    )

    # Add the perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Dự đoán hoàn hảo'
        )
    )

    fig.update_layout(
        xaxis_title='Giá thực tế ($)',
        yaxis_title='Giá dự đoán ($)',
        template='plotly_white'
    )
    return fig


def get_neighborhood_stats(df):
    """Get statistics of house prices by neighborhood"""
    neighborhood_stats = df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'median', 'std', 'count'])
    neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=False)
    return neighborhood_stats


def plot_neighborhood_prices(df):
    """Plot average house prices by neighborhood"""
    neighborhood_stats = get_neighborhood_stats(df)

    fig = px.bar(
        neighborhood_stats.reset_index(),
        x='Neighborhood',
        y='mean',
        error_y='std',
        labels={'mean': 'Giá trung bình ($)', 'Neighborhood': 'Khu vực'},
        title='Giá nhà trung bình theo khu vực'
    )
    fig.update_layout(
        xaxis_title='Khu vực',
        yaxis_title='Giá trung bình ($)',
        template='plotly_white',
        xaxis={'categoryorder': 'total descending'}
    )
    return fig


def format_price(price):
    """Format price to display with thousand separators"""
    return f"${price:,.2f}"