"""Visualization functions for supplied portfolio series and weights."""
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def plot_cumulative_returns(original: pd.Series, optimized: pd.Series) -> plt.Figure:
    """Plot cumulative growth of original and optimized test portfolios."""
    figure, axis = plt.subplots()
    (1 + original).cumprod().plot(ax=axis, label="Original")
    (1 + optimized).cumprod().plot(ax=axis, label="Optimized")
    axis.set(title="Test-period cumulative returns", ylabel="Growth of $1")
    axis.legend()
    figure.tight_layout()
    return figure

def plot_allocation_bar(assets: tuple[str, ...], weights: tuple[float, ...]) -> plt.Figure:
    """Plot portfolio allocations as a bar chart."""
    figure, axis = plt.subplots()
    axis.bar(assets, weights)
    axis.set(title="Optimized portfolio allocation", ylabel="Weight", ylim=(0, 1))
    figure.tight_layout()
    return figure

def plot_allocation_pie(assets: tuple[str, ...], weights: tuple[float, ...]) -> plt.Figure:
    """Plot portfolio allocations as a pie chart."""
    figure, axis = plt.subplots()
    axis.pie(weights, labels=assets, autopct="%1.1f%%")
    axis.set_title("Optimized portfolio allocation")
    figure.tight_layout()
    return figure

def plot_weights_history(weights_history: pd.DataFrame) -> plt.Figure:
    """Plot the history of portfolio weights over time."""
    figure, axis = plt.subplots()
    weights_history.plot(ax=axis)
    axis.set(title="Portfolio weights history", ylabel="Weight")
    axis.legend(title="Asset")
    figure.tight_layout()
    return figure

def plot_covariance_heatmap(covariance_matrix: pd.DataFrame) -> plt.Figure:
    """Plot a heatmap of the covariance matrix."""
    figure, axis = plt.subplots()
    cax = axis.matshow(covariance_matrix, cmap="coolwarm")
    figure.colorbar(cax)
    axis.set_xticks(range(len(covariance_matrix.columns)))
    axis.set_yticks(range(len(covariance_matrix.index)))
    axis.set_xticklabels(covariance_matrix.columns, rotation=90)
    axis.set_yticklabels(covariance_matrix.index)
    axis.set_title("Covariance Matrix Heatmap")
    figure.tight_layout()
    return figure