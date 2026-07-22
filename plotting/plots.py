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
