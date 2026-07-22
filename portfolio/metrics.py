"""Performance metrics for portfolio return series."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

TRADING_DAYS = 252

@dataclass(frozen=True)
class PortfolioMetrics:
    cumulative_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    def as_dict(self) -> dict[str, float]:
        """Return metrics as a plain dictionary."""
        return asdict(self)

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> PortfolioMetrics:
    """Calculate standard annualized metrics from a daily return series."""
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty or not np.isfinite(returns).all():
        raise ValueError("Metrics require finite return observations.")
    growth = (1.0 + returns).cumprod()
    cumulative_return = float(growth.iloc[-1] - 1.0)
    annual_return = float(growth.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess = returns - risk_free_rate / TRADING_DAYS
    excess_volatility = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = float((annual_return - risk_free_rate) / excess_volatility) if excess_volatility else np.nan
    downside_deviation = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = float((annual_return - risk_free_rate) / downside_deviation) if downside_deviation else np.nan
    drawdown = growth / growth.cummax() - 1.0
    return PortfolioMetrics(cumulative_return, annual_return, annual_volatility, sharpe, sortino, float(drawdown.min()))
