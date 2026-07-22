"""Extensible portfolio optimization routines."""
from __future__ import annotations
from enum import StrEnum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from portfolio.metrics import TRADING_DAYS

class OptimizationObjective(StrEnum):
    """Objectives supported by the optimizer."""
    MAXIMUM_SHARPE = "maximum_sharpe"

def optimize_weights(training_returns: pd.DataFrame, objective: OptimizationObjective | str = OptimizationObjective.MAXIMUM_SHARPE, risk_free_rate: float = 0.0) -> np.ndarray:
    """Return long-only fully-invested weights optimized on training data only."""
    objective = OptimizationObjective(objective)
    returns = training_returns.dropna(how="any")
    if returns.empty:
        raise ValueError("Optimization requires non-empty training returns.")
    daily_risk_free = risk_free_rate / TRADING_DAYS
    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_returns = returns.to_numpy() @ weights
        volatility = portfolio_returns.std(ddof=1)
        if volatility == 0 or not np.isfinite(volatility):
            return 1e6
        return -float((portfolio_returns.mean() - daily_risk_free) / volatility)
    if objective is not OptimizationObjective.MAXIMUM_SHARPE:
        raise NotImplementedError(f"Unsupported objective: {objective}")
    asset_count = returns.shape[1]
    result = minimize(negative_sharpe, x0=np.full(asset_count, 1.0 / asset_count), method="SLSQP", bounds=[(0.0, 1.0)] * asset_count, constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0})
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return result.x
