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
    MAXIMUM_RETURN = "maximum_return"


def maximum_sharpe(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    """Return the negative daily Sharpe ratio for minimisation.

    The negative sign lets ``scipy.optimize.minimize`` maximise the Sharpe
    ratio without changing the optimisation interface.
    """
    portfolio_returns = returns.to_numpy() @ weights
    volatility = portfolio_returns.std(ddof=1)
    if volatility == 0 or not np.isfinite(volatility):
        return 1e6
    daily_risk_free = risk_free_rate / TRADING_DAYS
    return -float((portfolio_returns.mean() - daily_risk_free) / volatility)


def maximum_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Return the negative mean daily portfolio return for minimisation."""
    portfolio_returns = returns.to_numpy() @ weights
    return -float(portfolio_returns.mean())


def optimize_weights(
    training_returns: pd.DataFrame,
    objective: OptimizationObjective | str = OptimizationObjective.MAXIMUM_SHARPE,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Optimize long-only, fully-invested weights using training data only.

    Parameters
    ----------
    training_returns:
        Asset-return observations available before the backtest split date.
    objective:
        Either ``maximum_sharpe`` or ``maximum_return``.
    risk_free_rate:
        Annual risk-free rate, used only by the maximum-Sharpe objective.
    """
    objective = OptimizationObjective(objective)
    returns = training_returns.dropna(how="any")
    if returns.empty:
        raise ValueError("Optimization requires non-empty training returns.")

    objective_functions = {
        OptimizationObjective.MAXIMUM_SHARPE: lambda weights: maximum_sharpe(
            weights, returns, risk_free_rate
        ),
        OptimizationObjective.MAXIMUM_RETURN: lambda weights: maximum_return(
            weights, returns
        ),
    }
    asset_count = returns.shape[1]
    result = minimize(
        objective_functions[objective],
        x0=np.full(asset_count, 1.0 / asset_count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return result.x
