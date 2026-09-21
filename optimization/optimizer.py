"""Portfolio optimizers independent of alpha and risk-model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

from alpha.historical_mean import HistoricalMeanAlpha
from portfolio.portfolio import Portfolio
from risk.historical_covariance import HistoricalCovarianceRiskModel


class OptimizationObjective(StrEnum):
    """Objectives supported by :class:`Optimizer`."""

    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_EXPECTED_RETURN = "maximum_expected_return"
    MINIMUM_VALUE_AT_RISK = "minimum_var"


@dataclass(frozen=True)
class PortfolioConstraints:
    """Long-only allocation constraints applied by an optimizer."""

    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError("Weights must satisfy 0 <= min_weight <= max_weight <= 1.")

    def bounds(self, asset_count: int) -> list[tuple[float, float]]:
        """Return a per-asset bounds list, validating feasibility."""
        if asset_count * self.min_weight > 1 or asset_count * self.max_weight < 1:
            raise ValueError("Weight bounds are infeasible for the number of assets.")
        return [(self.min_weight, self.max_weight)] * asset_count


class Optimizer:
    """Construct portfolios from expected returns, covariance, and constraints."""

    def __init__(
        self,
        objective: OptimizationObjective | str = OptimizationObjective.MAXIMUM_SHARPE,
        constraints: PortfolioConstraints | None = None,
        risk_free_rate: float = 0.0,
        confidence_level: float | None = None,
    ) -> None:
        self.objective = OptimizationObjective(objective)
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> Portfolio:
        """Return optimal weights for aligned alpha and covariance estimates."""
        assets = tuple(expected_returns.index)
        if not assets:
            raise ValueError("Optimization requires at least one asset.")
        if set(assets) != set(covariance.index) or set(assets) != set(covariance.columns):
            raise ValueError("Expected returns and covariance must contain the same assets.")

        expected = expected_returns.loc[list(assets)].to_numpy(dtype=float)
        covariance_values = covariance.loc[list(assets), list(assets)].to_numpy(dtype=float)
        if not np.isfinite(expected).all() or not np.isfinite(covariance_values).all():
            raise ValueError("Alpha and covariance estimates must be finite.")

        objective_function = self._objective_function(expected, covariance_values)
        count = len(assets)
        result = minimize(
            objective_function,
            x0=np.full(count, 1.0 / count),
            method="SLSQP",
            bounds=self.constraints.bounds(count),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        )
        if not result.success:
            raise RuntimeError(f"Weight optimization failed: {result.message}")
        return Portfolio(assets=assets, weights=result.x)

    def _objective_function(
        self, expected_returns: np.ndarray, covariance: np.ndarray
    ):
        if self.objective is OptimizationObjective.MAXIMUM_SHARPE:
            return lambda weights: self.maximum_sharpe(
                weights, expected_returns, covariance
            )
        elif self.objective is OptimizationObjective.MAXIMUM_EXPECTED_RETURN:
            return lambda weights: self.maximum_expected_return(
                weights, expected_returns
            )
        elif self.objective is OptimizationObjective.MINIMUM_VALUE_AT_RISK:
            return lambda weights: self.minimum_var(
                weights, expected_returns, covariance
            )
        else:
            raise ValueError("The portfolio optimization objective is ill-defined.")

    def maximum_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Return negative expected Sharpe ratio for minimisation."""
        volatility = float(np.sqrt(weights @ covariance @ weights))
        if volatility <= 0 or not np.isfinite(volatility):
            return 1e6
        return -float((weights @ expected_returns - self.risk_free_rate) / volatility)

    @staticmethod
    def maximum_expected_return(
        weights: np.ndarray, expected_returns: np.ndarray
    ) -> float:
        """Return negative expected portfolio return for minimisation."""
        return -float(weights @ expected_returns)
    
    def minimum_var(
        self,
        weights: np.ndarray, 
        expected_returns: np.ndarray, 
        covariance: np.ndarray,
    ) -> float:
        """Return value at risk of portfolio for minimisation."""
        volatility = float(np.sqrt(weights @ covariance @ weights))
        if self.confidence_level is None:
            print("Default confidence level of 0.95 used for VaR quantification.")
            self.confidence_level = 0.95
        z_value = stats.norm.ppf(self.confidence_level)

        return -float(weights @ expected_returns - z_value * volatility)



def optimize_weights(
    training_returns: pd.DataFrame,
    objective: OptimizationObjective | str = OptimizationObjective.MAXIMUM_SHARPE,
    risk_free_rate: float = 0.0,
    constraints: PortfolioConstraints | None = None,
) -> np.ndarray:
    """Compatibility wrapper that fits default models before optimization.

    New code should fit alpha and risk models explicitly, then call
    :meth:`Optimizer.optimize`.
    """
    returns = training_returns.dropna(how="any")
    alpha = HistoricalMeanAlpha().fit(returns).predict()
    covariance = HistoricalCovarianceRiskModel().fit(returns).covariance()
    portfolio = Optimizer(objective, constraints, risk_free_rate).optimize(
        alpha, covariance
    )
    return np.asarray(portfolio.weights)
