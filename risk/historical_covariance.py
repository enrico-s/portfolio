"""Historical covariance risk model."""

from __future__ import annotations

import pandas as pd

from portfolio.metrics import TRADING_DAYS

from .base import RiskModel


class HistoricalCovarianceRiskModel(RiskModel):
    """Estimate annualized covariance from historical returns."""

    def __init__(self, annualization_factor: int = TRADING_DAYS) -> None:
        self.annualization_factor = annualization_factor
        self._covariance: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame) -> "HistoricalCovarianceRiskModel":
        """Fit the sample covariance matrix from the training window."""
        if len(returns) < 2:
            raise ValueError("Covariance estimation requires two observations.")
        self._covariance = returns.cov() * self.annualization_factor
        return self

    def covariance(self) -> pd.DataFrame:
        """Return the fitted annualized covariance matrix."""
        if self._covariance is None:
            raise RuntimeError("Call fit before covariance.")
        return self._covariance.copy()
