"""Historical mean-return alpha model."""

from __future__ import annotations

import pandas as pd

from portfolio.metrics import TRADING_DAYS

from .base import AlphaModel


class HistoricalMeanAlpha(AlphaModel):
    """Estimate annual expected returns from historical arithmetic means."""

    def __init__(self, annualization_factor: int = TRADING_DAYS) -> None:
        self.annualization_factor = annualization_factor
        self._expected_returns: pd.Series | None = None

    def fit(self, returns: pd.DataFrame) -> "HistoricalMeanAlpha":
        """Fit annualized mean returns using the supplied training period."""
        if returns.empty:
            raise ValueError("Alpha estimation requires non-empty returns.")
        self._expected_returns = returns.mean() * self.annualization_factor
        return self

    def predict(self) -> pd.Series:
        """Return annual expected returns for the fitted assets."""
        if self._expected_returns is None:
            raise RuntimeError("Call fit before predict.")
        return self._expected_returns.copy()
