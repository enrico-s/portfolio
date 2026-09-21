"""Momentum alpha model."""

from __future__ import annotations

import pandas as pd

from .base import AlphaModel


class MomentumAlpha(AlphaModel):
    """Use trailing compounded returns as expected-return scores."""

    def __init__(self, lookback_period: int = 63) -> None:
        if lookback_period < 1:
            raise ValueError("lookback_period must be positive.")
        self.lookback_period = lookback_period
        self._scores: pd.Series | None = None

    def fit(self, returns: pd.DataFrame) -> "MomentumAlpha":
        """Fit scores from the latest observations in the training data."""
        if returns.empty:
            raise ValueError("Alpha estimation requires non-empty returns.")
        window = returns.tail(self.lookback_period)
        self._scores = (1.0 + window).prod() - 1.0
        return self

    def predict(self) -> pd.Series:
        """Return fitted momentum scores."""
        if self._scores is None:
            raise RuntimeError("Call fit before predict.")
        return self._scores.copy()
