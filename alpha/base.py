"""Base interface for alpha models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AlphaModel(ABC):
    """Estimate an expected-return vector from historical returns."""

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "AlphaModel":
        """Fit the model using only returns available at the rebalance date."""

    @abstractmethod
    def predict(self) -> pd.Series:
        """Return one expected-return estimate or score per asset."""
