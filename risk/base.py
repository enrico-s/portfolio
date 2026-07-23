"""Base interface for risk models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class RiskModel(ABC):
    """Estimate a covariance matrix from historical returns."""

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "RiskModel":
        """Fit the model using only returns available at the rebalance date."""

    @abstractmethod
    def covariance(self) -> pd.DataFrame:
        """Return the fitted asset covariance matrix."""
