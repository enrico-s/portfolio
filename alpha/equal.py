"""Equal alpha model."""

from __future__ import annotations

import pandas as pd

from .base import AlphaModel


class EqualAlpha(AlphaModel):
    """Assign the same expected-return score to every asset."""

    def __init__(self) -> None:
        self._assets: pd.Index | None = None

    def fit(self, returns: pd.DataFrame) -> "EqualAlpha":
        """Store asset names from the training data."""
        if returns.empty:
            raise ValueError("Alpha estimation requires non-empty returns.")
        self._assets = returns.columns
        return self

    def predict(self) -> pd.Series:
        """Return equal scores for the fitted assets."""
        if self._assets is None:
            raise RuntimeError("Call fit before predict.")
        return pd.Series(1.0, index=self._assets, name="expected_return")
