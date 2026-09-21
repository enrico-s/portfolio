"""Portfolio definition object."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Portfolio:
    """A long-only portfolio with assets and fully-invested weights."""
    assets: Sequence[str]
    weights: Sequence[float]

    def __post_init__(self) -> None:
        assets = tuple(self.assets)
        weights = np.asarray(self.weights, dtype=float)
        if not assets:
            raise ValueError("A portfolio must contain at least one asset.")
        if len(assets) != len(weights):
            raise ValueError("Assets and weights must have the same length.")
        if len(set(assets)) != len(assets):
            raise ValueError("Asset names must be unique.")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be finite, non-negative values.")
        if not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError("Portfolio weights must sum to one.")
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "weights", tuple(weights.tolist()))

    def returns_from(self, asset_returns: pd.DataFrame) -> pd.Series:
        """Calculate the portfolio return series from aligned asset returns."""
        missing = set(self.assets).difference(asset_returns.columns)
        if missing:
            raise ValueError(f"Return data is missing assets: {sorted(missing)}")
        return asset_returns.loc[:, self.assets].dot(np.asarray(self.weights)).rename("portfolio")
    
    def covariance_from(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate the covariance matrix of the portfolio's assets from aligned asset returns."""
        missing = set(self.assets).difference(asset_returns.columns)
        if missing:
            raise ValueError(f"Return data is missing assets: {sorted(missing)}")
        return asset_returns.loc[:, self.assets].cov().rename(columns=dict(zip(self.assets, self.assets)), index=dict(zip(self.assets, self.assets)))
