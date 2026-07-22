"""Functions for obtaining historical asset returns."""
from __future__ import annotations
from collections.abc import Sequence
import pandas as pd
import quantstats as qs

def load_returns(tickers: Sequence[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Download daily returns for *tickers* with QuantStats."""
    if not tickers:
        raise ValueError("At least one ticker is required.")
    series = []
    for ticker in tickers:
        returns = qs.utils.download_returns(ticker, period="max")
        returns.name = ticker
        series.append(returns)
    frame = pd.concat(series, axis=1).sort_index().dropna(how="any")
    if start is not None:
        frame = frame.loc[pd.Timestamp(start):]
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    if frame.empty:
        raise ValueError("No return data is available for the requested period.")
    return frame
