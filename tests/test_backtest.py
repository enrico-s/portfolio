import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

import numpy as np
import pandas as pd
from backtest.backtest import Backtest
from portfolio.portfolio import Portfolio

def test_backtest_uses_only_training_data_for_optimization():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    returns = pd.DataFrame({"A": [0.02, 0.01, 0.02, 0.01, -0.9, -0.9, -0.9, -0.9], "B": [-0.02, -0.01, -0.03, -0.02, 0.05, 0.05, 0.05, 0.05]}, index=index)
    result = Backtest(Portfolio(["A", "B"], [0.5, 0.5]), returns, "2020-01-05").run()
    assert result.optimized_portfolio.weights[0] > 0.99
    assert len(result.optimized_test_returns) == 4
    assert np.isfinite(result.optimized_test_metrics.cumulative_return)

if __name__ == "__main__":
    test_backtest_uses_only_training_data_for_optimization()