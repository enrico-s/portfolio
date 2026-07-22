"""Run the Version 1 portfolio optimizer."""
from pathlib import Path
import sys

# Support ``python examples/run_portfolio.py`` from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from backtest.backtest import Backtest
from data.loader import load_returns
from portfolio.portfolio import Portfolio

def main() -> None:
    portfolio = Portfolio(["AMZN", "META", "TSLA"], [0.5, 0.3, 0.2])
    returns = load_returns(portfolio.assets, start="2015-01-01")
    results = Backtest(portfolio, returns, split_date="2023-01-01").run()
    print("Original test metrics:", results.original_test_metrics.as_dict())
    print("Optimized test metrics:", results.optimized_test_metrics.as_dict())
    print("Optimized weights:", dict(zip(portfolio.assets, results.optimized_portfolio.weights)))
    results.plot()
    plt.show()

if __name__ == "__main__":
    main()
