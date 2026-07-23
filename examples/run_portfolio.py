"""Run a train/test portfolio-construction example."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

# Support ``python examples/run_portfolio.py`` from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha.momentum import MomentumAlpha
from backtest.backtest import Backtest
from data.loader import load_returns
from optimization.optimizer import Optimizer, PortfolioConstraints
from portfolio.portfolio import Portfolio
from risk.historical_covariance import HistoricalCovarianceRiskModel
from optimization.optimizer import OptimizationObjective

def main() -> None:
    stock_symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AVGO", "META", "TSLA", "MU", "LLY", "JPM", "AMD", "XOM", "JNJ", "ABBV"]
    initial_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    portfolio = Portfolio(assets=stock_symbols, weights=initial_weights)
    returns = load_returns(portfolio.assets, start="2018-01-01")
    print(returns.head())
    results = Backtest(
        portfolio=portfolio,
        returns=returns,
        split_date="2023-01-03",
        lookback_period=21*12,
        alpha_model=MomentumAlpha(lookback_period=21*12),
        risk_model=HistoricalCovarianceRiskModel(),
        optimizer=Optimizer(objective=OptimizationObjective.MAXIMUM_EXPECTED_RETURN, constraints=PortfolioConstraints()),
        mode="rebalance",
        rebalance_frequency=21*3,
    ).run()
    print("Original test metrics:", results.original_test_metrics.as_dict())
    print("Optimized test metrics:", results.optimized_test_metrics.as_dict())
    print("Optimized weights:", dict(zip(portfolio.assets, results.optimized_portfolio.weights)))
    print("Backtest Results Table:")
    print(results.table())
    results.plot()
    plt.show()


if __name__ == "__main__":
    main()
