"""Coordinate portfolio evaluation and training-only optimization."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from optimization.optimizer import OptimizationObjective, optimize_weights
from portfolio.metrics import PortfolioMetrics, calculate_metrics
from portfolio.portfolio import Portfolio

@dataclass(frozen=True)
class BacktestResults:
    """Outputs of a train/test portfolio backtest."""
    original_train_metrics: PortfolioMetrics
    original_test_metrics: PortfolioMetrics
    optimized_test_metrics: PortfolioMetrics
    optimized_portfolio: Portfolio
    original_test_returns: pd.Series
    optimized_test_returns: pd.Series
    def plot(self):
        """Create the Version 1 cumulative-return and allocation charts."""
        from plotting.plots import plot_allocation_bar, plot_allocation_pie, plot_cumulative_returns
        return {"cumulative_returns": plot_cumulative_returns(self.original_test_returns, self.optimized_test_returns), "allocation_bar": plot_allocation_bar(self.optimized_portfolio.assets, self.optimized_portfolio.weights), "allocation_pie": plot_allocation_pie(self.optimized_portfolio.assets, self.optimized_portfolio.weights)}

class Backtest:
    """Run a no-look-ahead train/test portfolio optimization workflow."""
    def __init__(self, portfolio: Portfolio, returns: pd.DataFrame, split_date: str) -> None:
        self.portfolio = portfolio
        self.returns = returns.sort_index().copy()
        self.split_date = pd.Timestamp(split_date)
    def run(self, objective: OptimizationObjective | str = OptimizationObjective.MAXIMUM_SHARPE) -> BacktestResults:
        """Optimize on observations before ``split_date`` and test afterwards."""
        missing = set(self.portfolio.assets).difference(self.returns.columns)
        if missing:
            raise ValueError(f"Return data is missing assets: {sorted(missing)}")
        train = self.returns.loc[self.returns.index < self.split_date, self.portfolio.assets]
        test = self.returns.loc[self.returns.index >= self.split_date, self.portfolio.assets]
        if train.empty or test.empty:
            raise ValueError("Split date must leave non-empty training and test periods.")
        original_train_returns = self.portfolio.returns_from(train)
        original_test_returns = self.portfolio.returns_from(test)
        optimized_portfolio = Portfolio(self.portfolio.assets, optimize_weights(train, objective=objective))
        optimized_test_returns = optimized_portfolio.returns_from(test)
        return BacktestResults(calculate_metrics(original_train_returns), calculate_metrics(original_test_returns), calculate_metrics(optimized_test_returns), optimized_portfolio, original_test_returns, optimized_test_returns)
