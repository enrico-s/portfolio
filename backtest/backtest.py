"""Walk-forward portfolio backtesting without look-ahead bias."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha.base import AlphaModel
from alpha.historical_mean import HistoricalMeanAlpha
from optimization.optimizer import OptimizationObjective, Optimizer
from portfolio.metrics import PortfolioMetrics, calculate_metrics
from portfolio.portfolio import Portfolio
from risk.base import RiskModel
from risk.historical_covariance import HistoricalCovarianceRiskModel


@dataclass(frozen=True)
class BacktestResults:
    """Results from a single split or a rolling portfolio backtest."""

    original_train_metrics: PortfolioMetrics | None
    original_test_metrics: PortfolioMetrics | None
    optimized_test_metrics: PortfolioMetrics
    optimized_portfolio: Portfolio | None
    original_test_returns: pd.Series | None
    optimized_test_returns: pd.Series
    weight_history: pd.DataFrame | None = None
    covariance_test: pd.DataFrame | None = None

    def plot(self):
        """Create charts for the available backtest results."""
        from plotting.plots import (
            plot_allocation_bar,
            plot_allocation_pie,
            plot_cumulative_returns,
            plot_weights_history,
            plot_covariance_heatmap,
        )

        figures = {}
        if self.original_test_returns is not None:
            figures["cumulative_returns"] = plot_cumulative_returns(
                self.original_test_returns, self.optimized_test_returns
            )
        if self.optimized_portfolio is not None:
            figures["allocation_bar"] = plot_allocation_bar(
                self.optimized_portfolio.assets,
                self.optimized_portfolio.weights,
            )
            figures["allocation_pie"] = plot_allocation_pie(
                self.optimized_portfolio.assets,
                self.optimized_portfolio.weights,
            )
        if self.weight_history is not None:
            figures["weights_history"] = plot_weights_history(self.weight_history)
        
        if self.covariance_test is not None:
            figures["covariance_test"] = plot_covariance_heatmap(self.covariance_test)

        return figures
    
    def table(self) -> pd.DataFrame:
        """Return a summary table of the available backtest results."""
        data = {
            "Original Train": self.original_train_metrics.as_dict()
            if self.original_train_metrics is not None
            else None,
            "Original Test": self.original_test_metrics.as_dict()
            if self.original_test_metrics is not None
            else None,
            "Optimized Test": self.optimized_test_metrics.as_dict(),
        }
        return pd.DataFrame(data)


class Backtest:
    """Coordinate alpha, risk, optimization, and out-of-sample evaluation."""

    def __init__(
        self,
        portfolio: Portfolio,
        returns: pd.DataFrame,
        split_date: str | None = None,
        alpha_model: AlphaModel | None = None,
        risk_model: RiskModel | None = None,
        optimizer: Optimizer | None = None,
        lookback_period: int = 252,
        rebalance_frequency: int = 21,
        mode: str | None = "rebalance",
    ) -> None:
        if lookback_period < 2:
            raise ValueError("lookback_period must be at least two observations.")
        if rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be positive.")
        self.portfolio = portfolio
        self.returns = returns.loc[:, portfolio.assets].sort_index().dropna()
        self.split_date = pd.Timestamp(split_date) if split_date is not None else None
        self.alpha_model = alpha_model or HistoricalMeanAlpha()
        self.risk_model = risk_model or HistoricalCovarianceRiskModel()
        self.optimizer = optimizer or Optimizer()
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.mode = mode

    def run(
        self, objective: OptimizationObjective | str | None = None
    ) -> BacktestResults:
        """Run a legacy one-off split or a rolling, walk-forward backtest."""
        if objective is not None:
            self.optimizer = Optimizer(
                objective=objective,
                constraints=self.optimizer.constraints,
                risk_free_rate=self.optimizer.risk_free_rate,
            )
        if self.returns.empty:
            raise ValueError("Backtesting requires non-empty asset returns.")
        if self.split_date is not None and self.mode == "single_split":
            return self._run_single_split()
        return self._run_rolling()

    def _fit_portfolio(self, training_returns: pd.DataFrame) -> Portfolio:
        expected_returns = self.alpha_model.fit(training_returns).predict()
        covariance = self.risk_model.fit(training_returns).covariance()
        return self.optimizer.optimize(expected_returns, covariance)

    def _run_single_split(self) -> BacktestResults:
        train = self.returns.loc[self.returns.index < self.split_date]
        test = self.returns.loc[self.returns.index >= self.split_date]
        if len(train) < 2 or test.empty:
            raise ValueError("Split date must leave two training observations and test data.")
        optimized = self._fit_portfolio(train)
        original_train = self.portfolio.returns_from(train)
        original_test = self.portfolio.returns_from(test)
        optimized_test = optimized.returns_from(test)
        covariance_test = self.portfolio.covariance_from(test)
        return BacktestResults(
            original_train_metrics=calculate_metrics(original_train),
            original_test_metrics=calculate_metrics(original_test),
            optimized_test_metrics=calculate_metrics(optimized_test),
            optimized_portfolio=optimized,
            original_test_returns=original_test,
            optimized_test_returns=optimized_test,
            weight_history=pd.DataFrame(
                [optimized.weights], index=[self.split_date], columns=optimized.assets
            ),
            covariance_test=covariance_test,
        )

    def _run_rolling(self) -> BacktestResults:
        if len(self.returns) <= self.lookback_period:
            raise ValueError("Returns do not contain enough observations for the lookback period.")
        periods = []
        weights = []
        # rebalance starts from the split date if provided, otherwise from the lookback period
        start_index = (
            self.returns.index.get_loc(self.split_date)
            if self.split_date is not None
            else self.lookback_period
        )
        for rebalance_index in range(
            start_index, len(self.returns), self.rebalance_frequency
        ):
            training = self.returns.iloc[
                rebalance_index - self.lookback_period : rebalance_index
            ]
            end_index = min(rebalance_index + self.rebalance_frequency, len(self.returns))
            holding_returns = self.returns.iloc[rebalance_index:end_index]
            optimized = self._fit_portfolio(training)
            periods.append(optimized.returns_from(holding_returns))
            weights.append(pd.Series(optimized.weights, index=optimized.assets, name=holding_returns.index[0]))
        original_test = self.portfolio.returns_from(self.returns.iloc[start_index:])
        strategy_returns = pd.concat(periods).rename("optimized_portfolio")
        weight_history = pd.DataFrame(weights)
        covariance_test = self.portfolio.covariance_from(self.returns.iloc[start_index:])
        return BacktestResults(
            original_train_metrics=None,
            original_test_metrics=calculate_metrics(original_test),
            optimized_test_metrics=calculate_metrics(strategy_returns),
            optimized_portfolio=optimized,
            original_test_returns=original_test,
            optimized_test_returns=strategy_returns,
            weight_history=weight_history,
            covariance_test=covariance_test,
        )
