# Portfolio Construction Framework

A modular framework for constructing and evaluating portfolios without
look-ahead bias.

```text
Returns -> Alpha model -> Risk model -> Optimizer -> Portfolio -> Backtest
```

- `alpha.AlphaModel` produces expected returns or scores.
- `risk.RiskModel` produces a covariance matrix.
- `optimization.Optimizer` produces a `Portfolio` from alpha, covariance, and constraints.
- `backtest.Backtest` performs a single train/test evaluation or rolling walk-forward re-optimisation.

Version 1 includes equal, historical-mean, and momentum alpha models; a historical covariance risk model; maximum-Sharpe and maximum-expected-return optimisation; allocation constraints; and metrics.

## Setup and example

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/run_portfolio.py
```

The example uses momentum alpha, historical covariance, and a 60% maximum allocation per asset. It trains before 1 January 2023 and evaluates later returns only.

## Rolling re-optimization

Omit `split_date` to use a walk-forward backtest. Every rebalance uses only the preceding `lookback_period` observations; the selected weights apply only to following returns.

```python
results = Backtest(
    portfolio=portfolio,
    returns=returns,
    alpha_model=MomentumAlpha(lookback_period=126),
    risk_model=HistoricalCovarianceRiskModel(),
    optimizer=Optimizer(
        objective="maximum_sharpe",
        constraints=PortfolioConstraints(max_weight=0.60),
    ),
    lookback_period=252,
    rebalance_frequency=21,
).run()

print(results.optimized_test_metrics.as_dict())
print(results.weight_history)
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```
