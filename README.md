# Portfolio Optimizer (Version 1)

A modular Python tool that compares an initial long-only portfolio with a maximum-Sharpe portfolio chosen strictly from historical training data.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python examples/run_portfolio.py
```

The example downloads SPY, QQQ, and GLD returns with QuantStats, optimizes only observations before 1 January 2023, then evaluates both portfolios on the later test period.

## Test

```bash
pytest
```

`data` obtains return data; `portfolio` validates portfolios and calculates metrics; `optimization` contains the extensible maximum-Sharpe routine; `backtest` coordinates the no-look-ahead workflow; and `plotting` renders charts.
