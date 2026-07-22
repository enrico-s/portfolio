import pandas as pd
import pytest
from portfolio.portfolio import Portfolio

def test_portfolio_calculates_weighted_returns():
    portfolio = Portfolio(["A", "B"], [0.6, 0.4])
    returns = pd.DataFrame({"A": [0.1, 0.0], "B": [0.0, 0.1]})
    assert portfolio.returns_from(returns).tolist() == pytest.approx([0.06, 0.04])

def test_portfolio_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        Portfolio(["A", "B"], [0.5, 0.4])
