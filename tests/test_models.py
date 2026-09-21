import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

import pandas as pd

from alpha.equal import EqualAlpha
from alpha.historical_mean import HistoricalMeanAlpha
from alpha.momentum import MomentumAlpha
from risk.historical_covariance import HistoricalCovarianceRiskModel


def test_alpha_models_and_risk_model_share_asset_labels():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.02]})

    assert EqualAlpha().fit(returns).predict().index.tolist() == ["A", "B"]
    assert HistoricalMeanAlpha().fit(returns).predict().index.tolist() == ["A", "B"]
    assert MomentumAlpha(2).fit(returns).predict().index.tolist() == ["A", "B"]
    assert HistoricalCovarianceRiskModel().fit(returns).covariance().index.tolist() == ["A", "B"]
