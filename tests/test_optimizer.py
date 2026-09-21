import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

import numpy as np
import pandas as pd

from optimization.optimizer import OptimizationObjective, optimize_weights


def test_maximum_return_allocates_to_highest_mean_return_asset():
    returns = pd.DataFrame(
        {
            "lower_return": [0.01, 0.02, 0.01, 0.02],
            "higher_return": [0.03, 0.04, 0.03, 0.04],
        }
    )

    weights = optimize_weights(returns, OptimizationObjective.MAXIMUM_EXPECTED_RETURN)

    assert np.isclose(weights.sum(), 1.0)
    assert weights[1] > 0.999
