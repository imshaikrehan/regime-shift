import unittest
import pandas as pd
import numpy as np
from src.metrics import calculate_metrics

class TestMetrics(unittest.TestCase):
    def test_sharpe_ratio(self):
        """
        I once spent 3 hours debugging a Sharpe ratio calculation 
        only to realize I forgot to annualize it. Never again.
        """
        # 1% daily return with 0 volatility should have a huge Sharpe
        rets = pd.Series([0.01] * 252)
        metrics = calculate_metrics(rets, rf_annual=0)
        self.assertTrue(metrics['Sharpe Ratio'] > 10)
        
    def test_max_drawdown(self):
        # 10% drop then back to 0
        rets = pd.Series([-0.1, 0.11111111])
        metrics = calculate_metrics(rets)
        self.assertAlmostEqual(metrics['Max Drawdown'], -0.1)

if __name__ == "__main__":
    unittest.main()
