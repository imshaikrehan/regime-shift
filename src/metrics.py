import numpy as np
import pandas as pd

def calculate_metrics(returns_series, rf_annual=0.02):
    """
    Calculate performance metrics. 
    Seeing a negative Sharpe ratio is my worst nightmare.
    """
    rf_daily = rf_annual / 252
    excess_returns = returns_series - rf_daily
    
    # Annualized Return
    ann_ret = (1 + returns_series.mean())**252 - 1
    
    # Sharpe Ratio
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    # Sortino Ratio
    downside_std = returns_series[returns_series < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf_annual) / downside_std
    
    # Max Drawdown
    cum_ret = (1 + returns_series).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar Ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        'Annualized Return': ann_ret,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar
    }

if __name__ == "__main__":
    # Test with dummy data
    rets = pd.Series(np.random.randn(1000) * 0.01)
    print(calculate_metrics(rets))
