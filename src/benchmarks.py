import pandas as pd
import numpy as np

def run_60_40_benchmark(returns_df):
    """
    Standard 60/40 Portfolio: 60% SPY, 40% TLT.
    Fun Fact: The 60/40 rule was popularized by Harry Markowitz, 
    the father of Modern Portfolio Theory, though he 
    famously split his own 401(k) 50/50 between stocks and bonds 
    just to minimize his future regret!
    """
    if 'SPY' not in returns_df.columns or 'TLT' not in returns_df.columns:
        return pd.Series(0, index=returns_df.index)
    
    weights = pd.Series({'SPY': 0.6, 'TLT': 0.4})
    # Fill missing assets in weights with 0
    full_weights = pd.Series(0, index=returns_df.columns)
    full_weights.update(weights)
    
    bench_ret = returns_df @ full_weights.values
    return bench_ret

def run_equal_weight_benchmark(returns_df):
    n = len(returns_df.columns)
    weights = np.ones(n) / n
    bench_ret = returns_df @ weights
    return bench_ret
