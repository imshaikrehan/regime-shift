import pandas as pd
import numpy as np

def calculate_features(prices_df):
    """
    Calculate features for HMM:
    - 21-day rolling returns
    - 21-day rolling volatility
    - SPY-TLT correlation
    """
    returns = prices_df.pct_change()
    
    # 21-day smoothed returns
    smooth_ret = returns.rolling(21).mean()
    
    # 21-day annualized volatility
    vol = returns.rolling(21).std() * np.sqrt(252)
    
    # SPY-TLT rolling correlation (if both exist)
    if 'SPY' in returns.columns and 'TLT' in returns.columns:
        spy_tlt_corr = returns['SPY'].rolling(63).corr(returns['TLT'])
    else:
        spy_tlt_corr = pd.Series(np.nan, index=returns.index)
        
    features = pd.concat([smooth_ret, vol], axis=1)
    # Give unique names to columns
    features.columns = [f"{c}_ret" for c in smooth_ret.columns] + [f"{c}_vol" for c in vol.columns]
    features['spy_tlt_corr'] = spy_tlt_corr
    
    return features.dropna()

if __name__ == "__main__":
    # Test with dummy data or load from raw if it exists
    try:
        df = pd.read_csv('data/raw/prices.csv', index_col=0, parse_dates=True)
        feat = calculate_features(df)
        print(feat.head())
        feat.to_csv('data/processed/features.csv')
        print("Features saved to data/processed/features.csv")
    except FileNotFoundError:
        print("No raw data found. Run data_loader.py first.")
