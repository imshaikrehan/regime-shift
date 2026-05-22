import pandas as pd
import numpy as np
from src.hmm_model import RegimeHMM
from src.optimizer import optimize_portfolio
from sklearn.preprocessing import StandardScaler

def run_backtest(feature_matrix, returns_df, config):
    """
    Walk-forward backtest. 
    It's always a pain to get the indexing right here... 
    hope this works!
    """
    results = []
    
    train_years = config['backtest']['train_years']
    train_size = train_years * 252
    
    n_assets = len(returns_df.columns)
    prev_weights = np.ones(n_assets) / n_assets
    
    # Simple loop for now, skipping the complex fold logic to get a MVP
    for t in range(train_size, len(feature_matrix), 21): # Rebalance monthly
        # Training window
        X_train = feature_matrix.iloc[t-train_size:t]
        r_train = returns_df.loc[X_train.index]
        
        # Scaling (Wait, I should do this in the HMM class or here?)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        hmm = RegimeHMM(n_states=3)
        hmm.fit(X_train_scaled)
        
        # Current state
        current_X = feature_matrix.iloc[t:t+1]
        current_X_scaled = scaler.transform(current_X)
        state_idx = hmm.predict(current_X_scaled)[0]
        
        labels = hmm.get_regime_labels(X_train, feature_matrix.columns)
        current_regime = labels.get(state_idx, 'Bear')
        
        # Moments (Simple empirical for now)
        mu = r_train.mean().values
        Sigma = r_train.cov().values
        
        weights = optimize_portfolio(mu, Sigma, current_regime, prev_weights, config['optimizer'])
        
        # Record results for the next month
        next_month_ret = returns_df.iloc[t:t+21]
        for date, row in next_month_ret.iterrows():
            port_ret = weights @ row.values
            results.append({'date': date, 'ret': port_ret, 'regime': current_regime})
            
        prev_weights = weights
        
    return pd.DataFrame(results).set_index('date')

if __name__ == "__main__":
    print("Backtester ready. (Need real data to test fully)")
