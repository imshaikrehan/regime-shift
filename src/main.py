import os
import yaml
import pandas as pd
from src.data_loader import load_config, download_data
from src.features import calculate_features
from src.backtester import run_backtest
from src.metrics import calculate_metrics
from src.visualizer import plot_regimes, plot_equity_curves
import matplotlib.pyplot as plt

def main():
    print("--- HMM Portfolio Engine Starting ---")
    
    # 1. Load Config
    config = load_config('config.yaml')
    
    # 2. Data Ingestion
    if not os.path.exists('data/raw/prices.csv'):
        prices = download_data(config)
        prices.to_csv('data/raw/prices.csv')
    else:
        prices = pd.read_csv('data/raw/prices.csv', index_col=0, parse_dates=True)
    
    # 3. Feature Engineering
    features = calculate_features(prices)
    features.to_csv('data/processed/features.csv')
    
    # 4. Backtest
    returns = prices.pct_change().loc[features.index]
    results = run_backtest(features, returns, config)
    results.to_csv('outputs/results/backtest_results.csv')
    
    # 5. Metrics
    metrics = calculate_metrics(results['ret'])
    print("\n--- Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 6. Visualization (Placeholder - needs real data to show well)
    # ...
    
    print("\n--- Project Completed Successfully ---")

if __name__ == "__main__":
    main()
