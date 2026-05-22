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
    
    # 5. Benchmarks
    from src.benchmarks import run_60_40_benchmark, run_equal_weight_benchmark
    bench_6040 = run_60_40_benchmark(returns)
    bench_ew = run_equal_weight_benchmark(returns)
    
    # 6. Metrics & Comparison
    hmm_metrics = calculate_metrics(results['ret'])
    bench_6040_metrics = calculate_metrics(bench_6040)
    bench_ew_metrics = calculate_metrics(bench_ew)
    
    summary = pd.DataFrame({
        'HMM Strategy': hmm_metrics,
        '60/40 Benchmark': bench_6040_metrics,
        'Equal Weight': bench_ew_metrics
    }).T
    
    print("\n--- Performance Comparison ---")
    print(summary)
    
    # 7. Visualization (Placeholder - needs real data to show well)
    # In a real run, we'd save these plots
    # plot_equity_curves({'HMM': (1+results['ret']).cumprod(), '60/40': (1+bench_6040).cumprod()})
    
    print("\n--- Project Completed Successfully ---")

if __name__ == "__main__":
    main()
