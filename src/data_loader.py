import yfinance as yf
import pandas as pd
import os
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_data(config):
    assets = config['data']['assets']
    tickers = assets['equities'] + assets['bonds'] + assets['safe_haven']
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    return data

if __name__ == "__main__":
    # Quick test
    cfg = load_config('config.yaml')
    df = download_data(cfg)
    print(df.head())
    # Save to raw
    df.to_csv('data/raw/prices.csv')
    print("Saved to data/raw/prices.csv")
