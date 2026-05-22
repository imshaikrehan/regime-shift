import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_regimes(prices, regimes, title="Market Regimes"):
    """
    Plot price with regime background. 
    Matplotlib is great, but its defaults are... well, they need work.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prices, color='black', alpha=0.7, label='Price')
    
    # Colors for regimes
    colors = {'Bull': 'green', 'Bear': 'yellow', 'Crisis': 'red'}
    
    # Shade background
    for i in range(len(regimes)-1):
        ax.axvspan(regimes.index[i], regimes.index[i+1], 
                   color=colors.get(regimes.iloc[i], 'grey'), alpha=0.2)
        
    ax.set_title(title)
    ax.legend()
    return fig

def plot_equity_curves(curves, title="Equity Curves"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, curve in curves.items():
        ax.plot(curve, label=name)
        
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    return fig

if __name__ == "__main__":
    print("Visualizer ready.")
