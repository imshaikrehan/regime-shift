# HMM-Driven Dynamic Portfolio Allocation Engine

This project implements a regime-aware portfolio optimization system using Hidden Markov Models (HMM) to detect market states and CVXPY for dynamic capital allocation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  yfinance (prices) → FRED API (macro) → VIX (cboe/yfinance)    │
│  ↓                                                              │
│  Raw OHLCV + Macro Indicators → Feature Engineering             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE STORE                                  │
│  Log returns | Volatility (21d rolling σ) | VIX levels          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HMM REGIME CLASSIFIER                          │
│  Gaussian HMM (N=3) | Bull, Bear, Crisis states                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CVXPY OPTIMIZER                                │
│  Regime-conditional Objective | Max Sharpe vs Min Var          │
└─────────────────────────────────────────────────────────────────┘
```

## Current Progress
- [x] Phase 0: Environment & Repository Setup
- [x] Phase 1: Data Ingestion & Feature Engineering
- [x] Phase 2: HMM Regime Classifier
- [x] Phase 3: Regime-Conditional Portfolio Optimizer
- [x] Phase 4: Walk-Forward Backtesting Harness
- [x] Phase 5: Performance Analytics
- [x] Phase 6: Visualization Implementation
- [x] Phase 7: Main Execution Script

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the main script: `python src/main.py`
