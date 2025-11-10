# Mag7 Trading Environment - Quick Reference

## What You Have Now

A fully functional multi-stock trading environment with:
- **Real market data** from Yahoo Finance for the Magnificent 7 stocks
- **Gymnasium-compatible** environment ready for any RL algorithm
- **Realistic trading**: Buy, hold, or sell each stock independently
- **Portfolio optimization**: Agent learns to maximize total value

## The Magnificent 7 Stocks

| Ticker | Company | Sector |
|--------|---------|--------|
| AAPL | Apple | Technology |
| MSFT | Microsoft | Technology |
| GOOGL | Alphabet (Google) | Technology |
| AMZN | Amazon | Consumer/Cloud |
| NVDA | NVIDIA | Technology/AI |
| META | Meta (Facebook) | Technology |
| TSLA | Tesla | Automotive/Energy |

## Environment Specs

### Observation Space (22 dimensions)
```
[cash,                              # 1 value
 aapl_shares, msft_shares, ...,     # 7 values
 aapl_price, msft_price, ...,       # 7 values  
 aapl_return, msft_return, ...]     # 7 values
```

### Action Space (7 discrete actions)
For each stock, agent chooses:
- **0** = Sell one share
- **1** = Hold (do nothing)
- **2** = Buy one share

### Reward Signal
- Reward = Change in portfolio value from previous step
- Portfolio Value = Cash + Σ(shares × prices)

## Quick Start Commands

### 1. Test Environment
```bash
python test_mag7.py
```

### 2. Run Baseline Strategies
```bash
python train_mag7.py
```

### 3. Analyze Data
```bash
python data_utils.py
```

### 4. Train a Professional PPO Agent
```bash
python ppo_train_mag7.py 100000
```

This will train for 100,000 timesteps and create a complete traceability folder with:
- TensorBoard logs
- Best model checkpoints
- Training summary
- Monitor CSV files

### 5. Visualize Training Results
```bash
python plot_results.py Traceability/PPO_model_Mag7_*/Test/*.monitor.csv
```

This will generate plots showing:
- Episode rewards over time
- Episode length over time
- Reward vs episode length scatter
- Smoothed reward trends

### 6. Evaluate Trained Model
```bash
python evaluate_mag7.py PPO Traceability/PPO_model_Mag7_*/BestModel/best_model.zip 20 --verbose
```

This provides comprehensive evaluation with:
- Win rate and profitability metrics
- Average returns and Sharpe ratio
- Detailed episode breakdowns
- Save results to JSON with `--save-results results.json`


## File Structure

```
RA-FinRL/
├── README.md                    # Project overview
├── QUICKSTART.md               # This file
├── TRAINING_GUIDE.md           # Professional training workflow guide
├── requirements.txt            # Dependencies
├── mag7_env.py                 # Main multi-stock environment
├── ppo_train_mag7.py          # Professional PPO training script
├── evaluate_mag7.py           # Comprehensive model evaluation
├── plot_results.py            # Training results visualization
├── train_mag7.py              # Baseline strategies demo
├── test_mag7.py               # Environment test suite
├── data_utils.py              # Data download utilities
├── sb3_integration.py         # SB3 integration example
├── custom_env.py              # Legacy single-stock env
├── train.py                   # Legacy training
└── test_env.py                # Legacy tests
```

## Common Issues & Solutions

### "Module not found: yfinance"
```bash
pip install yfinance
```

### "Module not found: gymnasium"
```bash
pip install gymnasium
```



## Resources

- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **yfinance**: https://github.com/ranaroussi/yfinance
- **FinRL**: https://github.com/AI4Finance-Foundation/FinRL

