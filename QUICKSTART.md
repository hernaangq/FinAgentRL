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

## Training a Real RL Agent

### Option 1: Stable-Baselines3 (Recommended)
```bash
pip install stable-baselines3[extra]
```

```python
from stable_baselines3 import PPO
from mag7_env import Mag7TradingEnv

env = Mag7TradingEnv(initial_cash=10000, lookback_period="1y")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("mag7_trader")
```

### Option 2: Custom Algorithm
```python
from mag7_env import Mag7TradingEnv
import numpy as np

env = Mag7TradingEnv()
obs, info = env.reset()

for episode in range(100):
    done = False
    while not done:
        # Your algorithm here
        action = your_policy(obs)  
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```

## Performance Benchmarks

From the test run:
- **Random Agent**: ~15% average return (lucky!)
- **Buy-and-Hold**: ~4% return (1-year period)
- **Your RL Agent**: Should beat buy-and-hold! 🎯

## Customization Options

### Change Training Period
```python
env = Mag7TradingEnv(
    lookback_period="2y"  # or "6mo", "3mo", etc.
)
```

### Change Initial Capital
```python
env = Mag7TradingEnv(
    initial_cash=100000  # Start with $100k
)
```

### Enable Visual Output
```python
env = Mag7TradingEnv(
    render_mode="human"  # Print detailed info each step
)
```

## Next Steps for Your Project

1. **Data Enhancement**
   - Add technical indicators (RSI, MACD, Moving Averages)
   - Include volume data
   - Add market sentiment features

2. **Environment Improvements**
   - Transaction costs
   - Portfolio constraints (max % per stock)
   - Short selling
   - Position sizing (buy/sell multiple shares)

3. **Training Optimization**
   - Hyperparameter tuning
   - Multiple episodes with different time periods
   - Train/validation/test split

4. **Evaluation Metrics**
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Risk-adjusted returns

## File Structure

```
RA-FinRL/
├── README.md                    # Project overview
├── QUICKSTART.md               # This file
├── requirements.txt            # Dependencies
├── mag7_env.py                 # Main environment
├── train_mag7.py              # Training script
├── test_mag7.py               # Test suite
├── data_utils.py              # Data download utilities
├── sb3_integration.py         # SB3 example
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

### Data download fails
- Check internet connection
- Try a shorter period: `lookback_period="6mo"`
- Data is cached, so subsequent runs are faster

### Environment runs too slow
- Use shorter lookback period
- Reduce number of episodes
- Consider using vectorized environments

## Resources

- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **yfinance**: https://github.com/ranaroussi/yfinance
- **FinRL**: https://github.com/AI4Finance-Foundation/FinRL

## Tips for Better Results

1. **Start Simple**: Train on shorter periods first
2. **Monitor Learning**: Use tensorboard to track training
3. **Compare Baselines**: Always compare to buy-and-hold
4. **Risk Management**: Consider adding constraints
5. **Diversification**: The environment already encourages this!

Good luck training your RL agent! 🚀📈
