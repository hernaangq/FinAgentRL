# Risk-Adjusted FinRL

Hernan Garcia Quijano
Alberto Mateo Muñoz

## Overview

A reinforcement learning environment for multi-stock trading using real market data from the Magnificent 7 (Mag7) tech stocks. The environment uses Gymnasium and integrates with Yahoo Finance for historical price data.

## Features

- **Multi-Stock Trading**: Trade 7 stocks simultaneously (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
- **Real Market Data**: Uses yfinance to fetch actual historical price data
- **Realistic Actions**: Buy, hold, or sell any stock at each time step
- **Portfolio Optimization**: Agent learns to maximize total portfolio value
- **Flexible Lookback**: Configurable training period (default: 1 year)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies include:
- `gymnasium` - RL environment framework
- `yfinance` - Yahoo Finance data downloader
- `numpy` - Numerical computing
- `pandas` - Data manipulation

## Files

### Core Environment
- `mag7_env.py` - Main multi-stock trading environment with yfinance integration
- `custom_env.py` - Simple single-stock environment (legacy)

### Training & Testing
- `train_mag7.py` - Training script with random and buy-hold strategies
- `test_mag7.py` - Comprehensive environment validation tests
- `train.py` - Simple training for legacy environment
- `test_env.py` - Tests for legacy environment

### Utilities
- `data_utils.py` - Data download and analysis utilities
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Test the Mag7 Environment

```bash
python test_mag7.py
```

This will:
- Download 1 year of historical data for Mag7 stocks
- Run comprehensive tests on the environment
- Validate action/observation spaces
- Test buy/sell/hold functionality

### 2. Run Training with Baseline Strategies

```bash
python train_mag7.py
```

This demonstrates:
- Random agent baseline
- Buy-and-hold strategy for comparison
- Portfolio performance metrics

### 3. Explore the Data

```bash
python data_utils.py
```

This will:
- Download and save Mag7 data to CSV
- Calculate returns and statistics
- Display stock information

## Environment Details

### Mag7TradingEnv

**Stocks**: Apple, Microsoft, Google, Amazon, Nvidia, Meta, Tesla

**Action Space**: MultiDiscrete([3, 3, 3, 3, 3, 3, 3])
- For each stock: 0=Sell, 1=Hold, 2=Buy
- Agent decides action for all 7 stocks simultaneously

**Observation Space**: Box(22)
- Cash available (1)
- Holdings for each stock (7)
- Current prices for each stock (7)
- Price returns for each stock (7)

**Reward**: Change in total portfolio value
- Portfolio value = cash + sum(holdings * prices)
- Agent objective: maximize final portfolio value

**Episode**: One complete pass through the historical data (~252 trading days for 1 year)

## Next Steps

To train a real RL agent, you can integrate algorithms like:

1. **DQN** (Deep Q-Network) - For discrete action spaces
2. **PPO** (Proximal Policy Optimization) - Popular choice for continuous control
3. **A2C/A3C** (Advantage Actor-Critic) - Good for financial applications
4. **SAC** (Soft Actor-Critic) - For exploration in trading

Example using Stable-Baselines3:
```python
from stable_baselines3 import PPO
from mag7_env import Mag7TradingEnv

env = Mag7TradingEnv(initial_cash=10000, lookback_period="1y")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Configuration Options

### Environment Parameters

```python
env = Mag7TradingEnv(
    initial_cash=10000.0,      # Starting cash
    lookback_period="1y",       # "1y", "2y", "6mo", etc.
    render_mode="human"         # "human" or None
)
```

## Performance Baseline

The random agent typically achieves ~0% return (expected).
The buy-and-hold strategy provides a realistic benchmark based on Mag7 actual performance.

Your RL agent should aim to beat buy-and-hold! 🚀

