# Risk-Adjusted FinRL
Authors: 
- Hernan Garcia Quijano
- Alberto Mateo Muñoz

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
- `ppo_train_mag7.py` - **Professional PPO training script with full tracking**
- `evaluate_mag7.py` - **Comprehensive model evaluation with detailed metrics**
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

### 3. Train a PPO Agent (Professional Setup)

```bash
python ppo_train_mag7.py 100000
```

This will:
- Train a PPO agent for 100,000 timesteps
- Save models to `./Traceability/` with timestamps
- Log training to TensorBoard
- Evaluate periodically and save best model
- Generate comprehensive training summary

Arguments:
- `<total_timesteps>`: Number of training steps (required)
- Use `--help` for more information

### 4. Evaluate a Trained Model

```bash
python evaluate_mag7.py PPO ./Traceability/.../BestModel/best_model.zip 10
```

This will:
- Evaluate the model over 10 episodes
- Calculate win rate, returns, Sharpe ratio
- Show detailed performance metrics
- Optionally save results to JSON

Arguments:
- `<policy>`: PPO, SAC, or A2C
- `<model_path>`: Path to the .zip model file
- `<n_eval_episodes>`: Number of evaluation episodes
- `--verbose`: Show detailed output
- `--save-results <file.json>`: Save results
- `--deterministic`: Use deterministic policy

### 5. Monitor Training with TensorBoard

```bash
tensorboard --logdir=./Traceability/
```

### 6. Explore the Data

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



