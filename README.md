# Risk-Adjusted FinRL

Hernan Garcia Quijano
Alberto Mateo Muñoz

## Overview

A minimal reinforcement learning environment for financial trading using Gymnasium.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Files

- `custom_env.py` - Custom Gymnasium environment for financial trading
- `train.py` - Simple training script with random agent
- `test_env.py` - Environment testing and validation script
- `requirements.txt` - Python dependencies

## Quick Start

1. Test the environment:
```bash
python test_env.py
```

2. Run training with random agent:
```bash
python train.py
```

## Environment Details

**Action Space**: Discrete(3)
- 0: Hold
- 1: Buy
- 2: Sell

**Observation Space**: Box(3)
- Cash available
- Number of stocks owned
- Current stock price

**Reward**: Change in portfolio value

