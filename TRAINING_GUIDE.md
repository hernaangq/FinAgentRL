# Professional RL Training Pipeline for Mag7 Trading

This guide shows you how to use the professional training and evaluation pipeline.

## Complete Training Workflow

### Step 1: Train a Model

Train a PPO agent for 100,000 timesteps:

```bash
python ppo_train_mag7.py 100000
```

This will create a directory structure like:
```
Traceability/
└── PPO_model_Mag7_2025-11-09-15-30-45_100000/
    ├── BestModel/
    │   ├── best_model.zip
    │   └── best_model_unzipped/
    ├── Network/
    │   ├── PPO_model_Mag7.zip
    │   └── PPO_model_Mag7_unzipped/
    ├── Logs/
    │   └── evaluations.npz
    ├── Tensorboard/
    │   └── PPO_0/
    ├── Test/
    │   └── PPO_model_Mag7_*.monitor.csv
    └── Info/
        ├── training_summary.txt
        └── mag7_env.py
```

### Step 2: Monitor Training (Optional)

While training is running or after completion:

```bash
tensorboard --logdir=./Traceability/
```

Then open http://localhost:6006 in your browser.

### Step 3: Evaluate the Best Model

After training completes, evaluate the best model:

```bash
python evaluate_mag7.py PPO ./Traceability/PPO_model_Mag7_2025-11-09-15-30-45_100000/BestModel/best_model.zip 20 --verbose
```

This will show:
- Win rate (% of profitable episodes)
- Average return and standard deviation
- Sharpe-like ratio (risk-adjusted performance)
- Best/worst returns
- Detailed episode breakdowns

### Step 4: Save Evaluation Results

To save evaluation results to JSON:

```bash
python evaluate_mag7.py PPO ./Traceability/PPO_model_Mag7_2025-11-09-15-30-45_100000/BestModel/best_model.zip 20 --save-results evaluation_results.json --verbose
```

## Advanced Usage

### Custom Training Parameters

Edit `ppo_train_mag7.py` to modify:

```python
# Network architecture
pi = [256, 256, 256]  # Actor network layers
vf = [256, 256, 256]  # Critic network layers

# Hyperparameters
learning_rate = 0.0003
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01

# Environment
initial_cash = 10000
lookback_period = "1y"  # or "2y", "6mo", etc.
```

### Resume from Pretrained Model

In `ppo_train_mag7.py`, set:

```python
use_pretrained_model = True
pretrained_model = r"./Traceability/PPO_model_Mag7_.../Network/PPO_model_Mag7.zip"
```

Then run training as normal.

### Evaluate with Different Settings

```bash
# Evaluate with different initial cash
python evaluate_mag7.py PPO model.zip 10 --initial-cash 50000

# Evaluate with different time period
python evaluate_mag7.py PPO model.zip 10 --lookback 2y

# Deterministic evaluation (no exploration)
python evaluate_mag7.py PPO model.zip 10 --deterministic

# Render the environment during evaluation
python evaluate_mag7.py PPO model.zip 3 --render --verbose
```

## Training Different Algorithms

### Train with SAC (for comparison)

You can create a similar script for SAC:

```python
# sac_train_mag7.py (modify ppo_train_mag7.py)
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    monitor_env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.001,
    batch_size=256,
    ent_coef="auto",
    verbose=1,
    tensorboard_log=f"./Traceability/{file_name}/Tensorboard"
)
```

### Train with A2C

```python
# a2c_train_mag7.py
from stable_baselines3 import A2C

model = A2C(
    "MlpPolicy",
    monitor_env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0007,
    n_steps=5,
    gamma=0.99,
    verbose=1,
    tensorboard_log=f"./Traceability/{file_name}/Tensorboard"
)
```

## Understanding the Results

### Training Summary (`Info/training_summary.txt`)

Contains complete information about:
- Training date and duration
- Environment parameters
- All hyperparameters used
- Network architecture
- File locations

### TensorBoard Metrics

Key metrics to monitor:
- `rollout/ep_rew_mean`: Average episode reward (portfolio change)
- `rollout/ep_len_mean`: Average episode length
- `train/learning_rate`: Current learning rate
- `train/policy_loss`: Policy network loss
- `train/value_loss`: Value network loss
- `train/entropy_loss`: Exploration metric

### Evaluation Metrics

The evaluation script provides:

1. **Reward Statistics**: Absolute dollar changes
2. **Return Statistics**: Percentage returns
3. **Sharpe-like Ratio**: Risk-adjusted performance
4. **Win Rate**: Percentage of profitable episodes
5. **Win/Loss Ratio**: Average profit vs average loss
6. **Episode Lengths**: Trading days per episode

### What's a Good Result?

For the Mag7 environment:
- **Win Rate > 60%**: Good
- **Average Return > 5%**: Beating typical market returns
- **Sharpe Ratio > 1.0**: Good risk-adjusted returns
- **Win/Loss Ratio > 1.5**: Profitable strategy

Compare against:
- Random agent: ~50% win rate, ~0% average return
- Buy-and-hold: Depends on market period

## Tips for Better Performance

1. **Train Longer**: Try 500,000 or 1,000,000 timesteps
2. **Tune Hyperparameters**: Adjust learning rate, batch size, etc.
3. **Network Size**: Try larger networks (512 or 1024 units)
4. **Add Features**: Modify environment to include technical indicators
5. **Multiple Seeds**: Train with different random seeds and average results
6. **Ensemble**: Combine multiple trained models

## Troubleshooting

### Training is slow
- Reduce `n_steps` or `batch_size`
- Use shorter `lookback_period`
- Train on GPU if available

### Agent not learning
- Increase training timesteps
- Adjust learning rate (try 0.0001 or 0.001)
- Check TensorBoard for policy/value loss trends
- Ensure environment rewards are properly scaled

### Out of memory
- Reduce network size (fewer layers or units)
- Reduce `n_steps` or `batch_size`
- Use shorter lookback period

## Example Complete Session

```bash
# 1. Test environment
python test_mag7.py

# 2. Train for 500k timesteps
python ppo_train_mag7.py 500000

# 3. Monitor training (in another terminal)
tensorboard --logdir=./Traceability/

# 4. After training, evaluate
python evaluate_mag7.py PPO ./Traceability/PPO_model_Mag7_*/BestModel/best_model.zip 50 --verbose --save-results results.json

# 5. Compare with baseline
python train_mag7.py
```

## Next Steps

- Implement position sizing (buy/sell multiple shares)
- Add transaction costs
- Include technical indicators in observations
- Test on different time periods
- Deploy best model for paper trading
- Create ensemble of multiple models
