import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mag7_env import Mag7TradingEnv
from datetime import datetime
import time
import zipfile
import shutil
import os
import sys

# TUNEABLE PARAMETERS - OPTIMIZED FOR FASTER LEARNING
# -------------------------------------------------------

# Total training timesteps
# Get total_timesteps from command line
if len(sys.argv) > 1:
    if sys.argv[1] in ('--help', '-h'):
        print("""
Usage: python ppo_train_mag7.py <total_timesteps>
              
Train a PPO agent on the Mag7 trading environment.

Arguments:
  <total_timesteps>   Number of training timesteps (integer, required)

Options:
  -h, --help          Show this help message

Examples:
  python ppo_train_mag7.py 100000   # Quick test
  python ppo_train_mag7.py 500000   # Full training
""")
        sys.exit(0)
    try:
        total_timesteps = int(sys.argv[1])
    except ValueError:
        print("Invalid total_timesteps value.")
        sys.exit(1)
else:
    print("No total_timesteps provided. Use --help for usage.")
    sys.exit(1)

# Network architecture - BIGGER NETWORK for complex trading
pi = [512, 512, 256]
vf = [512, 512, 256]

# PPO Hyperparameters - TUNED FOR FINANCIAL RL
learning_rate = 0.001          # Standard learning rate
n_steps = 2048                # Steps per update (about 8 trading days)
batch_size = 128              # Larger batch for stability
n_epochs = 10                 # Epochs per update
gamma = 0.99                  # Discount factor
gae_lambda = 0.95             # GAE lambda
clip_range = 0.2              # PPO clip range
ent_coef = 0.05               # HIGH entropy for exploration!
vf_coef = 0.5                 # Value function coefficient
max_grad_norm = 0.5           # Gradient clipping

# Environment parameters
initial_cash = 10000
lookback_period = "5y"
max_shares_per_trade = 10
transaction_cost_pct = 0.001  # 0.1% transaction cost

use_pretrained_model = False
pretrained_model = r"Traceability/PPO_model_Mag7_*/Network/PPO_model_Mag7.zip"

file_basename = "PPO_model_Mag7"
# -------------------------------------------------------

# Create timestamp
time_now = datetime.now()
formatted_time = time_now.strftime('%Y-%m-%d-%H-%M-%S')
file_name = f"{file_basename}_{formatted_time}_{total_timesteps}"
file_name_monitor = f'./Traceability/{file_name}/Test/{file_name}'

# Create environment
print("="*80)
print("Creating IMPROVED Mag7 Trading Environment...")
print("="*80)

env = Mag7TradingEnv(
    initial_cash=initial_cash,
    lookback_period=lookback_period,
    max_shares_per_trade=max_shares_per_trade,
    transaction_cost_pct=transaction_cost_pct
)
monitor_env = Monitor(env, filename=file_name_monitor)

# Create model
if not use_pretrained_model:
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=pi, vf=vf)
    )
    
    print(f"\nCreating PPO model:")
    print(f"  Policy: {pi}")
    print(f"  Value: {vf}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Entropy coef: {ent_coef} (HIGH for exploration)")
    
    model = PPO(
        "MlpPolicy",
        monitor_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=f"./Traceability/{file_name}/Tensorboard"
    )
else:
    print(f"\nLoading pretrained model from: {pretrained_model}")
    model = PPO.load(path=pretrained_model, env=monitor_env)


class CustomCallback(BaseCallback):
    """Custom callback for logging."""
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        return True


# Evaluation callback
print("\nSetting up evaluation...")
eval_callback = EvalCallback(
    monitor_env,
    callback_on_new_best=CustomCallback(),
    n_eval_episodes=5,
    best_model_save_path=f"./Traceability/{file_name}/BestModel",
    log_path=f"./Traceability/{file_name}/Logs",
    eval_freq=5000,  # Evaluate every 5000 steps
    deterministic=True,
    render=False,
    verbose=1
)

callbacks = [eval_callback]
kwargs = {"callback": callbacks}

# Create directories
network_dir = f"./Traceability/{file_name}/Network"
os.makedirs(network_dir, exist_ok=True)

# TRAINING
print("\n" + "="*80)
print(f"STARTING TRAINING - {total_timesteps:,} timesteps")
print("="*80)


start_time = time.time()
model.learn(total_timesteps=total_timesteps, progress_bar=True, **kwargs)
end_time = time.time()
training_time = end_time - start_time

print(f"\n✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

# Save model
print(f"\nSaving model to {network_dir}")
model.save(f"{network_dir}/{file_basename}")

# Unzip models
model_zip_path = f"{network_dir}/{file_basename}.zip"
if os.path.exists(model_zip_path):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"{network_dir}/{file_basename}_unzipped")

best_model_zip_path = f"./Traceability/{file_name}/BestModel/best_model.zip"
if os.path.exists(best_model_zip_path):
    with zipfile.ZipFile(best_model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"./Traceability/{file_name}/BestModel/best_model_unzipped")

# Save training summary
info_dir = f"./Traceability/{file_name}/Info"
os.makedirs(info_dir, exist_ok=True)

with open(f"{info_dir}/training_summary.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("IMPROVED PPO TRAINING - MAG7 TRADING\n")
    f.write("="*80 + "\n\n")
    f.write("ENVIRONMENT:\n")
    f.write(f"  Initial Cash: ${initial_cash:,}\n")
    f.write(f"  Data Period: {lookback_period}\n")
    f.write(f"  Trading Days: {len(env.price_data)}\n")
    f.write(f"  Action Space: Buy/Hold/Sell 10% per stock\n")
    f.write(f"  Transaction Cost: {transaction_cost_pct*100:.1f}%\n\n")
    f.write("NETWORK:\n")
    f.write(f"  Policy: {pi}\n")
    f.write(f"  Value: {vf}\n\n")
    f.write("PPO HYPERPARAMETERS:\n")
    f.write(f"  Learning rate: {learning_rate}\n")
    f.write(f"  Entropy coef: {ent_coef}\n")
    f.write(f"  Total timesteps: {total_timesteps:,}\n")
    f.write(f"  Training time: {training_time:.2f}s\n")
    f.write("="*80 + "\n")

# Copy env file
if os.path.exists("mag7_env.py"):
    shutil.copyfile("mag7_env.py", f"{info_dir}/mag7_env.py")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nResults: ./Traceability/{file_name}/")
print(f"\nEvaluate with:")
print(f"  python evaluate_mag7.py PPO ./Traceability/{file_name}/BestModel/best_model.zip 20 --verbose")
print("="*80)

