import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mag7_env import Mag7TradingEnv
import numpy as np
import time
from datetime import datetime
import zipfile
import shutil
import os
import sys

# TUNEABLE PARAMETERS
# -------------------------------------------------------

# Total training timesteps
# Get total_timesteps from command line
if len(sys.argv) > 1:
    if sys.argv[1] in ('--help', '-h'):
        print("""
Usage: python ppo_train_mag7.py <total_timesteps>
              
This script trains a Proximal Policy Optimization (PPO) reinforcement learning agent 
on the Mag7 multi-stock trading environment using Stable Baselines3. It supports 
training from scratch or resuming from a pretrained model, logs results to TensorBoard, 
saves the best model, and tracks training progress.

Arguments:
  <total_timesteps>   Number of training timesteps (integer, required)

Options:
  -h, --help          Show this help message

Example:
  python ppo_train_mag7.py 100000
""")
        sys.exit(0)
    try:
        total_timesteps = int(sys.argv[1])
    except ValueError:
        print("Invalid total_timesteps value. Must be an integer.")
        sys.exit(1)
else:
    print("No total_timesteps provided. Use --help for usage.")
    sys.exit(1)

# Policy (pi) & Value Function (vf) hidden layers
pi = [256, 256, 256]
vf = [256, 256, 256]

# Hyperparameters for PPO
# These parameters can be tuned to improve the performance of the PPO agent
# For more information on these parameters, refer to the Stable Baselines3 documentation
# (see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters)
learning_rate = 0.0003    # learning_rate: The learning rate for the optimizer
n_steps = 2048            # n_steps: Number of steps to run for each environment per update
batch_size = 64           # batch_size: Minibatch size
n_epochs = 10             # n_epochs: Number of epoch when optimizing the surrogate loss
gamma = 0.99              # gamma: Discount factor
gae_lambda = 0.95         # gae_lambda: Factor for trade-off of bias vs variance for GAE
clip_range = 0.2          # clip_range: Clipping parameter for PPO
ent_coef = 0.01           # ent_coef: Entropy coefficient for the loss calculation

# Environment parameters
initial_cash = 10000      # Starting cash for trading
lookback_period = "1y"    # Period of historical data to use

use_pretrained_model = False  # Set to True if you want to use a pretrained model
pretrained_model = r"Traceability/PPO_model_Mag7_2025-01-01-12-00-00_100000/Network/PPO_model_Mag7.zip"

# Define the file basename for saving the model
file_basename = "PPO_model_Mag7"
# -------------------------------------------------------

# Create timestamp for unique run identification
time_now = datetime.now()
formatted_time = time_now.strftime('%Y-%m-%d-%H-%M-%S')

# Define file name structure
file_name = f"{file_basename}_{formatted_time}_{total_timesteps}"
file_name_monitor = f'./Traceability/{file_name}/Test/{file_name}'

# Create environment
print("="*80)
print("Creating Mag7 Trading Environment...")
print("="*80)
env = Mag7TradingEnv(initial_cash=initial_cash, lookback_period=lookback_period)
monitor_env = Monitor(env, filename=file_name_monitor)

# Create model
if not use_pretrained_model:
    # Policy network architecture
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=pi, vf=vf)
    )
    
    print(f"\nCreating PPO model with architecture:")
    print(f"  Policy layers: {pi}")
    print(f"  Value layers: {vf}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    
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
        verbose=1,
        tensorboard_log=f"./Traceability/{file_name}/Tensorboard"
    )
else:
    print(f"\nLoading pretrained model from: {pretrained_model}")
    model = PPO.load(path=pretrained_model, env=monitor_env)


class CustomCallback(BaseCallback):
    """
    Custom callback for logging when a new best model is found.
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass


# Create evaluation callback
print("\nSetting up evaluation callback...")
eval_callback = EvalCallback(
    monitor_env,
    callback_on_new_best=CustomCallback(),
    n_eval_episodes=5,
    best_model_save_path=f"./Traceability/{file_name}/BestModel",
    log_path=f"./Traceability/{file_name}/Logs",
    eval_freq=10000,
    deterministic=True,
    render=False,
    verbose=1
)

callbacks = [eval_callback]
kwargs = {"callback": callbacks}

# Ensure directories exist
network_dir = f"./Traceability/{file_name}/Network"
os.makedirs(network_dir, exist_ok=True)

# Model Training
print("\n" + "="*80)
print(f"Starting PPO Training for {total_timesteps} timesteps...")
print("="*80)
start_time = time.time()

model.learn(total_timesteps=total_timesteps, progress_bar=True, **kwargs)

end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Save the final model
print(f"\nSaving model to {network_dir}/{file_basename}.zip")
model.save(f"{network_dir}/{file_basename}")

# Unzip the model
model_zip_path = f"{network_dir}/{file_basename}.zip"
unzip_dir = f"{network_dir}/{file_basename}_unzipped"

print(f"Unzipping model to {unzip_dir}")
with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# Unzip the best model if it exists
best_model_zip_path = f"./Traceability/{file_name}/BestModel/best_model.zip"
if os.path.exists(best_model_zip_path):
    best_model_unzip_dir = f"./Traceability/{file_name}/BestModel/best_model_unzipped"
    print(f"Unzipping best model to {best_model_unzip_dir}")
    with zipfile.ZipFile(best_model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(best_model_unzip_dir)

# Save training information
info_dir = f"./Traceability/{file_name}/Info"
os.makedirs(info_dir, exist_ok=True)

print(f"\nSaving training information to {info_dir}")
with open(f"{info_dir}/training_summary.txt", "w") as file:
    file.write("PPO TRAINING SUMMARY - MAG7 TRADING ENVIRONMENT\n")
    file.write("="*80 + "\n\n")
    file.write(f"Training Date: {formatted_time}\n")
    file.write(f"Total Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
    file.write("ENVIRONMENT PARAMETERS\n")
    file.write("-" * 80 + "\n")
    file.write(f"Initial Cash: ${initial_cash:,}\n")
    file.write(f"Lookback Period: {lookback_period}\n")
    file.write(f"Stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA\n")
    file.write(f"Action Space: MultiDiscrete([3, 3, 3, 3, 3, 3, 3])\n")
    file.write(f"Observation Space: Box(22,)\n\n")
    file.write("HYPERPARAMETERS\n")
    file.write("-" * 80 + "\n")
    file.write(f"Total Timesteps: {total_timesteps}\n")
    file.write(f"Learning Rate: {learning_rate}\n")
    file.write(f"N Steps: {n_steps}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"N Epochs: {n_epochs}\n")
    file.write(f"Gamma: {gamma}\n")
    file.write(f"GAE Lambda: {gae_lambda}\n")
    file.write(f"Clip Range: {clip_range}\n")
    file.write(f"Entropy Coefficient: {ent_coef}\n\n")
    file.write("NETWORK ARCHITECTURE\n")
    file.write("-" * 80 + "\n")
    file.write(f"Policy (Actor) Layers: {pi}\n")
    file.write(f"Value Function (Critic) Layers: {vf}\n")
    file.write(f"Activation Function: ReLU\n\n")
    file.write("FILES\n")
    file.write("-" * 80 + "\n")
    file.write(f"Model File: {network_dir}/{file_basename}.zip\n")
    file.write(f"Best Model: ./Traceability/{file_name}/BestModel/best_model.zip\n")
    file.write(f"TensorBoard Logs: ./Traceability/{file_name}/Tensorboard\n")
    file.write(f"Monitor Logs: {file_name_monitor}.monitor.csv\n")
    file.write("="*80 + "\n")

# Copy the environment file for traceability
source_file = "mag7_env.py"
destination_file = f"{info_dir}/mag7_env.py"
if os.path.exists(source_file):
    shutil.copyfile(source_file, destination_file)
    print(f"Environment file copied to {destination_file}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved to: ./Traceability/{file_name}/")
print(f"\nTo view TensorBoard logs, run:")
print(f"  tensorboard --logdir=./Traceability/{file_name}/Tensorboard")
print(f"\nTo load the best model:")
print(f"  model = PPO.load('./Traceability/{file_name}/BestModel/best_model.zip')")
print("="*80)

