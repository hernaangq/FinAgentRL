"""
Example of how to integrate the Mag7 environment with Stable-Baselines3 (SB3)
for training real RL agents.

To use this, first install stable-baselines3:
    pip install stable-baselines3[extra]
"""

# Uncomment the following to actually train an agent:
"""
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from mag7_env import Mag7TradingEnv
import numpy as np


def train_ppo_agent():
    '''Train a PPO agent on the Mag7 environment.'''
    
    # Create environment
    env = Mag7TradingEnv(initial_cash=10000, lookback_period="1y")
    
    # Wrap in DummyVecEnv for SB3
    env = DummyVecEnv([lambda: env])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Train the agent
    print("Training PPO agent...")
    model.learn(total_timesteps=100000)
    
    # Save the model
    model.save("ppo_mag7_trader")
    
    return model


def train_a2c_agent():
    '''Train an A2C agent on the Mag7 environment.'''
    
    env = Mag7TradingEnv(initial_cash=10000, lookback_period="1y")
    env = DummyVecEnv([lambda: env])
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Training A2C agent...")
    model.learn(total_timesteps=100000)
    model.save("a2c_mag7_trader")
    
    return model


def evaluate_trained_model(model_path, num_episodes=10):
    '''Evaluate a trained model.'''
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create environment
    env = Mag7TradingEnv(initial_cash=10000, lookback_period="1y")
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Run a few episodes manually to see details
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Initial Portfolio: ${info['portfolio_value']:,.2f}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        print(f"Final Portfolio: ${info['portfolio_value']:,.2f}")
        print(f"Return: ${episode_reward:,.2f} ({(episode_reward/10000)*100:+.2f}%)")


if __name__ == "__main__":
    # Train PPO agent
    print("="*80)
    print("Training PPO Agent on Mag7 Environment")
    print("="*80)
    model = train_ppo_agent()
    
    # Evaluate
    print("\n" + "="*80)
    print("Evaluating Trained Agent")
    print("="*80)
    evaluate_trained_model("ppo_mag7_trader")
"""

# For now, just print instructions
print("="*80)
print("Integration with Stable-Baselines3")
print("="*80)
print("\nTo train a real RL agent, follow these steps:\n")
print("1. Install Stable-Baselines3:")
print("   pip install stable-baselines3[extra]\n")
print("2. Uncomment the code in this file (sb3_integration.py)\n")
print("3. Run the training:")
print("   python sb3_integration.py\n")
print("Popular algorithms for trading:")
print("  - PPO (Proximal Policy Optimization) - Good all-rounder")
print("  - A2C (Advantage Actor-Critic) - Fast training")
print("  - DQN (Deep Q-Network) - For discrete actions")
print("  - SAC (Soft Actor-Critic) - For exploration\n")
print("="*80)
