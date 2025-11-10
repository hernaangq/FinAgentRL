"""
Simple training script for the custom financial environment.
This uses a random agent as a baseline example.
"""
import gymnasium as gym
import numpy as np
from custom_env import SimpleFinancialEnv


def random_agent(env, num_episodes=10, render=False):
    """Run a random agent for demonstration."""
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            done = terminated or truncated
            
            if render:
                env.render()
        
        print(f"Episode {episode + 1} finished after {step} steps")
        print(f"Final Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"Total Return: ${episode_reward:.2f}")
        print(f"Return %: {(episode_reward / 10000) * 100:.2f}%")
    
    env.close()


def main():
    """Main function to run the training."""
    
    # Create environment
    print("Creating Financial Trading Environment...")
    env = SimpleFinancialEnv(render_mode=None)
    
    # Verify environment
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Run random agent
    print("\nRunning random agent...")
    random_agent(env, num_episodes=5, render=False)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == "__main__":
    main()
