"""
Test script to verify the environment is working correctly.
"""
import gymnasium as gym
from custom_env import SimpleFinancialEnv


def test_environment():
    """Test basic environment functionality."""
    
    print("Testing Custom Financial Environment...")
    print("="*50)
    
    # Create environment
    env = SimpleFinancialEnv(render_mode="human")
    
    # Test reset
    print("\n1. Testing reset()...")
    observation, info = env.reset(seed=42)
    print(f"   Initial observation: {observation}")
    print(f"   Initial info: {info}")
    assert observation.shape == (3,), "Observation shape mismatch"
    print("   ✓ Reset successful")
    
    # Test single step
    print("\n2. Testing step()...")
    action = 1  # Buy
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"   Action: Buy")
    print(f"   Observation: {observation}")
    print(f"   Reward: {reward:.2f}")
    print(f"   Terminated: {terminated}, Truncated: {truncated}")
    print("   ✓ Step successful")
    
    # Test full episode
    print("\n3. Testing full episode...")
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"   Episode completed in {steps} steps")
    print(f"   Total reward: {total_reward:.2f}")
    print("   ✓ Full episode successful")
    
    # Test action space
    print("\n4. Testing action space...")
    for _ in range(5):
        action = env.action_space.sample()
        assert action in [0, 1, 2], "Invalid action"
    print(f"   Action space: {env.action_space}")
    print("   ✓ Action space valid")
    
    # Test observation space
    print("\n5. Testing observation space...")
    print(f"   Observation space: {env.observation_space}")
    assert env.observation_space.contains(observation), "Observation out of bounds"
    print("   ✓ Observation space valid")
    
    env.close()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    test_environment()
