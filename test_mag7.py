"""
Test script for the Mag7 multi-stock trading environment.
"""
from mag7_env import Mag7TradingEnv
import numpy as np


def test_mag7_environment():
    """Test the Mag7 trading environment."""
    
    print("Testing Mag7 Multi-Stock Trading Environment...")
    print("="*80)
    
    # Create environment
    print("\n1. Creating environment...")
    env = Mag7TradingEnv(
        initial_cash=10000.0,
        lookback_period="1y",
        render_mode=None
    )
    print(f"   ✓ Environment created")
    print(f"   - Stocks: {', '.join(env.MAG7_TICKERS)}")
    print(f"   - Trading days: {env.max_steps}")
    
    # Test reset
    print("\n2. Testing reset()...")
    observation, info = env.reset(seed=42)
    print(f"   ✓ Reset successful")
    print(f"   - Observation shape: {observation.shape}")
    print(f"   - Expected shape: ({env.observation_space.shape[0]},)")
    print(f"   - Initial portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"   - Initial cash: ${info['cash']:,.2f}")
    assert observation.shape == env.observation_space.shape, "Observation shape mismatch"
    assert info['cash'] == 10000.0, "Initial cash mismatch"
    
    # Test action space
    print("\n3. Testing action space...")
    print(f"   - Action space: {env.action_space}")
    for _ in range(5):
        action = env.action_space.sample()
        assert len(action) == env.n_stocks, "Action length mismatch"
        assert all(a in [0, 1, 2] for a in action), "Invalid action values"
    print(f"   ✓ Action space valid (7 stocks, 3 actions each)")
    
    # Test single step - buy action
    print("\n4. Testing step() with BUY action...")
    action = np.array([2, 1, 1, 1, 1, 1, 1])  # Buy AAPL, hold others
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Step executed")
    print(f"   - Action: Buy AAPL, hold others")
    print(f"   - New cash: ${info['cash']:,.2f}")
    print(f"   - AAPL holdings: {info['holdings']['AAPL']:.0f} shares")
    print(f"   - Portfolio value: ${info['portfolio_value']:,.2f}")
    assert info['holdings']['AAPL'] > 0, "AAPL purchase failed"
    
    # Test sell action
    print("\n5. Testing SELL action...")
    action = np.array([0, 1, 1, 1, 1, 1, 1])  # Sell AAPL, hold others
    prev_cash = info['cash']
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Sell executed")
    print(f"   - Previous cash: ${prev_cash:,.2f}")
    print(f"   - New cash: ${info['cash']:,.2f}")
    print(f"   - AAPL holdings: {info['holdings']['AAPL']:.0f} shares")
    assert info['cash'] > prev_cash, "Sell didn't increase cash"
    
    # Test full episode
    print("\n6. Testing full episode...")
    observation, info = env.reset()
    initial_value = info['portfolio_value']
    total_reward = 0
    steps = 0
    
    while steps < 50:  # Run for 50 steps
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    final_value = info['portfolio_value']
    print(f"   ✓ Episode completed")
    print(f"   - Steps taken: {steps}")
    print(f"   - Initial value: ${initial_value:,.2f}")
    print(f"   - Final value: ${final_value:,.2f}")
    print(f"   - Return: ${final_value - initial_value:,.2f}")
    
    # Test observation space bounds
    print("\n7. Testing observation space...")
    assert env.observation_space.contains(observation), "Observation out of bounds"
    print(f"   ✓ Observation space valid")
    
    # Test data integrity
    print("\n8. Testing price data...")
    print(f"   - Price data shape: {env.price_data.shape}")
    print(f"   - Columns: {list(env.price_data.columns)}")
    assert len(env.price_data.columns) == env.n_stocks, "Price data columns mismatch"
    assert len(env.price_data) > 200, "Insufficient price data"
    print(f"   ✓ Price data valid")
    
    # Test portfolio value calculation
    print("\n9. Testing portfolio value calculation...")
    observation, info = env.reset()
    # Buy one share of each stock
    for i in range(env.n_stocks):
        action = np.array([2] * env.n_stocks)  # Buy all
        observation, reward, terminated, truncated, info = env.step(action)
        if info['cash'] < 0:
            break
    
    calculated_value = info['cash']
    for ticker in env.MAG7_TICKERS:
        calculated_value += info['holdings'][ticker] * info['prices'][ticker]
    
    assert abs(calculated_value - info['portfolio_value']) < 0.01, "Portfolio calculation error"
    print(f"   ✓ Portfolio value calculation correct")
    print(f"   - Portfolio value: ${info['portfolio_value']:,.2f}")
    
    env.close()
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    print("\nEnvironment is ready for RL training!")
    print("- Observation space: Cash + Holdings + Prices + Returns")
    print("- Action space: Buy/Hold/Sell for each of 7 stocks")
    print("- Reward: Change in portfolio value")
    print("- Objective: Maximize portfolio value over time")


if __name__ == "__main__":
    test_mag7_environment()
