"""
Training script for the Mag7 multi-stock trading environment.
"""
import numpy as np
from mag7_env import Mag7TradingEnv


def random_agent(env, num_episodes=5, render=False):
    """Run a random agent for demonstration."""
    
    all_returns = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        initial_value = info['portfolio_value']
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Initial Portfolio Value: ${initial_value:,.2f}")
        print(f"{'='*80}")
        
        while not done:
            # Random action for each stock
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            done = terminated or truncated
            
            if render and step % 20 == 0:  # Render every 20 steps
                env.render()
        
        final_value = info['portfolio_value']
        total_return = final_value - initial_value
        return_pct = (total_return / initial_value) * 100
        all_returns.append(return_pct)
        
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Duration: {step} days")
        print(f"  Initial Value: ${initial_value:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
        print(f"  Final Cash: ${info['cash']:,.2f}")
        print(f"\n  Final Holdings:")
        for ticker, shares in info['holdings'].items():
            if shares > 0:
                value = info['stock_values'][ticker]
                print(f"    {ticker}: {shares:.0f} shares (${value:,.2f})")
        print(f"{'='*80}")
    
    env.close()
    
    # Print overall statistics
    print(f"\n{'='*80}")
    print(f"Overall Statistics (Random Agent):")
    print(f"  Episodes: {num_episodes}")
    print(f"  Average Return: {np.mean(all_returns):+.2f}%")
    print(f"  Std Dev: {np.std(all_returns):.2f}%")
    print(f"  Best Return: {np.max(all_returns):+.2f}%")
    print(f"  Worst Return: {np.min(all_returns):+.2f}%")
    print(f"{'='*80}")


def buy_and_hold_agent(env, num_episodes=1):
    """
    Buy and hold strategy: Buy equal amounts of each stock at the start.
    """
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        initial_value = info['portfolio_value']
        
        print(f"\n{'='*80}")
        print(f"Buy and Hold Strategy")
        print(f"Initial Portfolio Value: ${initial_value:,.2f}")
        print(f"{'='*80}")
        
        # First action: Buy all stocks
        action = np.array([2] * env.n_stocks)  # 2 = buy for all stocks
        
        for step in range(env.max_steps):
            observation, reward, terminated, truncated, info = env.step(action)
            
            # After buying initially, just hold
            action = np.array([1] * env.n_stocks)  # 1 = hold
            
            if terminated or truncated:
                break
        
        final_value = info['portfolio_value']
        total_return = final_value - initial_value
        return_pct = (total_return / initial_value) * 100
        
        print(f"\n{'='*80}")
        print(f"Buy and Hold Results:")
        print(f"  Duration: {step + 1} days")
        print(f"  Initial Value: ${initial_value:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
        print(f"  Final Cash: ${info['cash']:,.2f}")
        print(f"\n  Final Holdings:")
        for ticker, shares in info['holdings'].items():
            if shares > 0:
                value = info['stock_values'][ticker]
                price = info['prices'][ticker]
                print(f"    {ticker}: {shares:.0f} shares @ ${price:.2f} = ${value:,.2f}")
        print(f"{'='*80}")
    
    env.close()


def main():
    """Main function to run the training."""
    
    # Create environment
    print("Creating Mag7 Multi-Stock Trading Environment...")
    print("This will download 1 year of historical data for the Magnificent 7 stocks...")
    
    env = Mag7TradingEnv(
        initial_cash=10000.0,
        lookback_period="1y",
        render_mode=None
    )
    
    # Verify environment
    print(f"\nEnvironment Info:")
    print(f"  Action Space: {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Number of Stocks: {env.n_stocks}")
    print(f"  Tickers: {', '.join(env.MAG7_TICKERS)}")
    print(f"  Trading Days: {env.max_steps}")
    
    # Run random agent
    print("\n" + "="*80)
    print("Running Random Agent...")
    print("="*80)
    random_agent(env, num_episodes=3, render=False)
    
    # Run buy and hold strategy for comparison
    print("\n" + "="*80)
    print("Running Buy and Hold Strategy for Comparison...")
    print("="*80)
    env = Mag7TradingEnv(initial_cash=10000.0, lookback_period="1y", render_mode=None)
    buy_and_hold_agent(env, num_episodes=1)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("Next steps: Implement a proper RL algorithm (DQN, PPO, A2C, etc.)")
    print("="*80)


if __name__ == "__main__":
    main()
