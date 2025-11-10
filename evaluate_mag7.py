"""
Evaluation script for trained RL models (SAC or PPO) on Mag7 trading environment.

This script provides comprehensive evaluation metrics including episode rewards,
portfolio returns, win rates, and statistical analysis.
"""

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from mag7_env import Mag7TradingEnv
import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path
import json


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL models on Mag7 trading environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_mag7.py PPO model.zip 10
  python evaluate_mag7.py SAC model.zip 50 --verbose
  python evaluate_mag7.py PPO model.zip 20 --save-results results.json
  python evaluate_mag7.py PPO model.zip 10 --initial-cash 50000 --lookback 2y
        """
    )
    
    parser.add_argument(
        'policy', 
        choices=['SAC', 'PPO', 'A2C', 'sac', 'ppo', 'a2c', 'Sac', 'Ppo', 'A2c'],
        help='RL algorithm to use (SAC, PPO, or A2C)'
    )
    parser.add_argument(
        'model_path', 
        type=Path,
        help='Path to the model (.zip)'
    )
    parser.add_argument(
        'n_eval_episodes', 
        type=int,
        help='Number of episodes in evaluation (default: 10)',
        default=10
    )
    parser.add_argument(
        '--initial-cash',
        type=float,
        default=10000.0,
        help='Initial cash for trading (default: 10000)'
    )
    parser.add_argument(
        '--lookback',
        type=str,
        default='1y',
        help='Historical data period (e.g., "1y", "2y", "6mo") (default: 1y)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--save-results',
        type=Path,
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic policy during evaluation'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )
    
    return parser.parse_args()


def validate_model_path(model_path: Path) -> None:
    """Validate that the model file exists and is accessible."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.is_file():
        raise ValueError(f"Path is not a file: {model_path}")
    
    if model_path.suffix != '.zip':
        print(f"Warning: Model file doesn't have .zip extension: {model_path}")
    
    if not os.access(model_path, os.R_OK):
        raise PermissionError(f"Cannot read model file: {model_path}")


def load_model(policy_name: str, model_path: Path):
    """Load the trained model based on the policy type."""
    policy_name = policy_name.upper()
    
    model_classes = {
        'SAC': SAC,
        'PPO': PPO,
        'A2C': A2C
    }
    
    if policy_name not in model_classes:
        raise ValueError(f"Unsupported policy: {policy_name}. Supported: {list(model_classes.keys())}")
    
    try:
        model = model_classes[policy_name].load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load {policy_name} model from {model_path}: {e}")


def calculate_statistics(rewards: np.ndarray, lengths: np.ndarray, initial_cash: float) -> dict:
    """Calculate comprehensive statistics from evaluation results."""
    # Calculate returns as percentage
    returns_pct = (rewards / initial_cash) * 100
    
    stats = {
        'rewards': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards)),
            'q25': float(np.percentile(rewards, 25)),
            'q75': float(np.percentile(rewards, 75)),
        },
        'returns_pct': {
            'mean': float(np.mean(returns_pct)),
            'std': float(np.std(returns_pct)),
            'min': float(np.min(returns_pct)),
            'max': float(np.max(returns_pct)),
            'median': float(np.median(returns_pct)),
            'q25': float(np.percentile(returns_pct, 25)),
            'q75': float(np.percentile(returns_pct, 75)),
        },
        'episode_lengths': {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'median': float(np.median(lengths)),
        },
        'episodes_evaluated': len(rewards),
        'initial_cash': initial_cash
    }
    
    # Calculate Sharpe-like ratio (mean return / std return)
    if np.std(returns_pct) > 0:
        stats['sharpe_ratio'] = float(np.mean(returns_pct) / np.std(returns_pct))
    else:
        stats['sharpe_ratio'] = 0.0
    
    return stats


def calculate_success_rate(rewards: np.ndarray, initial_cash: float) -> dict:
    """Calculate success rate based on profitable trades."""
    # Success when final portfolio value > initial cash (positive return)
    profitable_episodes = np.sum(rewards > 0)
    success_rate = (profitable_episodes / len(rewards)) * 100
    
    # Calculate average profit and loss
    profits = rewards[rewards > 0]
    losses = rewards[rewards <= 0]
    
    avg_profit = float(np.mean(profits)) if len(profits) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    # Win/Loss ratio
    win_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
    
    return {
        'success_rate': float(success_rate),
        'profitable_episodes': int(profitable_episodes),
        'loss_episodes': int(len(rewards) - profitable_episodes),
        'total_episodes': len(rewards),
        'criterion': 'final_portfolio_value > initial_cash',
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'win_loss_ratio': float(win_loss_ratio),
        'best_return': float(np.max(rewards)),
        'worst_return': float(np.min(rewards))
    }


def print_evaluation_results(stats: dict, success_info: dict, verbose: bool = False):
    """Print formatted evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS - MAG7 TRADING ENVIRONMENT")
    print("="*80)
    
    # Episode statistics
    print(f"\nEpisodes evaluated: {stats['episodes_evaluated']}")
    print(f"Initial cash: ${stats['initial_cash']:,.2f}")
    
    # Reward statistics (absolute dollar amounts)
    print(f"\nReward Statistics (Portfolio Change in $):")
    print(f"  Mean:   ${stats['rewards']['mean']:>10,.2f} ± ${stats['rewards']['std']:>8,.2f}")
    print(f"  Median: ${stats['rewards']['median']:>10,.2f}")
    print(f"  Range:  [${stats['rewards']['min']:>10,.2f}, ${stats['rewards']['max']:>10,.2f}]")
    
    if verbose:
        print(f"  Q25:    ${stats['rewards']['q25']:>10,.2f}")
        print(f"  Q75:    ${stats['rewards']['q75']:>10,.2f}")
    
    # Return statistics (percentage)
    print(f"\nReturn Statistics (%):")
    print(f"  Mean:   {stats['returns_pct']['mean']:>8.2f}% ± {stats['returns_pct']['std']:>6.2f}%")
    print(f"  Median: {stats['returns_pct']['median']:>8.2f}%")
    print(f"  Range:  [{stats['returns_pct']['min']:>8.2f}%, {stats['returns_pct']['max']:>8.2f}%]")
    
    if verbose:
        print(f"  Q25:    {stats['returns_pct']['q25']:>8.2f}%")
        print(f"  Q75:    {stats['returns_pct']['q75']:>8.2f}%")
    
    # Sharpe ratio
    print(f"\nRisk-Adjusted Performance:")
    print(f"  Sharpe-like Ratio: {stats['sharpe_ratio']:>8.2f}")
    
    # Episode length statistics
    print(f"\nEpisode Length Statistics (Trading Days):")
    print(f"  Mean:   {stats['episode_lengths']['mean']:>8.1f} ± {stats['episode_lengths']['std']:>6.1f}")
    print(f"  Median: {stats['episode_lengths']['median']:>8.1f}")
    print(f"  Range:  [{stats['episode_lengths']['min']:>3d}, {stats['episode_lengths']['max']:>3d}]")
    
    # Success rate
    print(f"\nTrading Performance:")
    print(f"  Win Rate:           {success_info['success_rate']:>6.2f}%")
    print(f"  Profitable Episodes: {success_info['profitable_episodes']:>3d}/{success_info['total_episodes']:>3d}")
    print(f"  Loss Episodes:       {success_info['loss_episodes']:>3d}/{success_info['total_episodes']:>3d}")
    print(f"  Average Profit:      ${success_info['avg_profit']:>10,.2f}")
    print(f"  Average Loss:        ${success_info['avg_loss']:>10,.2f}")
    print(f"  Win/Loss Ratio:      {success_info['win_loss_ratio']:>6.2f}")
    print(f"  Best Return:         ${success_info['best_return']:>10,.2f}")
    print(f"  Worst Return:        ${success_info['worst_return']:>10,.2f}")
    
    print("="*80)


def save_results(results: dict, filepath: Path):
    """Save detailed results to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save results to {filepath}: {e}")


def run_detailed_episodes(model, env, n_episodes: int, deterministic: bool, verbose: bool):
    """Run episodes with detailed tracking for additional insights."""
    all_rewards = []
    all_lengths = []
    all_portfolio_values = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        initial_value = info['portfolio_value']
        
        if verbose and episode < 3:  # Show details for first 3 episodes
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1} Details:")
            print(f"Initial Portfolio Value: ${initial_value:,.2f}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        final_value = info['portfolio_value']
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_portfolio_values.append(final_value)
        
        if verbose and episode < 3:
            print(f"Final Portfolio Value: ${final_value:,.2f}")
            print(f"Return: ${episode_reward:,.2f} ({(episode_reward/initial_value)*100:+.2f}%)")
            print(f"Episode Length: {episode_length} days")
            print(f"Final Holdings:")
            for ticker, shares in info['holdings'].items():
                if shares > 0:
                    value = info['stock_values'][ticker]
                    print(f"  {ticker}: {shares:.0f} shares = ${value:,.2f}")
    
    return np.array(all_rewards), np.array(all_lengths), np.array(all_portfolio_values)


def main():
    """Main evaluation function."""
    try:
        args = parse_arguments()
        
        print("="*80)
        print("MAG7 TRADING MODEL EVALUATION")
        print("="*80)
        
        if args.verbose:
            print(f"\nConfiguration:")
            print(f"  Policy:        {args.policy.upper()}")
            print(f"  Model:         {args.model_path}")
            print(f"  Episodes:      {args.n_eval_episodes}")
            print(f"  Initial Cash:  ${args.initial_cash:,.2f}")
            print(f"  Lookback:      {args.lookback}")
            print(f"  Seed:          {args.seed}")
            print(f"  Deterministic: {args.deterministic}")
        
        # Validate inputs
        validate_model_path(args.model_path)
        
        if args.n_eval_episodes <= 0:
            raise ValueError("Number of evaluation episodes must be positive")
        
        # Load environment
        if args.verbose:
            print(f"\nCreating Mag7 trading environment...")
        
        render_mode = "human" if args.render else None
        env = Mag7TradingEnv(
            initial_cash=args.initial_cash,
            lookback_period=args.lookback,
            render_mode=render_mode
        )
        env.reset(seed=args.seed)
        
        # Load model
        if args.verbose:
            print(f"Loading {args.policy.upper()} model from: {args.model_path}")
        
        model = load_model(args.policy, args.model_path)
        
        # Evaluate model with detailed tracking
        print(f"\nEvaluating model for {args.n_eval_episodes} episodes...")
        start_time = time.time()
        
        episode_rewards, episode_lengths, portfolio_values = run_detailed_episodes(
            model, 
            env, 
            args.n_eval_episodes,
            args.deterministic,
            args.verbose
        )
        
        evaluation_time = time.time() - start_time
        
        # Calculate statistics
        stats = calculate_statistics(episode_rewards, episode_lengths, args.initial_cash)
        success_info = calculate_success_rate(episode_rewards, args.initial_cash)
        
        # Print results
        if args.verbose:
            print(f"\nRaw episode rewards: {episode_rewards.tolist()}")
            print(f"Raw episode lengths: {episode_lengths.tolist()}")
            print(f"Final portfolio values: {portfolio_values.tolist()}")
        
        print_evaluation_results(stats, success_info, args.verbose)
        
        print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
        print(f"Average time per episode: {evaluation_time/args.n_eval_episodes:.2f} seconds")
        
        # Save results if requested
        if args.save_results:
            results = {
                'evaluation_info': {
                    'policy': args.policy.upper(),
                    'model_path': str(args.model_path),
                    'environment': 'Mag7TradingEnv',
                    'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
                    'n_eval_episodes': args.n_eval_episodes,
                    'initial_cash': args.initial_cash,
                    'lookback_period': args.lookback,
                    'seed': args.seed,
                    'deterministic': args.deterministic,
                    'evaluation_time': evaluation_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'statistics': stats,
                'success_info': success_info,
                'raw_data': {
                    'episode_rewards': episode_rewards.tolist(),
                    'episode_lengths': episode_lengths.tolist(),
                    'final_portfolio_values': portfolio_values.tolist()
                }
            }
            save_results(results, args.save_results)
        
        env.close()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
