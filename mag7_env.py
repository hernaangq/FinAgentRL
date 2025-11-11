import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class Mag7TradingEnv(gym.Env):
    """
    Multi-stock trading environment for Magnificent 7 stocks using real market data.
    
    IMPROVED VERSION with better reward signals and action space for faster learning.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Magnificent 7 stocks
    MAG7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    def __init__(self, initial_cash=10000.0, lookback_period="1y", render_mode=None, 
                 reward_type="portfolio_return", max_shares_per_trade=10,
                 transaction_cost_pct=0.001):
        """
        Initialize the Mag7 trading environment.
        
        Args:
            initial_cash: Starting cash amount
            lookback_period: Period to download data (e.g., "1y", "2y", "6mo")
            render_mode: Rendering mode ("human" or None)
            reward_type: Type of reward ("portfolio_return", "sortino", "calmar")
            max_shares_per_trade: Maximum shares to buy/sell per action
            transaction_cost_pct: Transaction cost as percentage (e.g., 0.001 = 0.1%)
        """
        super().__init__()
        
        self.initial_cash = initial_cash
        self.lookback_period = lookback_period
        self.render_mode = render_mode
        self.n_stocks = len(self.MAG7_TICKERS)
        self.max_shares_per_trade = max_shares_per_trade
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_type = reward_type
        
        # IMPROVED ACTION SPACE: 
        # For each stock: 0 = sell 10%, 1 = hold, 2 = buy with 10% of cash
        # This makes learning faster than buying 1 share at a time
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        
        # Observation space:
        # - Portfolio value / initial cash (1)
        # - Cash / initial cash (1) 
        # - For each stock:
        #   - Holdings as % of portfolio (7)
        #   - Price return over 5 days (7)
        #   - Price return over 20 days (7)
        # Total: 1 + 1 + 7 + 7 + 7 = 23 features
        obs_dim = 2 + (self.n_stocks * 3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Download historical data
        print(f"Downloading Mag7 historical data ({lookback_period})...")
        self.price_data = self._download_data()
        self.max_steps = len(self.price_data) - 1
        
        # Initialize state
        self.reset()
        
    def _download_data(self):
        """Download historical price data for Mag7 stocks."""
        data = {}
        
        for ticker in self.MAG7_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=self.lookback_period)
                data[ticker] = hist['Close'].values
                print(f"  {ticker}: {len(hist)} days")
            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")
                raise
        
        # Ensure all stocks have the same length
        min_len = min(len(v) for v in data.values())
        for ticker in self.MAG7_TICKERS:
            data[ticker] = data[ticker][:min_len]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Downloaded {len(df)} days of data for {self.n_stocks} stocks")
        
        return df
    
    def _get_current_prices(self):
        """Get current prices for all stocks."""
        return self.price_data.iloc[self.current_step].values.astype(np.float32)
    
    def _get_price_returns(self, lookback=1):
        """Get price returns over lookback period."""
        if self.current_step < lookback:
            return np.zeros(self.n_stocks, dtype=np.float32)
        
        prev_prices = self.price_data.iloc[self.current_step - lookback].values
        curr_prices = self.price_data.iloc[self.current_step].values
        returns = (curr_prices - prev_prices) / (prev_prices + 1e-10)
        return returns.astype(np.float32)
    
    def _get_portfolio_value(self):
        """Calculate total portfolio value."""
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.holdings * current_prices)
        return self.cash + stock_value
    
    def _get_obs(self):
        """Get current observation with meaningful features."""
        current_prices = self._get_current_prices()
        portfolio_value = self._get_portfolio_value()
        
        # Normalize by initial values
        normalized_portfolio = portfolio_value / self.initial_cash
        normalized_cash = self.cash / self.initial_cash
        
        # Holdings as percentage of portfolio
        stock_values = self.holdings * current_prices
        holdings_pct = stock_values / (portfolio_value + 1e-10)
        
        # Price returns over different periods
        returns_5d = self._get_price_returns(5)
        returns_20d = self._get_price_returns(20)
        
        obs = np.concatenate([
            [normalized_portfolio],   # Total portfolio value (normalized)
            [normalized_cash],         # Cash (normalized)
            holdings_pct,              # Stock holdings as % of portfolio
            returns_5d,                # 5-day returns
            returns_20d                # 20-day returns
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self):
        """Get additional info."""
        current_prices = self._get_current_prices()
        portfolio_value = self._get_portfolio_value()
        
        info = {
            "portfolio_value": float(portfolio_value),
            "cash": float(self.cash),
            "total_return_pct": float((portfolio_value - self.initial_cash) / self.initial_cash * 100),
            "stock_values": {
                ticker: float(self.holdings[i] * current_prices[i])
                for i, ticker in enumerate(self.MAG7_TICKERS)
            },
            "holdings": {
                ticker: float(self.holdings[i])
                for i, ticker in enumerate(self.MAG7_TICKERS)
            },
            "prices": {
                ticker: float(current_prices[i])
                for i, ticker in enumerate(self.MAG7_TICKERS)
            }
        }
        
        return info
    
    def _do_action(self, action, current_prices):
        """
        Execute trading actions with improved action space.
        
        Action interpretation:
        - 1: Sell 10% of holdings for that stock
        - 0: Hold (do nothing)
        - 2: Buy stock with 10% of available cash
        """
        total_cost = 0.0
        
        for i in range(self.n_stocks):
            action_type = action[i]
            price = current_prices[i]
            
            if action_type == 2:  # Buy with 10% of cash
                budget = self.cash * 0.1
                shares_to_buy = int(budget / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    transaction_fee = cost * self.transaction_cost_pct
                    total_needed = cost + transaction_fee
                    
                    if self.cash >= total_needed:
                        self.holdings[i] += shares_to_buy
                        self.cash -= total_needed
                        total_cost += transaction_fee
                    
            elif action_type == 1:  # Sell 10% of holdings
                shares_to_sell = int(self.holdings[i] * 0.1)
                
                if shares_to_sell > 0:
                    revenue = shares_to_sell * price
                    transaction_fee = revenue * self.transaction_cost_pct
                    
                    self.holdings[i] -= shares_to_sell
                    self.cash += revenue - transaction_fee
                    total_cost += transaction_fee
            
            # action_type == 0 is Hold (do nothing)
        
        return total_cost
    
    def _compute_reward(self, new_value, prev_value, transaction_cost):
        """
        Compute reward based on portfolio returns.
        
        Key improvements:
        1. Use percentage returns instead of absolute changes
        2. Subtract transaction costs
        3. Add small penalty for holding too much cash
        """
        # Calculate return as percentage
        if prev_value > 0:
            portfolio_return_pct = (new_value - prev_value) / prev_value
        else:
            portfolio_return_pct = 0.0
        
        # Subtract transaction costs (normalized)
        cost_penalty = transaction_cost / self.initial_cash
        
        # Penalty for holding too much cash (encourage investment)
        cash_ratio = self.cash / new_value if new_value > 0 else 0
        cash_penalty = 0.0
        if cash_ratio > 0.5:  # If more than 50% in cash
            cash_penalty = (cash_ratio - 0.5) * 0.1
        
        # Final reward
        reward = 50 * portfolio_return_pct - cost_penalty - cash_penalty
        
        # Scale reward for better learning
        reward *= 100
        
        # Track for episode statistics
        self.episode_returns.append(portfolio_return_pct)
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        
        # Episode tracking
        self.episode_returns = []
        self.initial_portfolio_value = self.initial_cash
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment."""
        if self.current_step >= self.max_steps:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        
        # Store previous portfolio value
        prev_portfolio_value = self._get_portfolio_value()
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Execute actions
        transaction_cost = self._do_action(action, current_prices)
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value
        new_portfolio_value = self._get_portfolio_value()
        
        # Compute reward
        reward = self._compute_reward(new_portfolio_value, prev_portfolio_value, transaction_cost)
        
        # Check if episode is done
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Add episode statistics at the end
        if terminated:
            total_return = (new_portfolio_value - self.initial_cash) / self.initial_cash
            info["episode"] = {
                "total_return": float(total_return),
                "total_return_pct": float(total_return * 100),
                "final_value": float(new_portfolio_value),
                "sharpe_ratio": self._calculate_sharpe_ratio()
            }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio for the episode."""
        if len(self.episode_returns) < 2:
            return 0.0
        
        returns = np.array(self.episode_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return float(sharpe)
        return 0.0
    
    def render(self):
        """Render the environment to the screen."""
        if self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render one frame of the environment."""
        portfolio_value = self._get_portfolio_value()
        current_prices = self._get_current_prices()
        total_return = (portfolio_value - self.initial_cash) / self.initial_cash * 100
        
        print(f"\n{'='*70}")
        print(f"Day {self.current_step}/{self.max_steps}")
        print(f"Portfolio Value: ${portfolio_value:,.2f} ({total_return:+.2f}%)")
        print(f"Cash: ${self.cash:,.2f} ({self.cash/portfolio_value*100:.1f}% of portfolio)")
        print(f"\nHoldings:")
        for i, ticker in enumerate(self.MAG7_TICKERS):
            if self.holdings[i] > 0:
                value = self.holdings[i] * current_prices[i]
                pct = value / portfolio_value * 100
                print(f"  {ticker}: {self.holdings[i]:.0f} shares @ ${current_prices[i]:.2f} = ${value:,.2f} ({pct:.1f}%)")
        print(f"{'='*70}")
    
    def close(self):
        """Clean up resources."""
        pass
