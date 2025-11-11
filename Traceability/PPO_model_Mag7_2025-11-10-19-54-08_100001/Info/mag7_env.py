import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class Mag7TradingEnv(gym.Env):
    """
    Multi-stock trading environment for Magnificent 7 stocks using real market data.
    
    The agent can buy, hold, or sell any of the Mag7 stocks to maximize portfolio value.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Magnificent 7 stocks
    MAG7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    def __init__(self, initial_cash=10000.0, lookback_period="1y", render_mode=None):
        """
        Initialize the Mag7 trading environment.
        
        Args:
            initial_cash: Starting cash amount
            lookback_period: Period to download data (e.g., "1y", "2y", "6mo")
            render_mode: Rendering mode ("human" or None)
        """
        super().__init__()
        
        self.initial_cash = initial_cash
        self.lookback_period = lookback_period
        self.render_mode = render_mode
        self.n_stocks = len(self.MAG7_TICKERS)
        
        # Action space: For each stock, we can buy (1), hold (0), or sell (-1)
        # We'll use MultiDiscrete: each stock has 3 possible actions
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        
        # Observation space:
        # - Cash (1)
        # - Stock holdings for each stock (7)
        # - Current prices for each stock (7)
        # - Price changes (returns) for each stock (7)
        # Total: 1 + 7 + 7 + 7 = 22 features
        obs_dim = 1 + (self.n_stocks * 3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Risk-aware reward parameters
        self.volatility_penalty = 1.0      # λ_v
        self.drawdown_penalty = 2.0        # λ_d
        self.trading_cost_penalty = 0.05   # λ_c
        self.rolling_window = 20           # Window for volatility calculation
        
        # Tracking variables
        self.portfolio_history = []        # Track V_t over time
        self.return_history = []           # Track R_t over time
        self.max_portfolio_value = initial_cash  # V_max

        
        # Download historical data
        print("Downloading Mag7 historical data...")
        self.price_data = self._download_data()
        self.max_steps = len(self.price_data) - 1
        
        # Initialize state
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        
    def _download_data(self):
        """Download historical price data for Mag7 stocks."""
        data = {}
        
        for ticker in self.MAG7_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=self.lookback_period)
                data[ticker] = hist['Close'].values
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                # Fallback to random data if download fails
                data[ticker] = np.random.randn(252) * 10 + 100
        
        # Ensure all stocks have the same length
        min_len = min(len(v) for v in data.values())
        for ticker in self.MAG7_TICKERS:
            data[ticker] = data[ticker][-min_len:]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Downloaded {len(df)} days of data for {self.n_stocks} stocks")
        
        return df
    
    def _get_current_prices(self):
        """Get current prices for all stocks."""
        return self.price_data.iloc[self.current_step].values.astype(np.float32)
    
    def _get_price_returns(self):
        """Get price returns (percentage change from previous day)."""
        if self.current_step == 0:
            return np.zeros(self.n_stocks, dtype=np.float32)
        
        prev_prices = self.price_data.iloc[self.current_step - 1].values
        curr_prices = self.price_data.iloc[self.current_step].values
        returns = (curr_prices - prev_prices) / prev_prices
        return returns.astype(np.float32)
    
    def _get_obs(self):
        """Get current observation."""
        current_prices = self._get_current_prices()
        price_returns = self._get_price_returns()
        
        obs = np.concatenate([
            [self.cash],
            self.holdings,
            current_prices,
            price_returns
        ]).astype(np.float32)
        
        return obs
    
    def _get_portfolio_value(self):
        """Calculate total portfolio value."""
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.holdings * current_prices)
        return self.cash + stock_value
    
    def _get_info(self):
        """Get additional info."""
        current_prices = self._get_current_prices()
        portfolio_value = self._get_portfolio_value()
        
        info = {
            "portfolio_value": float(portfolio_value),
            "cash": float(self.cash),
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
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def _do_action(self, action, current_prices):
        """Helper function to execute actions."""
        # Convert action encoding: 0->sell, 1->hold, 2->buy
        # To: -1->sell, 0->hold, 1->buy
        action_decoded = action - 1
        
        current_prices = self._get_current_prices()
        
        # Execute actions for each stock
        for i, act in enumerate(action_decoded):
            if act == 1:  # Buy
                if self.cash >= current_prices[i]:
                    self.holdings[i] += 1
                    self.cash -= current_prices[i]
            elif act == -1:  # Sell
                if self.holdings[i] > 0:
                    self.holdings[i] -= 1
                    self.cash += current_prices[i]
            # act == 0 is hold, do nothing

        n_trades = np.sum(action_decoded != 0)
        return n_trades

    def _compute_reward_legacy(self, new_value, prev_value):
        """Legacy reward: simple portfolio value change."""
        return new_value - prev_value
    
    def _compute_reward_risk_aware(self, new_value, prev_value, n_trades):
        """Risk-aware reward with volatility and drawdown penalties."""
        
        # 1. Calculate portfolio return
        if prev_value > 0:
            portfolio_return = (new_value - prev_value) / prev_value
        else:
            portfolio_return = 0.0
        
        self.return_history.append(portfolio_return)
        self.portfolio_history.append(new_value)
        
        # 2. Calculate volatility (standard deviation of recent returns)
        if len(self.return_history) >= 2:
            recent_returns = self.return_history[-min(self.rolling_window, len(self.return_history)):]
            volatility = np.std(recent_returns)
        else:
            volatility = 0.0
        
        # 3. Calculate drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, new_value)
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - new_value) / self.max_portfolio_value
            drawdown = max(0, drawdown)  # Only penalize positive drawdowns
        else:
            drawdown = 0.0
        
        # 4. Calculate trading activity penalty
        trading_ratio = n_trades / self.n_stocks
        
        # 5. Combine into risk-aware reward
        reward = (
            portfolio_return 
            - self.volatility_penalty * volatility
            - self.drawdown_penalty * drawdown
            - self.trading_cost_penalty * trading_ratio
        )
        
        return reward

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Array of actions for each stock [0=sell, 1=hold, 2=buy]
        """
        current_prices = self._get_current_prices()
        
        n_trades = self._do_action(action, current_prices)

        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self._get_portfolio_value()
        # Move to next step
        self.current_step += 1
        # Calculate new portfolio value and reward
        new_portfolio_value = self._get_portfolio_value()
        reward = self._compute_reward_risk_aware(new_portfolio_value, prev_portfolio_value, n_trades)

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render a single frame."""
        info = self._get_info()
        print(f"\n{'='*80}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
        print(f"Cash: ${info['cash']:,.2f}")
        print(f"\nHoldings:")
        for ticker in self.MAG7_TICKERS:
            shares = info['holdings'][ticker]
            value = info['stock_values'][ticker]
            price = info['prices'][ticker]
            if shares > 0:
                print(f"  {ticker}: {shares:.0f} shares @ ${price:.2f} = ${value:,.2f}")
        print(f"{'='*80}")
    
    def close(self):
        """Clean up resources."""
        pass
