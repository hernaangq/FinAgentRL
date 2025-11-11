import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class Mag7TradingEnv(gym.Env):
    """
    Multi-stock trading environment for Magnificent 7 stocks using real market data.
    
    The agent can buy, hold, or sell any of the Mag7 stocks to maximize portfolio value.
    Supports multiple reward functions: legacy, risk_aware, and sharpe.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Magnificent 7 stocks
    MAG7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    def __init__(self, initial_cash=10000.0, lookback_period="1y", render_mode=None, 
                 reward_type="risk_aware", volatility_penalty=1.0, drawdown_penalty=2.0,
                 trading_cost_penalty=0.05, rolling_window=20):
        """
        Initialize the Mag7 trading environment.
        
        Args:
            initial_cash: Starting cash amount
            lookback_period: Period to download data (e.g., "1y", "2y", "6mo")
            render_mode: Rendering mode ("human" or None)
            reward_type: Type of reward function ("legacy", "risk_aware", "sharpe")
            volatility_penalty: Coefficient for volatility penalty (lambda_v)
            drawdown_penalty: Coefficient for drawdown penalty (lambda_d)
            trading_cost_penalty: Coefficient for trading cost penalty (lambda_c)
            rolling_window: Window size for volatility calculation
        """
        super().__init__()
        
        self.initial_cash = initial_cash
        self.lookback_period = lookback_period
        self.render_mode = render_mode
        self.n_stocks = len(self.MAG7_TICKERS)
        
        # Reward function configuration
        self.reward_type = reward_type
        self.volatility_penalty = volatility_penalty
        self.drawdown_penalty = drawdown_penalty
        self.trading_cost_penalty = trading_cost_penalty
        self.rolling_window = rolling_window
        
        # Action space: For each stock, we can sell (0), hold (1), or buy (2)
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
        
        # Download historical data
        print(f"Downloading Mag7 historical data ({lookback_period})...")
        self.price_data = self._download_data()
        self.max_steps = len(self.price_data) - 1
        
        # Initialize state
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        
        # Risk tracking variables
        self.portfolio_history = [initial_cash]
        self.return_history = []
        self.max_portfolio_value = initial_cash
        
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
        
        # Ensure all components are numpy arrays
        cash_array = np.array([self.cash], dtype=np.float32)
        
        obs = np.concatenate([
            cash_array,           # [1] - cash
            self.holdings,        # [7] - holdings
            current_prices,       # [7] - current prices
            price_returns         # [7] - price returns
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
    
    def _do_action(self, action, current_prices):
        """
        Execute trading actions for all stocks.
        
        Args:
            action: Array of actions for each stock (0=sell, 1=hold, 2=buy)
            current_prices: Current prices for all stocks
            
        Returns:
            Number of trades executed
        """
        n_trades = 0
        
        for i in range(self.n_stocks):
            action_type = action[i]
            price = current_prices[i]
            
            if action_type == 2:  # Buy
                if self.cash >= price:
                    self.holdings[i] += 1
                    self.cash -= price
                    n_trades += 1
                    
            elif action_type == 0:  # Sell
                if self.holdings[i] > 0:
                    self.holdings[i] -= 1
                    self.cash += price
                    n_trades += 1
            
            # action_type == 1 is Hold (do nothing)
        
        return n_trades
    
    def _compute_reward_legacy(self, new_value, prev_value):
        """
        Legacy reward: simple portfolio value change.
        
        Args:
            new_value: Current portfolio value
            prev_value: Previous portfolio value
            
        Returns:
            Reward (dollar change in portfolio value)
        """
        return new_value - prev_value
    
    def _compute_reward_risk_aware(self, new_value, prev_value, n_trades):
        """
        Risk-aware reward with volatility and drawdown penalties.
        
        Args:
            new_value: Current portfolio value
            prev_value: Previous portfolio value
            n_trades: Number of trades executed this step
            
        Returns:
            Risk-adjusted reward
        """
        # 1. Calculate portfolio return (normalized)
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
        trading_ratio = n_trades / self.n_stocks if self.n_stocks > 0 else 0
        
        # 5. Combine into risk-aware reward
        reward = (
            portfolio_return 
            - self.volatility_penalty * volatility
            - self.drawdown_penalty * drawdown
            - self.trading_cost_penalty * trading_ratio
        )
        
        return reward
    
    def _compute_reward_sharpe(self, new_value, prev_value, n_trades):
        """
        Sharpe-inspired reward: return/volatility ratio with penalties.
        
        Args:
            new_value: Current portfolio value
            prev_value: Previous portfolio value
            n_trades: Number of trades executed this step
            
        Returns:
            Sharpe-like reward
        """
        # Calculate portfolio return
        if prev_value > 0:
            portfolio_return = (new_value - prev_value) / prev_value
        else:
            portfolio_return = 0.0
        
        self.return_history.append(portfolio_return)
        self.portfolio_history.append(new_value)
        
        # Calculate volatility
        if len(self.return_history) >= 2:
            recent_returns = self.return_history[-min(self.rolling_window, len(self.return_history)):]
            volatility = np.std(recent_returns) + 1e-6  # Add epsilon to avoid division by zero
        else:
            volatility = 1e-6
        
        # Sharpe-like ratio
        sharpe_reward = portfolio_return / volatility
        
        # Calculate drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, new_value)
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - new_value) / self.max_portfolio_value
            drawdown = max(0, drawdown)
        else:
            drawdown = 0.0
        
        # Trading activity penalty
        trading_ratio = n_trades / self.n_stocks if self.n_stocks > 0 else 0
        
        # Combine
        reward = (
            sharpe_reward
            - self.drawdown_penalty * drawdown
            - self.trading_cost_penalty * trading_ratio
        )
        
        return reward
    
    def _compute_reward(self, new_value, prev_value, n_trades=0):
        """
        Compute reward based on selected reward type.
        
        Args:
            new_value: Current portfolio value
            prev_value: Previous portfolio value
            n_trades: Number of trades executed this step
            
        Returns:
            Computed reward
        """
        if self.reward_type == "legacy":
            return self._compute_reward_legacy(new_value, prev_value)
        elif self.reward_type == "risk_aware":
            return self._compute_reward_risk_aware(new_value, prev_value, n_trades)
        elif self.reward_type == "sharpe":
            return self._compute_reward_sharpe(new_value, prev_value, n_trades)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        
        # Reset tracking for risk-aware rewards
        self.portfolio_history = [self.initial_cash]
        self.return_history = []
        self.max_portfolio_value = self.initial_cash
        
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
        
        # Execute actions and count trades
        n_trades = self._do_action(action, current_prices)
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value
        new_portfolio_value = self._get_portfolio_value()
        
        # Compute reward (pass n_trades for risk-aware calculation)
        reward = self._compute_reward(new_portfolio_value, prev_portfolio_value, n_trades)
        
        # Check if episode is done
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Add risk metrics to info for risk-aware rewards
        if self.reward_type in ["risk_aware", "sharpe"]:
            if len(self.return_history) >= 2:
                recent_returns = self.return_history[-min(self.rolling_window, len(self.return_history)):]
                volatility = np.std(recent_returns)
            else:
                volatility = 0.0
                
            drawdown = (self.max_portfolio_value - new_portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0.0
            
            info["risk_metrics"] = {
                "volatility": float(volatility),
                "drawdown": float(max(0, drawdown)),
                "max_portfolio_value": float(self.max_portfolio_value),
                "n_trades": int(n_trades),
                "portfolio_return": float(self.return_history[-1]) if self.return_history else 0.0
            }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment to the screen."""
        if self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render one frame of the environment."""
        portfolio_value = self._get_portfolio_value()
        current_prices = self._get_current_prices()
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"\nHoldings:")
        for i, ticker in enumerate(self.MAG7_TICKERS):
            if self.holdings[i] > 0:
                value = self.holdings[i] * current_prices[i]
                print(f"  {ticker}: {self.holdings[i]:.0f} shares @ ${current_prices[i]:.2f} = ${value:,.2f}")
        
        if self.reward_type in ["risk_aware", "sharpe"] and len(self.return_history) > 0:
            print(f"\nRisk Metrics:")
            if len(self.return_history) >= 2:
                recent_returns = self.return_history[-min(self.rolling_window, len(self.return_history)):]
                volatility = np.std(recent_returns)
                print(f"  Volatility: {volatility:.4f}")
            drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0.0
            print(f"  Drawdown: {max(0, drawdown)*100:.2f}%")
            print(f"  Max Portfolio Value: ${self.max_portfolio_value:,.2f}")
        
        print(f"{'='*60}")
    
    def close(self):
        """Clean up resources."""
        pass
