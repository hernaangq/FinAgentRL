import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleFinancialEnv(gym.Env):
    """A simple financial trading environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Define action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: [cash, stock_owned, current_price]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        
        # Initialize state variables
        self.initial_cash = 10000.0
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.current_step = 0
        self.max_steps = 100
        
        # Simulated price data (random walk)
        self.prices = self._generate_prices()
        
    def _generate_prices(self):
        """Generate simple random walk price data."""
        np.random.seed(42)
        prices = [100.0]
        for _ in range(self.max_steps):
            change = np.random.randn() * 2
            prices.append(max(prices[-1] + change, 1.0))
        return np.array(prices)
    
    def _get_obs(self):
        """Get current observation."""
        current_price = self.prices[self.current_step]
        return np.array([self.cash, self.stock_owned, current_price], dtype=np.float32)
    
    def _get_info(self):
        """Get additional info."""
        current_price = self.prices[self.current_step]
        portfolio_value = self.cash + self.stock_owned * current_price
        return {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "stock_owned": self.stock_owned,
            "current_price": current_price
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.current_step = 0
        
        if seed is not None:
            np.random.seed(seed)
            self.prices = self._generate_prices()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment."""
        current_price = self.prices[self.current_step]
        
        # Execute action
        if action == 1:  # Buy
            if self.cash >= current_price:
                self.stock_owned += 1
                self.cash -= current_price
        elif action == 2:  # Sell
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.cash += current_price
        # action == 0 is hold, do nothing
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        next_price = self.prices[self.current_step]
        portfolio_value = self.cash + self.stock_owned * next_price
        reward = portfolio_value - self.initial_cash
        
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
        print(f"Step: {self.current_step}/{self.max_steps} | "
              f"Portfolio: ${info['portfolio_value']:.2f} | "
              f"Cash: ${info['cash']:.2f} | "
              f"Stock: {info['stock_owned']} @ ${info['current_price']:.2f}")
    
    def close(self):
        """Clean up resources."""
        pass
