import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class DynamicPricingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, product_data, demand_patterns):
        super(DynamicPricingEnv, self).__init__()
        
        self.product_data = product_data
        self.demand_patterns = demand_patterns
        
        # Extract product params
        self.base_price = product_data['base_price']
        self.elasticity = product_data['elasticity']
        self.product_name = product_data['product_name']
        
        # Action space: Price multiplier [0.5, 1.5] of base price
        # We use a continuous action space for precise pricing
        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(1,), dtype=np.float32)
        
        # Observation space:
        # [Current Price, Competitor Price, Day of Week (normalized), Last Demand (normalized)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([np.inf, np.inf, 1, np.inf]), 
            dtype=np.float32
        )
        
        self.state = None
        self.current_step = 0
        self.max_steps = 365 # Simulate a year
        
        # Simulation state variables
        self.current_price = self.base_price
        self.competitor_price = self.base_price
        self.day_of_week = 0
        self.last_demand = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.day_of_week = 0
        self.current_price = self.base_price
        # Competitor starts slightly different
        self.competitor_price = self.base_price * np.random.uniform(0.9, 1.1)
        self.last_demand = self._get_base_demand(self.day_of_week)
        
        self.state = np.array([
            self.current_price,
            self.competitor_price,
            self.day_of_week / 7.0,
            self.last_demand / 100.0 # Normalize roughly
        ], dtype=np.float32)
        
        return self.state, {}

    def step(self, action):
        # 1. Apply Action (Set Price)
        price_multiplier = np.clip(action[0], 0.5, 1.5)
        self.current_price = self.base_price * price_multiplier
        
        # 2. Simulate Competitor (Simple logic: they follow trend with noise)
        # If we are higher, they might undercut. If we are lower, they might match.
        # Random walk with mean reversion to base price
        drift = np.random.normal(0, 0.02)
        reversion = 0.1 * (self.base_price - self.competitor_price) / self.base_price
        self.competitor_price *= (1 + drift + reversion)
        
        # 3. Calculate Demand
        # Base demand from patterns
        base_demand = self._get_base_demand(self.day_of_week)
        
        # Price Elasticity Effect
        # Demand = Base * (P / P_base) ^ elasticity
        price_effect = (self.current_price / self.base_price) ** self.elasticity
        
        # Competitor Effect (Cross Elasticity)
        # If our price > competitor, demand drops.
        # We assume a cross elasticity of 2.0 (positive, as substitute goods)
        competitor_ratio = self.competitor_price / self.current_price
        competitor_effect = (competitor_ratio) ** 2.0 
        
        # Final Demand with noise
        demand = base_demand * price_effect * competitor_effect * np.random.uniform(0.9, 1.1)
        demand = max(0, demand)
        
        self.last_demand = demand
        
        # 4. Calculate Reward (Revenue)
        revenue = self.current_price * demand
        reward = revenue
        
        # 5. Update State
        self.current_step += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        self.state = np.array([
            self.current_price,
            self.competitor_price,
            self.day_of_week / 7.0,
            self.last_demand / 100.0
        ], dtype=np.float32)
        
        info = {
            "revenue": revenue,
            "demand": demand,
            "competitor_price": self.competitor_price
        }
        
        return self.state, reward, done, truncated, info

    def _get_base_demand(self, dow):
        # Get average demand for this day of week from data
        row = self.demand_patterns[
            (self.demand_patterns['product_id'] == self.product_data['product_id']) & 
            (self.demand_patterns['order_dow'] == dow)
        ]
        if not row.empty:
            return row['avg_demand'].values[0]
        return 10 # Default fallback
