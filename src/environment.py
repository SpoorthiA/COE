import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class RealisticPricingEnv(gym.Env):
    """
    Production-Grade Dynamic Pricing Environment with:
    - Supply Chain Constraints & Spoilage Optimization (Inventory + Expiry)
    - AI Guardrails & Risk Mitigation (Price bounds)
    - POMDP Handling via Frame Stacking (Temporal dependencies)
    """
    metadata = {'render_modes': None}

    def __init__(self, dataset, production_cost=5.0, initial_inventory=10000, price_change_penalty=100.0,
                 base_demand=100.0, cross_elasticity=-0.5, competitor_reaction_speed=0.5, competitor_noise=0.05,
                 # Supply Chain Constraints & Spoilage Parameters
                 days_to_expiry=30, spoilage_penalty_rate=1.5, low_inventory_threshold=0.2,
                 # AI Guardrails Parameters  
                 min_margin_pct=0.10, max_markup_factor=2.0, enable_safety_guardrails=True,
                 # POMDP Frame Stacking Parameters
                 frame_stack_size=4, enable_frame_stacking=True):
        super().__init__()
        self.dataset = dataset
        self.production_cost = production_cost
        self.initial_inventory = initial_inventory
        self.price_change_penalty = price_change_penalty
        self.base_demand = base_demand
        self.cross_elasticity = cross_elasticity
        self.competitor_reaction_speed = competitor_reaction_speed
        self.competitor_noise = competitor_noise
        
        # === Supply Chain Constraints & Spoilage Optimization ===
        self.initial_days_to_expiry = days_to_expiry
        self.spoilage_penalty_rate = spoilage_penalty_rate  # Multiplier on cost for expired inventory
        self.low_inventory_threshold = low_inventory_threshold  # % of initial inventory
        
        # === AI Guardrails & Risk Mitigation ===
        self.min_margin_pct = min_margin_pct  # Minimum margin (e.g., 10%)
        self.max_markup_factor = max_markup_factor  # Maximum price as multiple of base
        self.enable_safety_guardrails = enable_safety_guardrails
        
        # === POMDP Handling via Frame Stacking ===
        self.frame_stack_size = frame_stack_size
        self.enable_frame_stacking = enable_frame_stacking
        
        # Base observation: [price, demand, competitor_price, hour, dow, inventory, elasticity, 
        #                    days_to_expiry, inventory_ratio, urgency_score]
        self.base_obs_dim = 10
        
        if enable_frame_stacking:
            # Stacked observations for temporal pattern detection
            obs_dim = self.base_obs_dim * frame_stack_size
            self.observation_space = spaces.Box(
                low=np.array([-np.inf] * obs_dim, dtype=np.float32),
                high=np.array([np.inf] * obs_dim, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -2.5, 0, 0, 0], dtype=np.float32),
                high=np.array([100, 1000, 100, 23, 6, self.initial_inventory, -0.1, days_to_expiry, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        self.action_space = spaces.Discrete(7)
        
        # Frame stack buffer for temporal observations
        self.obs_buffer = None

    def _get_base_obs(self):
        """Get single-step observation with enhanced state variables."""
        inventory_ratio = self.inventory / self.initial_inventory
        
        # Urgency score: combines expiry pressure and low inventory signal
        expiry_urgency = 1.0 - (self.days_to_expiry / self.initial_days_to_expiry)
        inventory_urgency = 1.0 - inventory_ratio if inventory_ratio < self.low_inventory_threshold else 0.0
        urgency_score = max(expiry_urgency, inventory_urgency)
        
        return np.array([
            self.current_price,
            self.historical_demand,
            self.competitor_price,
            float(self.current_hour),
            float(self.current_dow),
            self.inventory,
            self.elasticity,
            float(self.days_to_expiry),
            inventory_ratio,
            urgency_score
        ], dtype=np.float32)

    def _get_obs(self):
        """Get observation with optional frame stacking for POMDP handling."""
        base_obs = self._get_base_obs()
        
        if self.enable_frame_stacking:
            self.obs_buffer.append(base_obs)
            # Stack all frames into single observation vector
            stacked = np.concatenate(list(self.obs_buffer))
            return stacked.astype(np.float32)
        else:
            return base_obs

    def _get_info(self):
        return {
            'price': float(self.current_price),
            'demand': float(self.historical_demand),
            'inventory': float(self.inventory),
            'competitor_price': float(self.competitor_price),
            'days_to_expiry': int(self.days_to_expiry),
            'spoiled_units': float(getattr(self, 'last_spoiled_units', 0)),
            'inventory_ratio': float(self.inventory / self.initial_inventory),
            'price_floor': float(self._get_price_floor()),
            'price_ceiling': float(self._get_price_ceiling())
        }
    
    def _get_price_floor(self):
        """AI Guardrail: Minimum price = cost + margin to prevent price gouging violations."""
        return self.production_cost * (1 + self.min_margin_pct)
    
    def _get_price_ceiling(self):
        """AI Guardrail: Maximum price cap to prevent exploitative pricing."""
        base_price = self.production_cost * 4  # Reasonable base price estimate
        return base_price * self.max_markup_factor
    
    def _apply_safety_guardrails(self, price):
        """Clip price to valid bounds before execution - critical for production systems."""
        if not self.enable_safety_guardrails:
            return price
        
        floor = self._get_price_floor()
        ceiling = self._get_price_ceiling()
        return float(np.clip(price, floor, ceiling))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.max_steps = 365 
        self.current_hour = self.np_random.integers(0, 24)
        self.current_dow = self.np_random.integers(0, 7)
        self.inventory = self.initial_inventory
        
        # === Supply Chain State Reset ===
        self.days_to_expiry = self.initial_days_to_expiry
        self.total_spoiled_units = 0
        self.last_spoiled_units = 0
        self.total_sold_units = 0
        self.total_revenue = 0
        self.total_spoilage_penalty = 0

        elasticity_values = list(self.dataset.price_elasticity_map.values())
        self.elasticity = elasticity_values[self.np_random.integers(0, len(elasticity_values))]

        self.min_price = max(1.0, self.production_cost)
        self.max_price = 100.0
        self.current_price = float(self.np_random.uniform(self.min_price * 1.2, self.max_price * 0.8))
        self.competitor_price = float(self.np_random.uniform(self.min_price * 1.1, self.max_price * 0.9))
        self.historical_demand = float(self._get_base_demand())
        
        # === Initialize Frame Stack Buffer for POMDP ===
        if self.enable_frame_stacking:
            initial_obs = self._get_base_obs()
            self.obs_buffer = deque([initial_obs] * self.frame_stack_size, maxlen=self.frame_stack_size)

        return self._get_obs(), self._get_info()

    def step(self, action):
        price_adjustments = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
        delta_price = price_adjustments[action]

        old_price = self.current_price
        self.current_price *= (1 + delta_price)
        self.current_price = float(max(self.min_price, min(self.max_price, self.current_price)))
        
        # === AI Guardrails: Apply Safety Bounds ===
        self.current_price = self._apply_safety_guardrails(self.current_price)

        base_demand = self._get_base_demand()
        price_ratio = self.current_price / old_price if old_price > 0 else 1.0
        demand = base_demand * (price_ratio ** self.elasticity)

        # Demand is affected by the competitor's price from the *previous* step
        competitive_ratio = self.current_price / self.competitor_price if self.competitor_price > 0 else 1.0
        demand *= (competitive_ratio ** self.cross_elasticity)
        
        # === Supply Chain Constraint: Units sold cannot exceed inventory ===
        demand = min(demand, self.inventory)
        if not np.isfinite(demand) or demand < 0:
            demand = 0.0
        
        # === Spoilage Optimization: Expiry-driven dynamics ===
        self.days_to_expiry -= 1
        spoiled_units = 0
        spoilage_penalty = 0
        
        if self.days_to_expiry <= 0:
            # All remaining inventory spoils - catastrophic loss
            spoiled_units = self.inventory
            spoilage_penalty = spoiled_units * self.production_cost * self.spoilage_penalty_rate
            self.inventory = 0
            self.total_spoilage_penalty += spoilage_penalty
        else:
            # Gradual spoilage as expiry approaches (simulates quality degradation)
            if self.days_to_expiry <= 5:
                spoilage_rate = 0.02 * (6 - self.days_to_expiry)  # Up to 10% daily near expiry
                spoiled_units = self.inventory * spoilage_rate
                spoilage_penalty = spoiled_units * self.production_cost * self.spoilage_penalty_rate
                self.inventory -= spoiled_units
                self.total_spoilage_penalty += spoilage_penalty
        
        self.last_spoiled_units = spoiled_units
        self.total_spoiled_units += spoiled_units
        
        # Update inventory after sales
        self.inventory -= demand
        self.total_sold_units += demand

        # === Reward Calculation with Spoilage Penalties ===
        revenue = (self.current_price - self.production_cost) * demand
        self.total_revenue += revenue
        
        reward = float(revenue)
        reward -= abs(delta_price) * self.price_change_penalty
        reward -= spoilage_penalty  # Penalize spoilage
        
        # Bonus for selling near-expiry inventory (encourages proactive markdown)
        if self.days_to_expiry <= 5 and demand > 0:
            urgency_bonus = demand * 0.5  # Small bonus for moving near-expiry stock
            reward += urgency_bonus

        # Update competitor price *after* our move
        self._update_competitor_price(old_price)

        self.current_step += 1
        self.historical_demand = float(demand)
        
        # Episode terminates on: max steps, zero inventory, or full spoilage
        terminated = (self.current_step >= self.max_steps or 
                      self.inventory <= 0 or 
                      self.days_to_expiry <= 0)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_base_demand(self):
        temporal_factor = self.dataset.demand_patterns.get((self.current_hour, self.current_dow), 1.0)
        base_demand = self.base_demand * temporal_factor
        return float(base_demand * (1 + self.np_random.normal(0, 0.1)))

    def _update_competitor_price(self, agent_old_price):
        # Competitor reacts to the agent's price from the previous step (agent_old_price)
        # Strategy: slightly undercut if the agent's price is lower, or slowly increase if agent's price is higher.
        reaction_speed = self.competitor_reaction_speed 
        price_diff_ratio = (agent_old_price - self.competitor_price) / self.competitor_price
        
        adjustment_factor = reaction_speed * price_diff_ratio
        
        # Add some noise to make it less predictable
        noise = self.np_random.normal(0, self.competitor_noise)
        
        self.competitor_price *= (1 + adjustment_factor + noise)
        
        # Ensure competitor price stays within reasonable bounds
        self.competitor_price = float(max(self.min_price, min(self.max_price, self.competitor_price)))