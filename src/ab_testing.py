"""
Counterfactual Evaluation & A/B Testing Framework
==================================================
Production-grade Digital Twin Simulation for comparing pricing strategies
with synchronized random seeds for statistically valid comparisons.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environment import RealisticPricingEnv


@dataclass
class EpisodeMetrics:
    """Comprehensive metrics captured during a single episode."""
    total_reward: float = 0.0
    total_revenue: float = 0.0
    total_units_sold: float = 0.0
    total_spoiled_units: float = 0.0
    total_spoilage_penalty: float = 0.0
    final_inventory: float = 0.0
    inventory_turnover: float = 0.0  # % of inventory sold
    avg_price: float = 0.0
    price_volatility: float = 0.0
    avg_margin: float = 0.0
    episode_length: int = 0
    termination_reason: str = ""
    
    # Time series for visualization
    prices: List[float] = field(default_factory=list)
    demands: List[float] = field(default_factory=list)
    inventories: List[float] = field(default_factory=list)
    days_to_expiry: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    competitor_prices: List[float] = field(default_factory=list)


@dataclass
class ABTestResult:
    """Results from a complete A/B test comparison."""
    agent_name: str
    episodes: int
    seed: int
    
    # Aggregate metrics
    avg_reward: float = 0.0
    std_reward: float = 0.0
    avg_revenue: float = 0.0
    avg_units_sold: float = 0.0
    avg_spoilage: float = 0.0
    avg_inventory_turnover: float = 0.0
    avg_price_volatility: float = 0.0
    
    # Per-episode data
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    
    # Statistical confidence
    confidence_interval_95: tuple = (0.0, 0.0)


class ABTestingFramework:
    """
    Digital Twin Simulation Framework for Counterfactual Evaluation
    
    Enables head-to-head comparison of pricing strategies under identical
    market conditions using synchronized random seeds.
    """
    
    def __init__(self, dataset, env_params: Dict[str, Any]):
        self.dataset = dataset
        self.env_params = env_params
        self.results: Dict[str, ABTestResult] = {}
    
    def _create_env(self, seed: int) -> DummyVecEnv:
        """Create environment with specific seed for reproducibility."""
        env_lambda = lambda: RealisticPricingEnv(self.dataset, **self.env_params)
        vec_env = DummyVecEnv([env_lambda])
        vec_env.seed(seed)
        return vec_env
    
    def _run_episode(self, env: DummyVecEnv, policy_fn: Callable, seed: int) -> EpisodeMetrics:
        """
        Run a single episode and collect comprehensive metrics.
        
        Args:
            env: The vectorized environment
            policy_fn: Function that takes observation and returns action
            seed: Random seed for this episode
        """
        metrics = EpisodeMetrics()
        
        obs = env.reset()
        # Set seed after reset for consistent episode start
        env.seed(seed)
        
        done = False
        production_cost = self.env_params.get('production_cost', 5.0)
        initial_inventory = self.env_params.get('initial_inventory', 10000)
        
        while not done:
            action = policy_fn(obs)
            obs, reward, done_arr, info_arr = env.step(action)
            
            done = done_arr[0]
            info = info_arr[0]
            
            # Collect time series data
            metrics.rewards.append(float(reward[0]))
            metrics.prices.append(info['price'])
            metrics.demands.append(info['demand'])
            metrics.inventories.append(info['inventory'])
            metrics.days_to_expiry.append(info.get('days_to_expiry', 0))
            metrics.competitor_prices.append(info['competitor_price'])
            
            metrics.total_reward += float(reward[0])
            metrics.total_units_sold += info['demand']
            metrics.total_spoiled_units += info.get('spoiled_units', 0)
            metrics.episode_length += 1
        
        # Calculate aggregate metrics
        metrics.final_inventory = info['inventory']
        metrics.inventory_turnover = (initial_inventory - metrics.final_inventory - metrics.total_spoiled_units) / initial_inventory
        metrics.avg_price = np.mean(metrics.prices) if metrics.prices else 0
        metrics.price_volatility = np.std(metrics.prices) if metrics.prices else 0
        metrics.avg_margin = (metrics.avg_price - production_cost) / metrics.avg_price if metrics.avg_price > 0 else 0
        metrics.total_revenue = metrics.total_units_sold * metrics.avg_price
        
        # Determine termination reason
        if metrics.final_inventory <= 0:
            metrics.termination_reason = "stockout"
        elif info.get('days_to_expiry', 1) <= 0:
            metrics.termination_reason = "expiry"
        else:
            metrics.termination_reason = "max_steps"
        
        return metrics
    
    def run_ab_test(
        self,
        agent_name: str,
        policy_fn: Callable,
        n_episodes: int = 50,
        base_seed: int = 42
    ) -> ABTestResult:
        """
        Run A/B test for a single agent/policy.
        
        Args:
            agent_name: Identifier for this agent
            policy_fn: Function(obs) -> action
            n_episodes: Number of episodes to run
            base_seed: Starting seed for reproducibility
            
        Returns:
            ABTestResult with comprehensive metrics
        """
        result = ABTestResult(
            agent_name=agent_name,
            episodes=n_episodes,
            seed=base_seed
        )
        
        all_rewards = []
        
        for episode in range(n_episodes):
            episode_seed = base_seed + episode
            env = self._create_env(episode_seed)
            
            metrics = self._run_episode(env, policy_fn, episode_seed)
            result.episode_metrics.append(metrics)
            all_rewards.append(metrics.total_reward)
            
            env.close()
        
        # Calculate aggregate statistics
        result.avg_reward = np.mean(all_rewards)
        result.std_reward = np.std(all_rewards)
        result.avg_revenue = np.mean([m.total_revenue for m in result.episode_metrics])
        result.avg_units_sold = np.mean([m.total_units_sold for m in result.episode_metrics])
        result.avg_spoilage = np.mean([m.total_spoiled_units for m in result.episode_metrics])
        result.avg_inventory_turnover = np.mean([m.inventory_turnover for m in result.episode_metrics])
        result.avg_price_volatility = np.mean([m.price_volatility for m in result.episode_metrics])
        
        # 95% confidence interval
        se = result.std_reward / np.sqrt(n_episodes)
        result.confidence_interval_95 = (
            result.avg_reward - 1.96 * se,
            result.avg_reward + 1.96 * se
        )
        
        self.results[agent_name] = result
        return result
    
    def compare_agents(self, agent_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate comparison table for specified agents.
        
        Args:
            agent_names: List of agent names to compare (default: all)
            
        Returns:
            DataFrame with side-by-side comparison
        """
        if agent_names is None:
            agent_names = list(self.results.keys())
        
        comparison_data = []
        for name in agent_names:
            if name not in self.results:
                continue
            r = self.results[name]
            comparison_data.append({
                'Agent': name,
                'Avg Reward': f"${r.avg_reward:,.0f}",
                'Std Dev': f"${r.std_reward:,.0f}",
                '95% CI': f"(${r.confidence_interval_95[0]:,.0f}, ${r.confidence_interval_95[1]:,.0f})",
                'Avg Revenue': f"${r.avg_revenue:,.0f}",
                'Units Sold': f"{r.avg_units_sold:,.0f}",
                'Spoilage': f"{r.avg_spoilage:,.0f}",
                'Inventory Turnover': f"{r.avg_inventory_turnover:.1%}",
                'Price Volatility': f"${r.avg_price_volatility:.2f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_statistical_significance(self, agent_a: str, agent_b: str) -> Dict[str, Any]:
        """
        Perform statistical significance test between two agents.
        
        Returns:
            Dictionary with t-statistic, p-value, and interpretation
        """
        from scipy import stats
        
        if agent_a not in self.results or agent_b not in self.results:
            return {"error": "Agent not found in results"}
        
        rewards_a = [m.total_reward for m in self.results[agent_a].episode_metrics]
        rewards_b = [m.total_reward for m in self.results[agent_b].episode_metrics]
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2)
        cohens_d = (np.mean(rewards_a) - np.mean(rewards_b)) / pooled_std if pooled_std > 0 else 0
        
        interpretation = "not significant"
        if p_value < 0.05:
            interpretation = "significant" if p_value < 0.01 else "marginally significant"
        
        winner = agent_a if np.mean(rewards_a) > np.mean(rewards_b) else agent_b
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "interpretation": interpretation,
            "winner": winner,
            "mean_diff": np.mean(rewards_a) - np.mean(rewards_b),
            "percent_improvement": ((np.mean(rewards_a) / np.mean(rewards_b)) - 1) * 100 if np.mean(rewards_b) != 0 else 0
        }


def create_rl_policy(model):
    """Create policy function from trained RL model."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def create_fixed_policy(action: int = 3):
    """Create fixed-price policy (action 3 = no change)."""
    def policy(obs):
        return np.array([action])
    return policy


def create_random_policy(action_space_n: int = 7):
    """Create random action policy."""
    def policy(obs):
        return np.array([np.random.randint(0, action_space_n)])
    return policy


def create_markdown_policy(urgency_threshold: float = 0.3, markdown_action: int = 0):
    """
    Rule-based markdown policy that reduces price as expiry approaches.
    
    This represents a common industry heuristic for perishable goods.
    """
    def policy(obs):
        # obs structure: [..., days_to_expiry, inventory_ratio, urgency_score]
        # With frame stacking, we need the last frame
        if len(obs.shape) > 1:
            obs = obs[0]  # Unwrap VecEnv
        
        # Get urgency score from the most recent observation
        base_obs_dim = 10
        if len(obs) > base_obs_dim:
            # Frame stacking enabled - get last frame
            urgency_score = obs[-1]  # Last element of last frame
        else:
            urgency_score = obs[-1] if len(obs) >= 10 else 0.5
        
        if urgency_score > urgency_threshold:
            return np.array([markdown_action])  # Aggressive markdown
        elif urgency_score > urgency_threshold * 0.5:
            return np.array([1])  # Moderate markdown
        else:
            return np.array([3])  # Hold price
    
    return policy


def create_adaptive_policy(production_cost: float = 5.0):
    """
    Adaptive policy that considers inventory, expiry, and competitor pricing.
    
    Represents sophisticated rule-based system used in industry.
    """
    def policy(obs):
        if len(obs.shape) > 1:
            obs = obs[0]
        
        base_obs_dim = 10
        if len(obs) > base_obs_dim:
            # Get values from last frame in stack
            frame_start = len(obs) - base_obs_dim
            current_price = obs[frame_start]
            competitor_price = obs[frame_start + 2]
            inventory_ratio = obs[frame_start + 8]
            urgency_score = obs[frame_start + 9]
        else:
            current_price = obs[0]
            competitor_price = obs[2]
            inventory_ratio = obs[8] if len(obs) > 8 else 0.5
            urgency_score = obs[9] if len(obs) > 9 else 0.0
        
        # Rule 1: Urgent markdown for expiring/low inventory
        if urgency_score > 0.6:
            return np.array([0])  # Maximum markdown
        
        # Rule 2: Competitive pricing
        price_diff_pct = (current_price - competitor_price) / competitor_price if competitor_price > 0 else 0
        
        if price_diff_pct > 0.15:  # We're 15% more expensive
            return np.array([1])  # Reduce price
        elif price_diff_pct < -0.15:  # We're 15% cheaper
            if inventory_ratio > 0.5:  # Still have stock
                return np.array([5])  # Increase price
            else:
                return np.array([3])  # Hold
        
        # Rule 3: Inventory-based pricing
        if inventory_ratio > 0.7:  # High inventory
            return np.array([2])  # Small discount
        elif inventory_ratio < 0.2:  # Low inventory
            return np.array([4])  # Small increase
        
        return np.array([3])  # Default: hold
    
    return policy
