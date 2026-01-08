"""
Production-Grade Dynamic Pricing with Reinforcement Learning
=============================================================

This module demonstrates the four advanced modifications for real-world pricing systems:

1. POMDP Handling - Frame stacking for temporal pattern detection
2. Supply Chain Constraints & Spoilage Optimization - Inventory & expiry awareness
3. AI Guardrails & Risk Mitigation - Price bounds and safety constraints
4. Counterfactual Evaluation - A/B testing with synchronized seeds

Run the Streamlit dashboard: streamlit run app.py
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import RealisticPricingEnv
from src.data_loader import PricingDataset
from src.model import train_model, evaluate_pricing_policy
from src.baselines import run_baseline_policy, run_multiple_baselines
from src.ab_testing import (
    ABTestingFramework,
    create_rl_policy,
    create_fixed_policy,
    create_adaptive_policy,
    create_markdown_policy
)


def main():
    """
    Demonstrate the production-grade dynamic pricing system.
    """
    print("=" * 60)
    print("Production-Grade Dynamic Pricing with Reinforcement Learning")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/5] Loading pricing dataset...")
    dataset = PricingDataset(
        instacart_dir="data/instacart-market-basket-analysis",
        retail_path="data/online-retail-dataset/Online Retail Data Set.xlsx",
        retailrocket_path="data/ecommerce-dataset/events.csv"
    )
    
    # Environment configuration with all advanced features
    env_params = {
        # Core parameters
        "production_cost": 5.0,
        "initial_inventory": 10000,
        "base_demand": 100,
        "price_change_penalty": 100,
        
        # Supply Chain Constraints & Spoilage Optimization
        "days_to_expiry": 30,
        "spoilage_penalty_rate": 1.5,
        "low_inventory_threshold": 0.2,
        
        # AI Guardrails & Risk Mitigation
        "enable_safety_guardrails": True,
        "min_margin_pct": 0.10,  # 10% minimum margin
        "max_markup_factor": 2.0,  # Max 2x base price
        
        # POMDP Handling via Frame Stacking
        "enable_frame_stacking": True,
        "frame_stack_size": 4,  # Last 4 days of observations
        
        # Competitor dynamics
        "cross_elasticity": -0.5,
        "competitor_reaction_speed": 0.5,
        "competitor_noise": 0.05,
    }
    
    # Create environment
    print("\n[2/5] Creating enhanced environment...")
    env_lambda = lambda: RealisticPricingEnv(dataset, **env_params)
    vec_env = DummyVecEnv([env_lambda])
    
    print(f"      - Observation space: {vec_env.observation_space.shape}")
    print(f"      - Action space: {vec_env.action_space}")
    print(f"      - Frame stacking: {env_params['frame_stack_size']} frames")
    print(f"      - Safety guardrails: {'Enabled' if env_params['enable_safety_guardrails'] else 'Disabled'}")
    
    # Train RL agent
    print("\n[3/5] Training PPO agent (this may take a few minutes)...")
    model = train_model(vec_env, total_timesteps=30000)
    
    # Evaluate with comprehensive metrics
    print("\n[4/5] Evaluating trained agent...")
    rl_results = evaluate_pricing_policy(model, vec_env, n_episodes=20)
    
    print("\n--- RL Agent Performance ---")
    print(f"Average Reward:        ${rl_results['avg_reward']:,.0f}")
    print(f"Std Dev:               ${rl_results['std_reward']:,.0f}")
    print(f"Avg Price Volatility:  ${rl_results['avg_price_volatility']:.2f}")
    print(f"Inventory Turnover:    {rl_results['avg_inventory_turnover']:.1%}")
    print(f"Spoilage Rate:         {rl_results['avg_spoilage_rate']:.1%}")
    
    # A/B Testing Framework - Counterfactual Evaluation
    print("\n[5/5] Running A/B Testing Framework...")
    print("      (Comparing RL vs baselines under identical conditions)")
    
    ab_framework = ABTestingFramework(dataset, env_params)
    
    # Run tests with synchronized seeds
    seed = 42
    n_episodes = 20
    
    rl_result = ab_framework.run_ab_test(
        "RL Agent (PPO)", 
        create_rl_policy(model), 
        n_episodes=n_episodes, 
        base_seed=seed
    )
    
    fixed_result = ab_framework.run_ab_test(
        "Fixed Price", 
        create_fixed_policy(), 
        n_episodes=n_episodes, 
        base_seed=seed
    )
    
    adaptive_result = ab_framework.run_ab_test(
        "Adaptive Policy", 
        create_adaptive_policy(), 
        n_episodes=n_episodes, 
        base_seed=seed
    )
    
    markdown_result = ab_framework.run_ab_test(
        "Markdown Policy", 
        create_markdown_policy(), 
        n_episodes=n_episodes, 
        base_seed=seed
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("A/B TEST RESULTS - Counterfactual Evaluation")
    print("=" * 60)
    
    comparison_df = ab_framework.compare_agents()
    print(comparison_df.to_string(index=False))
    
    # Statistical significance
    print("\n--- Statistical Significance: RL vs Fixed ---")
    sig = ab_framework.get_statistical_significance("RL Agent (PPO)", "Fixed Price")
    if "error" not in sig:
        print(f"p-value:      {sig['p_value']:.4f}")
        print(f"Cohen's d:    {sig['cohens_d']:.2f}")
        print(f"Result:       {sig['interpretation'].title()}")
        print(f"Winner:       {sig['winner']}")
        print(f"Improvement:  {sig['percent_improvement']:.1f}%")
    
    print("\n" + "=" * 60)
    print("For the full interactive dashboard, run: streamlit run app.py")
    print("=" * 60)
    
    vec_env.close()


if __name__ == "__main__":
    main()
