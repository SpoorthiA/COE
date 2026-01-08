import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import RealisticPricingEnv
from src.data_loader import PricingDataset
from src.model import train_model, evaluate_pricing_policy
from src.baselines import run_baseline_policy, run_multiple_baselines
from src.ab_testing import (
    ABTestingFramework, 
    create_rl_policy, 
    create_fixed_policy, 
    create_random_policy,
    create_markdown_policy,
    create_adaptive_policy
)

st.set_page_config(layout="wide", page_title="Production-Grade RL Pricing Dashboard")

# --- Caching ---
@st.cache_resource
def load_data():
    return PricingDataset(
        instacart_dir="data/instacart-market-basket-analysis",
        retail_path="data/online-retail-dataset/Online Retail Data Set.xlsx",
        retailrocket_path="data/ecommerce-dataset/events.csv"
    )

@st.cache_resource
def get_trained_model(_env_params, total_timesteps):
    """Cached function to train the model."""
    env_params_dict = dict(_env_params)
    pricing_dataset = load_data()
    env_lambda = lambda: RealisticPricingEnv(pricing_dataset, **env_params_dict)
    env = DummyVecEnv([env_lambda])
    model = train_model(env, total_timesteps)
    return model

# --- UI Layout ---
st.title("🤖 Production-Grade Dynamic Pricing with Reinforcement Learning")
st.markdown("""
**Advanced Features:**
- 📦 **Supply Chain Constraints & Spoilage Optimization** - Inventory & expiry-aware pricing
- 🛡️ **AI Guardrails & Risk Mitigation** - Price bounds & safety constraints  
- 🔄 **POMDP Handling** - Frame stacking for temporal pattern detection
- 📊 **Counterfactual Evaluation** - A/B testing with synchronized seeds
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Simulation Configuration")
    if "run_sim" not in st.session_state:
        st.session_state.run_sim = False

    total_timesteps = st.slider("Training Timesteps", 10000, 100000, 50000, 5000)
    
    st.subheader("📦 Supply Chain Parameters")
    production_cost = st.number_input("Production Cost per Unit", 1.0, 20.0, 5.0, 0.5)
    initial_inventory = st.number_input("Initial Inventory", 1000, 50000, 10000, 1000)
    days_to_expiry = st.slider("Days to Expiry", 7, 60, 30, 1)
    spoilage_penalty_rate = st.slider("Spoilage Penalty Multiplier", 1.0, 3.0, 1.5, 0.1)
    
    st.subheader("🛡️ AI Guardrails")
    enable_guardrails = st.checkbox("Enable Safety Guardrails", value=True)
    min_margin_pct = st.slider("Minimum Margin %", 0.05, 0.30, 0.10, 0.01)
    max_markup_factor = st.slider("Maximum Price Markup Factor", 1.5, 3.0, 2.0, 0.1)
    
    st.subheader("🔄 POMDP Configuration")
    enable_frame_stacking = st.checkbox("Enable Frame Stacking", value=True)
    frame_stack_size = st.slider("Frame Stack Size", 2, 8, 4, 1) if enable_frame_stacking else 1
    
    st.subheader("🏪 Market Parameters")
    base_demand = st.slider("Base Demand per Step", 50, 500, 100, 10)
    price_change_penalty = st.slider("Price Change Penalty", 0, 1000, 100, 10)
    
    st.subheader("🏁 Competitor Dynamics")
    cross_elasticity = st.slider("Cross-Price Elasticity", -1.0, 0.0, -0.5, 0.05)
    competitor_reaction_speed = st.slider("Competitor Reaction Speed", 0.1, 1.0, 0.5, 0.1)
    competitor_noise = st.slider("Competitor Noise (Std. Dev)", 0.0, 0.2, 0.05, 0.01)
    
    st.subheader("📊 A/B Testing Configuration")
    n_test_episodes = st.slider("Test Episodes per Agent", 10, 100, 30, 5)
    ab_test_seed = st.number_input("Random Seed", 0, 9999, 42, 1)

    if st.button("🚀 Train & Run A/B Testing"):
        st.session_state.run_sim = True
    else:
        st.info("Adjust parameters and click the button to run.")

# --- Main Content Area ---
if st.session_state.run_sim:
    env_params = {
        "production_cost": production_cost,
        "initial_inventory": initial_inventory,
        "price_change_penalty": price_change_penalty,
        "base_demand": base_demand,
        "cross_elasticity": cross_elasticity,
        "competitor_reaction_speed": competitor_reaction_speed,
        "competitor_noise": competitor_noise,
        "days_to_expiry": days_to_expiry,
        "spoilage_penalty_rate": spoilage_penalty_rate,
        "min_margin_pct": min_margin_pct,
        "max_markup_factor": max_markup_factor,
        "enable_safety_guardrails": enable_guardrails,
        "frame_stack_size": frame_stack_size,
        "enable_frame_stacking": enable_frame_stacking,
    }
    
    with st.spinner("Loading dataset..."):
        pricing_dataset = load_data()

    # Create environment for training
    env_lambda = lambda: RealisticPricingEnv(pricing_dataset, **env_params)
    vec_env = DummyVecEnv([env_lambda])

    # Train RL model
    with st.spinner(f"Training RL Agent for {total_timesteps} timesteps..."):
        model = get_trained_model(tuple(env_params.items()), total_timesteps)

    # =====================================================
    # A/B TESTING DASHBOARD - Counterfactual Evaluation
    # =====================================================
    st.header("📊 Counterfactual Evaluation Dashboard")
    st.markdown("""
    **Digital Twin Simulation**: All agents are tested under identical market conditions 
    using synchronized random seeds for statistically valid comparison.
    """)
    
    with st.spinner("Running A/B Testing Framework..."):
        ab_framework = ABTestingFramework(pricing_dataset, env_params)
        
        # Run tests for all policies
        rl_result = ab_framework.run_ab_test(
            "RL Agent (PPO)", 
            create_rl_policy(model), 
            n_episodes=n_test_episodes,
            base_seed=int(ab_test_seed)
        )
        
        fixed_result = ab_framework.run_ab_test(
            "Fixed Price", 
            create_fixed_policy(), 
            n_episodes=n_test_episodes,
            base_seed=int(ab_test_seed)
        )
        
        random_result = ab_framework.run_ab_test(
            "Random", 
            create_random_policy(), 
            n_episodes=n_test_episodes,
            base_seed=int(ab_test_seed)
        )
        
        markdown_result = ab_framework.run_ab_test(
            "Markdown Policy", 
            create_markdown_policy(), 
            n_episodes=n_test_episodes,
            base_seed=int(ab_test_seed)
        )
        
        adaptive_result = ab_framework.run_ab_test(
            "Adaptive Policy", 
            create_adaptive_policy(production_cost), 
            n_episodes=n_test_episodes,
            base_seed=int(ab_test_seed)
        )

    # --- Summary Metrics ---
    st.subheader("🏆 Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    baseline_reward = fixed_result.avg_reward
    
    def format_delta(value, baseline):
        if baseline == 0:
            return "N/A"
        pct = ((value / baseline) - 1) * 100
        return f"{pct:+.1f}% vs Fixed"
    
    col1.metric(
        "🤖 RL Agent", 
        f"${rl_result.avg_reward:,.0f}",
        format_delta(rl_result.avg_reward, baseline_reward)
    )
    col2.metric(
        "📊 Fixed Price", 
        f"${fixed_result.avg_reward:,.0f}",
        "Baseline"
    )
    col3.metric(
        "🎲 Random", 
        f"${random_result.avg_reward:,.0f}",
        format_delta(random_result.avg_reward, baseline_reward)
    )
    col4.metric(
        "📉 Markdown", 
        f"${markdown_result.avg_reward:,.0f}",
        format_delta(markdown_result.avg_reward, baseline_reward)
    )
    col5.metric(
        "🧠 Adaptive", 
        f"${adaptive_result.avg_reward:,.0f}",
        format_delta(adaptive_result.avg_reward, baseline_reward)
    )

    # --- Detailed Comparison Table ---
    st.subheader("📈 Detailed A/B Test Results")
    comparison_df = ab_framework.compare_agents()
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # --- Statistical Significance ---
    st.subheader("📐 Statistical Significance Testing")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown("**RL Agent vs Fixed Price**")
        sig_test = ab_framework.get_statistical_significance("RL Agent (PPO)", "Fixed Price")
        if "error" not in sig_test:
            st.write(f"- **p-value**: {sig_test['p_value']:.4f}")
            st.write(f"- **Effect Size (Cohen's d)**: {sig_test['cohens_d']:.2f}")
            st.write(f"- **Result**: {sig_test['interpretation'].title()}")
            st.write(f"- **Winner**: {sig_test['winner']}")
            st.write(f"- **Improvement**: {sig_test['percent_improvement']:.1f}%")
    
    with col_stat2:
        st.markdown("**RL Agent vs Adaptive Policy**")
        sig_test2 = ab_framework.get_statistical_significance("RL Agent (PPO)", "Adaptive Policy")
        if "error" not in sig_test2:
            st.write(f"- **p-value**: {sig_test2['p_value']:.4f}")
            st.write(f"- **Effect Size (Cohen's d)**: {sig_test2['cohens_d']:.2f}")
            st.write(f"- **Result**: {sig_test2['interpretation'].title()}")
            st.write(f"- **Winner**: {sig_test2['winner']}")
            st.write(f"- **Improvement**: {sig_test2['percent_improvement']:.1f}%")

    # --- Visualization Section ---
    st.header("📊 Visual Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Reward Distribution", 
        "📦 Inventory & Spoilage", 
        "💰 Price Behavior",
        "🔄 Episode Timeline"
    ])
    
    with tab1:
        st.subheader("Reward Distribution Across Episodes")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart comparison
        agents = ["RL Agent", "Fixed", "Random", "Markdown", "Adaptive"]
        rewards = [rl_result.avg_reward, fixed_result.avg_reward, random_result.avg_reward,
                   markdown_result.avg_reward, adaptive_result.avg_reward]
        stds = [rl_result.std_reward, fixed_result.std_reward, random_result.std_reward,
                markdown_result.std_reward, adaptive_result.std_reward]
        
        colors = ['#2E8B57', '#4682B4', '#DC143C', '#FF8C00', '#9370DB']
        axes[0].bar(agents, rewards, yerr=stds, color=colors, capsize=5, alpha=0.8)
        axes[0].set_ylabel("Average Reward ($)")
        axes[0].set_title("Reward Comparison with 95% CI")
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Box plot
        reward_data = [
            [m.total_reward for m in rl_result.episode_metrics],
            [m.total_reward for m in fixed_result.episode_metrics],
            [m.total_reward for m in random_result.episode_metrics],
            [m.total_reward for m in markdown_result.episode_metrics],
            [m.total_reward for m in adaptive_result.episode_metrics]
        ]
        bp = axes[1].boxplot(reward_data, labels=agents, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1].set_ylabel("Total Episode Reward ($)")
        axes[1].set_title("Reward Distribution")
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Supply Chain Performance: Inventory Turnover & Spoilage")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inventory turnover
        turnovers = [rl_result.avg_inventory_turnover, fixed_result.avg_inventory_turnover,
                     random_result.avg_inventory_turnover, markdown_result.avg_inventory_turnover,
                     adaptive_result.avg_inventory_turnover]
        axes[0].bar(agents, [t * 100 for t in turnovers], color=colors, alpha=0.8)
        axes[0].set_ylabel("Inventory Turnover (%)")
        axes[0].set_title("Inventory Turnover Rate")
        axes[0].set_ylim(0, 100)
        
        # Spoilage comparison
        spoilages = [rl_result.avg_spoilage, fixed_result.avg_spoilage,
                     random_result.avg_spoilage, markdown_result.avg_spoilage,
                     adaptive_result.avg_spoilage]
        axes[1].bar(agents, spoilages, color=colors, alpha=0.8)
        axes[1].set_ylabel("Average Spoiled Units")
        axes[1].set_title("Spoilage Loss Comparison")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Metrics cards
        col1, col2, col3 = st.columns(3)
        col1.metric("🤖 RL Turnover", f"{rl_result.avg_inventory_turnover:.1%}")
        col2.metric("🤖 RL Spoilage", f"{rl_result.avg_spoilage:.0f} units")
        col3.metric("📊 Fixed Spoilage", f"{fixed_result.avg_spoilage:.0f} units")

    with tab3:
        st.subheader("Price Volatility & Behavior Analysis")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Price volatility comparison
        volatilities = [rl_result.avg_price_volatility, fixed_result.avg_price_volatility,
                        random_result.avg_price_volatility, markdown_result.avg_price_volatility,
                        adaptive_result.avg_price_volatility]
        axes[0].bar(agents, volatilities, color=colors, alpha=0.8)
        axes[0].set_ylabel("Price Volatility (Std Dev)")
        axes[0].set_title("Price Stability Analysis")
        
        # Average price comparison
        avg_prices = [
            np.mean([m.avg_price for m in rl_result.episode_metrics]),
            np.mean([m.avg_price for m in fixed_result.episode_metrics]),
            np.mean([m.avg_price for m in random_result.episode_metrics]),
            np.mean([m.avg_price for m in markdown_result.episode_metrics]),
            np.mean([m.avg_price for m in adaptive_result.episode_metrics])
        ]
        axes[1].bar(agents, avg_prices, color=colors, alpha=0.8)
        axes[1].axhline(y=production_cost, color='red', linestyle='--', label=f'Cost: ${production_cost}')
        axes[1].axhline(y=production_cost * (1 + min_margin_pct), color='orange', linestyle='--', 
                        label=f'Min Price: ${production_cost * (1 + min_margin_pct):.2f}')
        axes[1].set_ylabel("Average Price ($)")
        axes[1].set_title("Average Selling Price")
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Sample Episode Timeline (Last Test Episode)")
        
        # Get last episode data from RL agent
        if rl_result.episode_metrics:
            last_episode = rl_result.episode_metrics[-1]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Price over time
            axes[0, 0].plot(last_episode.prices, color='#2E8B57', linewidth=2)
            axes[0, 0].plot(last_episode.competitor_prices, color='#DC143C', linewidth=1, alpha=0.7, linestyle='--')
            axes[0, 0].axhline(y=production_cost * (1 + min_margin_pct), color='orange', linestyle=':', alpha=0.5)
            axes[0, 0].set_ylabel("Price ($)")
            axes[0, 0].set_xlabel("Time Step")
            axes[0, 0].set_title("Price Evolution (RL Agent vs Competitor)")
            axes[0, 0].legend(['RL Agent', 'Competitor', 'Min Price'])
            
            # Inventory over time
            axes[0, 1].fill_between(range(len(last_episode.inventories)), last_episode.inventories, 
                                     color='#4682B4', alpha=0.5)
            axes[0, 1].plot(last_episode.inventories, color='#4682B4', linewidth=2)
            axes[0, 1].set_ylabel("Inventory (Units)")
            axes[0, 1].set_xlabel("Time Step")
            axes[0, 1].set_title("Inventory Depletion Over Time")
            
            # Days to expiry
            axes[1, 0].fill_between(range(len(last_episode.days_to_expiry)), last_episode.days_to_expiry,
                                     color='#FF8C00', alpha=0.5)
            axes[1, 0].plot(last_episode.days_to_expiry, color='#FF8C00', linewidth=2)
            axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Critical Zone')
            axes[1, 0].set_ylabel("Days to Expiry")
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_title("Product Freshness Timeline")
            axes[1, 0].legend()
            
            # Cumulative reward
            cumulative_rewards = np.cumsum(last_episode.rewards)
            axes[1, 1].plot(cumulative_rewards, color='#2E8B57', linewidth=2)
            axes[1, 1].fill_between(range(len(cumulative_rewards)), cumulative_rewards, 
                                     color='#2E8B57', alpha=0.3)
            axes[1, 1].set_ylabel("Cumulative Reward ($)")
            axes[1, 1].set_xlabel("Time Step")
            axes[1, 1].set_title("Profit Accumulation")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Episode summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Episode Length", f"{last_episode.episode_length} steps")
            col2.metric("Total Reward", f"${last_episode.total_reward:,.0f}")
            col3.metric("Final Inventory", f"{last_episode.final_inventory:,.0f}")
            col4.metric("Termination", last_episode.termination_reason.title())

    # --- Business Insights ---
    st.header("💡 Business Insights & Recommendations")
    
    # Determine winner
    all_results = {
        "RL Agent (PPO)": rl_result.avg_reward,
        "Fixed Price": fixed_result.avg_reward,
        "Random": random_result.avg_reward,
        "Markdown Policy": markdown_result.avg_reward,
        "Adaptive Policy": adaptive_result.avg_reward
    }
    winner = max(all_results, key=all_results.get)
    winner_improvement = ((all_results[winner] / fixed_result.avg_reward) - 1) * 100
    
    st.success(f"""
    **🏆 Best Performing Agent: {winner}**
    
    With an average reward of **${all_results[winner]:,.0f}**, this represents a 
    **{winner_improvement:.1f}%** improvement over the fixed-price baseline.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Findings
        - **Spoilage Management**: The RL agent learns to markdown prices proactively 
          as expiry approaches, reducing spoilage loss.
        - **Competitive Response**: Frame stacking enables detection of competitor 
          pricing patterns over time.
        - **Safety Compliance**: All prices remain within guardrail bounds, ensuring 
          regulatory compliance.
        """)
    
    with col2:
        st.markdown(f"""
        ### System Configuration
        - **Guardrails**: {"✅ Enabled" if enable_guardrails else "❌ Disabled"}
        - **Min Price Floor**: ${production_cost * (1 + min_margin_pct):.2f}
        - **Max Price Ceiling**: ${production_cost * 4 * max_markup_factor:.2f}
        - **Frame Stack Size**: {frame_stack_size if enable_frame_stacking else "Disabled"}
        - **Days to Expiry**: {days_to_expiry} days
        """)

else:
    st.info("⬅️ Configure your simulation in the sidebar and click 'Train & Run A/B Testing' to start.")
    
    # Show feature overview
    st.header("🎯 System Features Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📦 Supply Chain Constraints & Spoilage Optimization
        - Days-to-expiry tracking in observation space
        - Progressive spoilage penalties as products age
        - Inventory-aware demand constraints
        - Urgency scoring for near-expiry items
        
        ### 🛡️ AI Guardrails & Risk Mitigation
        - Minimum margin enforcement (cost + X%)
        - Maximum markup caps (base × Y factor)
        - Action clipping before execution
        - Audit-ready price bounds logging
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 POMDP Handling via Frame Stacking
        - Configurable observation history (2-8 frames)
        - Temporal pattern detection capability
        - Competitor strategy inference
        - Market trend awareness
        
        ### 📊 Counterfactual Evaluation (A/B Testing)
        - Synchronized random seeds across agents
        - Statistical significance testing
        - 95% confidence intervals
        - Multiple baseline comparisons
        """)