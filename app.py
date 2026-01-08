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

st.set_page_config(layout="wide", page_title="AI Pricing Optimizer - Business Dashboard")

# --- Custom CSS for Business-Friendly Styling ---
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
    }
    .savings-positive {
        color: #2E8B57;
        font-weight: bold;
    }
    .savings-negative {
        color: #DC143C;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 4px solid #4682B4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .confidence-high {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 4px;
    }
    .confidence-low {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

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

# --- Helper Functions for Business Metrics ---
def format_as_cost(value):
    """Convert reward to positive cost display."""
    return abs(value)

def format_savings(value, baseline):
    """Calculate and format savings vs baseline."""
    savings = value - baseline
    return savings

def get_confidence_label(p_value):
    """Convert p-value to business-friendly confidence level."""
    if p_value < 0.01:
        return "Very High Confidence (99%+)", "confidence-high"
    elif p_value < 0.05:
        return "High Confidence (95%+)", "confidence-high"
    elif p_value < 0.10:
        return "Moderate Confidence (90%+)", "confidence-low"
    else:
        return "Low Confidence (<90%)", "confidence-low"

def get_effect_interpretation(cohens_d):
    """Convert Cohen's d to business-friendly impact description."""
    d = abs(cohens_d)
    if d >= 0.8:
        return "Large Impact"
    elif d >= 0.5:
        return "Medium Impact"
    elif d >= 0.2:
        return "Small Impact"
    else:
        return "Negligible Impact"

# --- UI Layout ---
st.title("💰 AI Pricing Optimizer")
st.markdown("""
Make smarter pricing decisions with AI-powered strategy simulation. 
Compare different approaches and see projected savings instantly.
""")

# --- Sidebar with Collapsible Sections ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if "run_sim" not in st.session_state:
        st.session_state.run_sim = False
    
    # --- Quick Preset Scenarios ---
    st.subheader("🎯 Quick Start Scenarios")
    scenario = st.selectbox(
        "Choose a preset scenario",
        ["Custom Settings", "Conservative Strategy", "Aggressive Growth", "High Volatility Market", "Perishable Goods Focus"],
        help="Select a pre-configured scenario or customize your own settings below"
    )
    
    # Apply presets
    if scenario == "Conservative Strategy":
        preset = {"production_cost": 5.0, "initial_inventory": 10000, "days_to_expiry": 45, 
                  "spoilage_penalty_rate": 1.2, "min_margin_pct": 0.15, "max_markup_factor": 1.5,
                  "base_demand": 80, "price_change_penalty": 200, "cross_elasticity": -0.3,
                  "competitor_reaction_speed": 0.3, "competitor_noise": 0.02, "total_timesteps": 30000}
    elif scenario == "Aggressive Growth":
        preset = {"production_cost": 5.0, "initial_inventory": 15000, "days_to_expiry": 30,
                  "spoilage_penalty_rate": 1.5, "min_margin_pct": 0.08, "max_markup_factor": 2.5,
                  "base_demand": 150, "price_change_penalty": 50, "cross_elasticity": -0.7,
                  "competitor_reaction_speed": 0.7, "competitor_noise": 0.08, "total_timesteps": 50000}
    elif scenario == "High Volatility Market":
        preset = {"production_cost": 5.0, "initial_inventory": 8000, "days_to_expiry": 25,
                  "spoilage_penalty_rate": 1.8, "min_margin_pct": 0.10, "max_markup_factor": 2.0,
                  "base_demand": 120, "price_change_penalty": 80, "cross_elasticity": -0.8,
                  "competitor_reaction_speed": 0.8, "competitor_noise": 0.15, "total_timesteps": 50000}
    elif scenario == "Perishable Goods Focus":
        preset = {"production_cost": 8.0, "initial_inventory": 5000, "days_to_expiry": 14,
                  "spoilage_penalty_rate": 2.5, "min_margin_pct": 0.12, "max_markup_factor": 1.8,
                  "base_demand": 100, "price_change_penalty": 100, "cross_elasticity": -0.4,
                  "competitor_reaction_speed": 0.4, "competitor_noise": 0.05, "total_timesteps": 40000}
    else:
        preset = None
    
    st.divider()
    
    # --- Collapsible: Supply Chain Settings ---
    with st.expander("📦 Supply Chain Settings", expanded=False):
        st.caption("Configure inventory and product lifecycle parameters")
        
        production_cost = st.number_input(
            "Unit Cost ($)", 1.0, 20.0, 
            preset["production_cost"] if preset else 5.0, 0.5,
            help="Your cost to produce or acquire each unit"
        )
        initial_inventory = st.number_input(
            "Starting Inventory (units)", 1000, 50000, 
            preset["initial_inventory"] if preset else 10000, 1000,
            help="Total units available at start of simulation"
        )
        days_to_expiry = st.slider(
            "Product Shelf Life (days)", 7, 60, 
            preset["days_to_expiry"] if preset else 30, 1,
            help="Days until product expires and becomes unsellable"
        )
        spoilage_penalty_rate = st.slider(
            "Waste Cost Multiplier", 1.0, 3.0, 
            preset["spoilage_penalty_rate"] if preset else 1.5, 0.1,
            help="How much expired inventory costs relative to unit cost (1.5x = 50% extra loss)"
        )
    
    # --- Collapsible: Pricing Rules ---
    with st.expander("🛡️ Pricing Rules & Guardrails", expanded=False):
        st.caption("Set boundaries for AI pricing decisions")
        
        enable_guardrails = st.checkbox(
            "Enable Price Guardrails", value=True,
            help="Prevent prices from going too low (unprofitable) or too high (unfair)"
        )
        min_margin_pct = st.slider(
            "Minimum Profit Margin (%)", 5, 30, 
            int((preset["min_margin_pct"] if preset else 0.10) * 100), 1,
            help="Lowest acceptable profit margin on each sale"
        ) / 100
        max_markup_factor = st.slider(
            "Maximum Price Cap (x cost)", 1.5, 3.0, 
            preset["max_markup_factor"] if preset else 2.0, 0.1,
            help="Highest price as a multiple of base cost"
        )
    
    # --- Collapsible: Market Conditions ---
    with st.expander("🏪 Market Conditions", expanded=False):
        st.caption("Simulate different market environments")
        
        base_demand = st.slider(
            "Average Daily Demand (units)", 50, 500, 
            preset["base_demand"] if preset else 100, 10,
            help="Expected customer demand per day"
        )
        price_change_penalty = st.slider(
            "Price Stability Preference", 0, 1000, 
            preset["price_change_penalty"] if preset else 100, 10,
            help="Higher = prefer stable prices; Lower = allow frequent changes"
        )
    
    # --- Collapsible: Competitor Behavior ---
    with st.expander("🏁 Competitor Behavior", expanded=False):
        st.caption("Model how competitors react to your pricing")
        
        cross_elasticity = st.slider(
            "Customer Price Sensitivity", -1.0, 0.0, 
            preset["cross_elasticity"] if preset else -0.5, 0.05,
            help="How much customers switch based on competitor prices (-1 = very sensitive)"
        )
        competitor_reaction_speed = st.slider(
            "Competitor Aggressiveness", 0.1, 1.0, 
            preset["competitor_reaction_speed"] if preset else 0.5, 0.1,
            help="How quickly competitors match your price changes"
        )
        competitor_noise = st.slider(
            "Market Unpredictability", 0.0, 0.2, 
            preset["competitor_noise"] if preset else 0.05, 0.01,
            help="Random variation in competitor behavior"
        )
    
    # --- Collapsible: Advanced AI Settings ---
    with st.expander("🤖 Advanced AI Settings", expanded=False):
        st.caption("Fine-tune the AI optimization engine")
        
        enable_frame_stacking = st.checkbox(
            "Enable Trend Detection", value=True,
            help="Allow AI to detect patterns over multiple days (recommended)"
        )
        frame_stack_size = st.slider(
            "Trend Window (days)", 2, 8, 4, 1,
            help="How many days of history the AI considers"
        ) if enable_frame_stacking else 1
        
        total_timesteps = st.slider(
            "AI Training Intensity", 10000, 100000, 
            preset["total_timesteps"] if preset else 50000, 5000,
            help="Higher = smarter AI but longer training time"
        )
    
    # --- Collapsible: Simulation Settings ---
    with st.expander("📊 Simulation Settings", expanded=False):
        st.caption("Configure the strategy comparison test")
        
        n_test_episodes = st.slider(
            "Test Scenarios to Run", 10, 100, 30, 5,
            help="More scenarios = more reliable results but longer runtime"
        )
        ab_test_seed = st.number_input(
            "Reproducibility Seed", 0, 9999, 42, 1,
            help="Use same seed to reproduce exact results"
        )
    
    st.divider()
    
    if st.button("🚀 Run Strategy Simulation", type="primary", use_container_width=True):
        st.session_state.run_sim = True

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
    
    # Progress indicators
    progress_bar = st.progress(0, text="Loading market data...")
    
    with st.spinner("Loading dataset..."):
        pricing_dataset = load_data()
    progress_bar.progress(20, text="Training AI pricing model...")

    env_lambda = lambda: RealisticPricingEnv(pricing_dataset, **env_params)
    vec_env = DummyVecEnv([env_lambda])

    with st.spinner(f"Training AI model..."):
        model = get_trained_model(tuple(env_params.items()), total_timesteps)
    progress_bar.progress(60, text="Running strategy simulations...")
    
    with st.spinner("Running strategy comparison..."):
        ab_framework = ABTestingFramework(pricing_dataset, env_params)
        
        rl_result = ab_framework.run_ab_test(
            "RL Agent (PPO)", create_rl_policy(model), 
            n_episodes=n_test_episodes, base_seed=int(ab_test_seed)
        )
        fixed_result = ab_framework.run_ab_test(
            "Fixed Price", create_fixed_policy(), 
            n_episodes=n_test_episodes, base_seed=int(ab_test_seed)
        )
        random_result = ab_framework.run_ab_test(
            "Random", create_random_policy(), 
            n_episodes=n_test_episodes, base_seed=int(ab_test_seed)
        )
        markdown_result = ab_framework.run_ab_test(
            "Markdown Policy", create_markdown_policy(), 
            n_episodes=n_test_episodes, base_seed=int(ab_test_seed)
        )
        adaptive_result = ab_framework.run_ab_test(
            "Adaptive Policy", create_adaptive_policy(production_cost), 
            n_episodes=n_test_episodes, base_seed=int(ab_test_seed)
        )
    
    progress_bar.progress(100, text="Analysis complete!")
    progress_bar.empty()

    # ===========================================
    # EXECUTIVE SUMMARY - Top of Dashboard
    # ===========================================
    st.header("📊 Executive Summary")
    
    # Calculate key business metrics
    baseline_profit = fixed_result.avg_reward
    all_results = {
        "AI Optimizer": rl_result.avg_reward,
        "Fixed Pricing": fixed_result.avg_reward,
        "Random Pricing": random_result.avg_reward,
        "Markdown Strategy": markdown_result.avg_reward,
        "Smart Rules": adaptive_result.avg_reward
    }
    
    best_strategy = max(all_results, key=all_results.get)
    best_profit = all_results[best_strategy]
    savings_vs_baseline = best_profit - baseline_profit
    pct_improvement = ((best_profit / baseline_profit) - 1) * 100 if baseline_profit != 0 else 0
    
    # Hero metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏆 Best Strategy",
            best_strategy,
            f"+{pct_improvement:.1f}% vs Fixed" if pct_improvement > 0 else f"{pct_improvement:.1f}% vs Fixed"
        )
    
    with col2:
        # Display profit as positive value
        display_profit = best_profit
        st.metric(
            "💰 Projected Profit",
            f"${display_profit:,.0f}",
            f"${savings_vs_baseline:+,.0f} vs baseline" if savings_vs_baseline != 0 else "Baseline"
        )
    
    with col3:
        spoilage_saved = fixed_result.avg_spoilage - rl_result.avg_spoilage
        st.metric(
            "📦 Waste Reduction",
            f"{rl_result.avg_spoilage:.0f} units",
            f"{spoilage_saved:+,.0f} units saved" if spoilage_saved > 0 else f"{spoilage_saved:,.0f} units"
        )
    
    with col4:
        st.metric(
            "📈 Inventory Sold",
            f"{rl_result.avg_inventory_turnover:.0%}",
            f"of {initial_inventory:,} units"
        )
    
    # Confidence indicator
    sig_test = ab_framework.get_statistical_significance("RL Agent (PPO)", "Fixed Price")
    if "error" not in sig_test:
        confidence_label, confidence_class = get_confidence_label(sig_test['p_value'])
        impact_label = get_effect_interpretation(sig_test['cohens_d'])
        
        st.markdown(f"""
        <div class="insight-box">
            <strong>📌 Result Confidence:</strong> {confidence_label} | 
            <strong>Business Impact:</strong> {impact_label}
            <br><small>Based on {n_test_episodes} simulated scenarios with identical market conditions</small>
        </div>
        """, unsafe_allow_html=True)

    # ===========================================
    # STRATEGY COMPARISON - Normalized to Baseline
    # ===========================================
    st.header("📈 Strategy Performance Comparison")
    st.markdown("*All values shown relative to Fixed Pricing baseline (set to $0)*")
    
    # Calculate savings relative to baseline
    strategies = ["AI Optimizer", "Fixed Pricing", "Random", "Markdown", "Smart Rules"]
    results_list = [rl_result, fixed_result, random_result, markdown_result, adaptive_result]
    savings_vs_fixed = [r.avg_reward - baseline_profit for r in results_list]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Normalized bar chart - baseline = 0
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = ['#2E8B57' if s >= 0 else '#DC143C' for s in savings_vs_fixed]
        bars = ax.barh(strategies, savings_vs_fixed, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, saving in zip(bars, savings_vs_fixed):
            width = bar.get_width()
            label = f"+${saving:,.0f}" if saving >= 0 else f"-${abs(saving):,.0f}"
            x_pos = width + 500 if width >= 0 else width - 500
            ha = 'left' if width >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, label, 
                   va='center', ha=ha, fontweight='bold', fontsize=11)
        
        ax.axvline(x=0, color='#4682B4', linestyle='-', linewidth=2, label='Fixed Pricing (Baseline)')
        ax.set_xlabel("Savings vs Fixed Pricing ($)", fontsize=12)
        ax.set_title("Strategy Value Comparison", fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set reasonable x-axis limits
        max_abs = max(abs(min(savings_vs_fixed)), abs(max(savings_vs_fixed))) * 1.3
        ax.set_xlim(-max_abs, max_abs)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 💡 Quick Insights")
        
        # Find best and worst
        best_idx = savings_vs_fixed.index(max(savings_vs_fixed))
        worst_idx = savings_vs_fixed.index(min(savings_vs_fixed))
        
        if savings_vs_fixed[best_idx] > 0:
            st.success(f"**{strategies[best_idx]}** generates **${savings_vs_fixed[best_idx]:,.0f}** more profit than fixed pricing")
        
        if savings_vs_fixed[worst_idx] < 0:
            st.error(f"**{strategies[worst_idx]}** loses **${abs(savings_vs_fixed[worst_idx]):,.0f}** compared to fixed pricing")
        
        # AI vs Smart Rules comparison
        ai_vs_rules = rl_result.avg_reward - adaptive_result.avg_reward
        if ai_vs_rules > 0:
            st.info(f"AI outperforms rule-based approach by **${ai_vs_rules:,.0f}**")
        else:
            st.info(f"Rule-based approach outperforms AI by **${abs(ai_vs_rules):,.0f}**")

    # ===========================================
    # DETAILED PERFORMANCE CARDS
    # ===========================================
    st.header("🎯 Detailed Strategy Analysis")
    
    tab1, tab2, tab3 = st.tabs([
        "💰 Financial Performance", 
        "📦 Operations & Waste", 
        "📊 Price Stability"
    ])
    
    with tab1:
        st.subheader("Profit Distribution by Strategy")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Performance cards as horizontal comparison
        strategy_names = ["AI\nOptimizer", "Fixed\nPricing", "Random", "Markdown", "Smart\nRules"]
        profits = [r.avg_reward for r in results_list]
        stds = [r.std_reward for r in results_list]
        
        colors = ['#2E8B57', '#4682B4', '#DC143C', '#FF8C00', '#9370DB']
        
        # Normalize to show relative to baseline
        normalized_profits = [p - baseline_profit for p in profits]
        
        bars = axes[0].bar(strategy_names, normalized_profits, color=colors, alpha=0.8, edgecolor='white')
        axes[0].axhline(y=0, color='#4682B4', linestyle='--', linewidth=2, alpha=0.7)
        axes[0].set_ylabel("Profit vs Baseline ($)", fontsize=11)
        axes[0].set_title("Relative Profitability", fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, normalized_profits):
            height = bar.get_height()
            label = f"+${val:,.0f}" if val >= 0 else f"${val:,.0f}"
            y_pos = height + 200 if height >= 0 else height - 200
            va = 'bottom' if height >= 0 else 'top'
            axes[0].text(bar.get_x() + bar.get_width()/2, y_pos, label,
                        ha='center', va=va, fontsize=10, fontweight='bold')
        
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Right: Distribution with outlier annotations
        reward_data = [
            [m.total_reward for m in rl_result.episode_metrics],
            [m.total_reward for m in fixed_result.episode_metrics],
            [m.total_reward for m in random_result.episode_metrics],
            [m.total_reward for m in markdown_result.episode_metrics],
            [m.total_reward for m in adaptive_result.episode_metrics]
        ]
        
        bp = axes[1].boxplot(reward_data, labels=strategy_names, patch_artist=True, 
                             medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel("Profit per Scenario ($)", fontsize=11)
        axes[1].set_title("Profit Range & Consistency", fontsize=12, fontweight='bold')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        # Add annotation for outliers
        axes[1].annotate('Outliers may indicate:\n• Stockout events\n• Price wars\n• Demand spikes', 
                        xy=(0.98, 0.02), xycoords='axes fraction',
                        fontsize=8, ha='right', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Insight cards below
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="insight-box">
                <strong>📊 Reading This Chart</strong><br>
                <small>
                • <strong>Bar height</strong> = average profit difference vs baseline<br>
                • <strong>Box width</strong> = consistency (narrower = more predictable)<br>
                • <strong>Dots outside boxes</strong> = unusual scenarios (stockouts, price wars)
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculate consistency metric
            most_consistent = strategies[stds.index(min(stds))]
            least_consistent = strategies[stds.index(max(stds))]
            st.markdown(f"""
            <div class="insight-box">
                <strong>🎯 Consistency Analysis</strong><br>
                <small>
                • Most predictable: <strong>{most_consistent}</strong><br>
                • Most variable: <strong>{least_consistent}</strong><br>
                • Lower variability = more reliable forecasting
                </small>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Inventory Management & Waste Reduction")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inventory turnover
        turnovers = [r.avg_inventory_turnover * 100 for r in results_list]
        bars = axes[0].bar(strategy_names, turnovers, color=colors, alpha=0.8)
        axes[0].set_ylabel("Inventory Sold (%)", fontsize=11)
        axes[0].set_title("Sales Efficiency", fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target: 80%')
        axes[0].legend()
        
        # Add value labels
        for bar, val in zip(bars, turnovers):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Spoilage/Waste
        spoilages = [r.avg_spoilage for r in results_list]
        waste_costs = [s * production_cost * spoilage_penalty_rate for s in spoilages]
        
        bars = axes[1].bar(strategy_names, waste_costs, color=['#90EE90' if w == min(waste_costs) else '#FFB6C1' for w in waste_costs], 
                          alpha=0.8, edgecolor='darkgray')
        axes[1].set_ylabel("Waste Cost ($)", fontsize=11)
        axes[1].set_title("Product Waste & Spoilage Costs", fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, waste_costs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        best_turnover_idx = turnovers.index(max(turnovers))
        lowest_waste_idx = waste_costs.index(min(waste_costs))
        
        col1.metric("🏆 Best Sales Efficiency", strategies[best_turnover_idx], f"{turnovers[best_turnover_idx]:.0f}% sold")
        col2.metric("♻️ Lowest Waste", strategies[lowest_waste_idx], f"${waste_costs[lowest_waste_idx]:,.0f}")
        col3.metric("💸 AI Waste Savings", f"${waste_costs[1] - waste_costs[0]:,.0f}", "vs Fixed Pricing")

    with tab3:
        st.subheader("Pricing Stability & Behavior")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Price volatility (lower is more stable)
        volatilities = [r.avg_price_volatility for r in results_list]
        stability_scores = [100 / (1 + v) for v in volatilities]  # Convert to stability (higher = better)
        
        bars = axes[0].bar(strategy_names, stability_scores, color=colors, alpha=0.8)
        axes[0].set_ylabel("Price Stability Score", fontsize=11)
        axes[0].set_title("Pricing Consistency", fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, max(stability_scores) * 1.2)
        
        for bar, val in zip(bars, stability_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Average price comparison
        avg_prices = [np.mean([m.avg_price for m in r.episode_metrics]) for r in results_list]
        
        bars = axes[1].bar(strategy_names, avg_prices, color=colors, alpha=0.8)
        axes[1].axhline(y=production_cost, color='red', linestyle='--', linewidth=2, label=f'Cost: ${production_cost}')
        axes[1].axhline(y=production_cost * (1 + min_margin_pct), color='orange', linestyle='--', 
                       linewidth=2, label=f'Min Price: ${production_cost * (1 + min_margin_pct):.2f}')
        axes[1].set_ylabel("Average Selling Price ($)", fontsize=11)
        axes[1].set_title("Average Price by Strategy", fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        
        for bar, val in zip(bars, avg_prices):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'${val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **💡 Price Stability Matters:** Frequent price changes can confuse customers and damage brand perception. 
        Higher stability scores indicate more consistent pricing that's easier for customers to understand.
        """)

    # ===========================================
    # TIMELINE VIEW - Sample Episode
    # ===========================================
    st.header("🔄 Sample Scenario Deep Dive")
    st.markdown("*Detailed view of one complete simulation showing AI pricing decisions over time*")
    
    if rl_result.episode_metrics:
        last_episode = rl_result.episode_metrics[-1]
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📅 Duration", f"{last_episode.episode_length} days")
        col2.metric("💰 Total Profit", f"${last_episode.total_reward:,.0f}")
        col3.metric("📦 Final Stock", f"{last_episode.final_inventory:,.0f} units")
        
        termination_emoji = {"stockout": "📦", "expiry": "⏰", "max_steps": "✅"}
        col4.metric("🏁 Ended By", 
                   last_episode.termination_reason.replace("_", " ").title(),
                   termination_emoji.get(last_episode.termination_reason, ""))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price over time
        axes[0, 0].plot(last_episode.prices, color='#2E8B57', linewidth=2, label='AI Price')
        axes[0, 0].plot(last_episode.competitor_prices, color='#DC143C', linewidth=1.5, 
                       alpha=0.7, linestyle='--', label='Competitor')
        axes[0, 0].axhline(y=production_cost * (1 + min_margin_pct), color='orange', 
                          linestyle=':', alpha=0.7, label='Min Price')
        axes[0, 0].fill_between(range(len(last_episode.prices)), 
                                [production_cost] * len(last_episode.prices),
                                alpha=0.1, color='red', label='Cost Zone')
        axes[0, 0].set_ylabel("Price ($)")
        axes[0, 0].set_xlabel("Day")
        axes[0, 0].set_title("Price Evolution Over Time", fontweight='bold')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # Inventory over time
        axes[0, 1].fill_between(range(len(last_episode.inventories)), last_episode.inventories, 
                                color='#4682B4', alpha=0.3)
        axes[0, 1].plot(last_episode.inventories, color='#4682B4', linewidth=2)
        axes[0, 1].axhline(y=initial_inventory * 0.2, color='orange', linestyle='--', 
                          alpha=0.7, label='Low Stock Warning')
        axes[0, 1].set_ylabel("Inventory (Units)")
        axes[0, 1].set_xlabel("Day")
        axes[0, 1].set_title("Inventory Levels", fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # Days to expiry with danger zone
        axes[1, 0].fill_between(range(len(last_episode.days_to_expiry)), last_episode.days_to_expiry,
                                color='#FF8C00', alpha=0.3)
        axes[1, 0].plot(last_episode.days_to_expiry, color='#FF8C00', linewidth=2)
        axes[1, 0].axhspan(0, 5, alpha=0.2, color='red', label='Danger Zone (< 5 days)')
        axes[1, 0].set_ylabel("Days Until Expiry")
        axes[1, 0].set_xlabel("Day")
        axes[1, 0].set_title("Product Freshness", fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        
        # Cumulative profit
        cumulative_profit = np.cumsum(last_episode.rewards)
        axes[1, 1].plot(cumulative_profit, color='#2E8B57', linewidth=2)
        axes[1, 1].fill_between(range(len(cumulative_profit)), cumulative_profit, 
                                where=[p >= 0 for p in cumulative_profit],
                                color='#2E8B57', alpha=0.3, label='Profit')
        axes[1, 1].fill_between(range(len(cumulative_profit)), cumulative_profit,
                                where=[p < 0 for p in cumulative_profit],
                                color='#DC143C', alpha=0.3, label='Loss')
        axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1, 1].set_ylabel("Cumulative Profit ($)")
        axes[1, 1].set_xlabel("Day")
        axes[1, 1].set_title("Profit Accumulation", fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

    # ===========================================
    # BUSINESS RECOMMENDATIONS
    # ===========================================
    st.header("💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Key Findings
        
        {"✅" if rl_result.avg_reward > fixed_result.avg_reward else "⚠️"} **AI vs Fixed Pricing:** 
        {"AI generates " + f"${abs(rl_result.avg_reward - fixed_result.avg_reward):,.0f} more profit" if rl_result.avg_reward > fixed_result.avg_reward else "Fixed pricing performs better in this scenario"}
        
        {"✅" if rl_result.avg_spoilage < fixed_result.avg_spoilage else "⚠️"} **Waste Management:** 
        {"AI reduces waste by " + f"{fixed_result.avg_spoilage - rl_result.avg_spoilage:,.0f} units" if rl_result.avg_spoilage < fixed_result.avg_spoilage else "Consider adjusting expiry markdown rules"}
        
        {"✅" if rl_result.avg_inventory_turnover > 0.7 else "⚠️"} **Sales Efficiency:** 
        {f"{rl_result.avg_inventory_turnover:.0%} of inventory sold" + (" (Good)" if rl_result.avg_inventory_turnover > 0.7 else " (Consider price adjustments)")}
        """)
    
    with col2:
        st.markdown(f"""
        ### Configuration Summary
        
        | Setting | Value |
        |---------|-------|
        | Unit Cost | ${production_cost:.2f} |
        | Min Selling Price | ${production_cost * (1 + min_margin_pct):.2f} |
        | Max Selling Price | ${production_cost * 4 * max_markup_factor:.2f} |
        | Shelf Life | {days_to_expiry} days |
        | Price Rules | {"Enabled ✅" if enable_guardrails else "Disabled ⚠️"} |
        | AI Trend Detection | {"Enabled ✅" if enable_frame_stacking else "Disabled"} |
        """)

else:
    # ===========================================
    # LANDING PAGE - Before Simulation
    # ===========================================
    st.markdown("""
    ### 👋 Welcome! Let's optimize your pricing strategy.
    
    This tool uses AI to simulate different pricing approaches and find the most profitable strategy 
    for your business. Configure your settings in the sidebar and click **Run Strategy Simulation** to begin.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📦 Supply Chain Aware
        - Tracks product shelf life
        - Accounts for waste costs
        - Optimizes inventory turnover
        """)
    
    with col2:
        st.markdown("""
        #### 🛡️ Safe & Controlled
        - Enforces minimum profit margins
        - Caps maximum prices
        - Prevents unprofitable decisions
        """)
    
    with col3:
        st.markdown("""
        #### 📊 Data-Driven
        - Compares 5 strategies
        - Shows confidence levels
        - Visualizes trade-offs
        """)
    
    st.divider()
    
    st.markdown("""
    ### 🎯 How It Works
    
    1. **Configure** your business parameters (costs, inventory, shelf life)
    2. **Simulate** multiple pricing strategies under identical market conditions  
    3. **Compare** results to find the best approach for your situation
    4. **Decide** with confidence using clear, business-focused insights
    
    ---
    
    *💡 Tip: Use the "Quick Start Scenarios" dropdown in the sidebar to try pre-configured settings*
    """)
