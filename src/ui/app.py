import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.pricing_env import DynamicPricingEnv

st.set_page_config(page_title="Dynamic Pricing System", layout="wide")

st.title("🛒 AI-Powered Dynamic Pricing System")

# Load Data
@st.cache_data
def load_data():
    meta_df = pd.read_csv(os.path.join(project_root, "src/data/product_metadata.csv"))
    demand_df = pd.read_csv(os.path.join(project_root, "src/data/demand_patterns.csv"))
    return meta_df, demand_df

meta_df, demand_df = load_data()

# Sidebar
st.sidebar.header("Configuration")
product_name = st.sidebar.selectbox("Select Product", meta_df['product_name'])
product_info = meta_df[meta_df['product_name'] == product_name].iloc[0]
product_id = product_info['product_id']

# Load Model
model_path = os.path.join(project_root, f"src/models/saved_models/ppo_pricing_{product_id}.zip")
# Fallback to the one we trained if specific one doesn't exist (since we only trained one)
if not os.path.exists(model_path):
    # Find the one we trained
    saved_models_dir = os.path.join(project_root, "src/models/saved_models")
    available_models = [f for f in os.listdir(saved_models_dir) if f.endswith(".zip")]
    if available_models:
        model_path = os.path.join(saved_models_dir, available_models[0])
        st.sidebar.warning(f"Model for {product_name} not found. Using {available_models[0]} as demo.")
    else:
        st.error("No trained models found. Please run training script.")
        st.stop()

model = PPO.load(model_path)

# Simulation State
if 'env' not in st.session_state or st.session_state.product_id != product_id:
    st.session_state.env = DynamicPricingEnv(product_info, demand_df)
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.product_id = product_id
    st.session_state.history = []

# Simulation Controls
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Step Simulation (Agent)"):
        action, _ = model.predict(st.session_state.obs)
        obs, reward, done, truncated, info = st.session_state.env.step(action)
        st.session_state.obs = obs
        
        # Record history
        st.session_state.history.append({
            "step": st.session_state.env.current_step,
            "price": obs[0],
            "competitor_price": obs[1],
            "demand": info['demand'],
            "revenue": info['revenue'],
            "source": "Agent"
        })

with col2:
    manual_price = st.number_input("Manual Price Override", value=float(st.session_state.obs[0]), min_value=0.0)
    if st.button("Step (Manual)"):
        # Calculate action from manual price to keep env consistent
        # Action is multiplier: price / base_price
        action = np.array([manual_price / product_info['base_price']])
        obs, reward, done, truncated, info = st.session_state.env.step(action)
        st.session_state.obs = obs
        
        st.session_state.history.append({
            "step": st.session_state.env.current_step,
            "price": obs[0],
            "competitor_price": obs[1],
            "demand": info['demand'],
            "revenue": info['revenue'],
            "source": "Manual"
        })

with col3:
    if st.button("Run Full Year (Agent)"):
        # Reset and run 365 steps
        env = DynamicPricingEnv(product_info, demand_df)
        obs, _ = env.reset()
        history = []
        for _ in range(365):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            history.append({
                "step": env.current_step,
                "price": obs[0],
                "competitor_price": obs[1],
                "demand": info['demand'],
                "revenue": info['revenue'],
                "source": "Agent"
            })
        st.session_state.history = history
        st.session_state.env = env
        st.session_state.obs = obs

if st.button("Reset Simulation"):
    st.session_state.env = DynamicPricingEnv(product_info, demand_df)
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.history = []

# Dashboard
if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    
    # Metrics
    last_step = df_hist.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${last_step['price']:.2f}")
    m2.metric("Competitor Price", f"${last_step['competitor_price']:.2f}", delta=f"{last_step['competitor_price'] - last_step['price']:.2f}")
    m3.metric("Daily Demand", f"{int(last_step['demand'])}")
    m4.metric("Daily Revenue", f"${last_step['revenue']:.2f}")
    
    # Charts
    st.subheader("Price vs Competitor")
    st.line_chart(df_hist[['price', 'competitor_price']])
    
    st.subheader("Demand & Revenue")
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Demand', color='tab:blue')
    ax1.plot(df_hist['step'], df_hist['demand'], color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Revenue', color='tab:green')
    ax2.plot(df_hist['step'], df_hist['revenue'], color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    st.pyplot(fig)

else:
    st.info("Start the simulation to see results.")

# Current State Info
st.subheader("Current Market State")
st.write(f"**Base Price:** ${product_info['base_price']}")
st.write(f"**Elasticity:** {product_info['elasticity']}")
st.write(f"**Day of Week:** {int(st.session_state.obs[2] * 7)}")
