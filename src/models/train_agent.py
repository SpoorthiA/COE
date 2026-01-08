import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.models.pricing_env import DynamicPricingEnv

def train_agent():
    print("Loading data for training...")
    meta_df = pd.read_csv("src/data/product_metadata.csv")
    demand_df = pd.read_csv("src/data/demand_patterns.csv")
    
    # Train on the first product
    product_info = meta_df.iloc[0]
    print(f"Training on product: {product_info['product_name']}")
    
    # Create Environment
    env = DynamicPricingEnv(product_info, demand_df)
    env = DummyVecEnv([lambda: env])
    
    # Initialize Agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    # Train
    print("Starting training...")
    model.learn(total_timesteps=10000)
    
    # Save
    os.makedirs("src/models/saved_models", exist_ok=True)
    model_path = f"src/models/saved_models/ppo_pricing_{product_info['product_id']}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_agent()
