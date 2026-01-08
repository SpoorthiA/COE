import numpy as np
from stable_baselines3 import PPO

def train_model(env, total_timesteps=50000, learning_rate=3e-4, n_steps=1024, 
                batch_size=64, n_epochs=10, gamma=0.99, verbose=0):
    """
    Train PPO model with optimized hyperparameters for pricing environment.
    
    Key tuning choices:
    - Smaller n_steps (1024) for faster updates with short episodes
    - ent_coef for exploration in discrete action space
    - clip_range for stable learning
    - Larger network for complex observations
    """
    # Calculate appropriate n_steps based on expected episode length
    # For short episodes, we want more frequent updates
    
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])  # Larger networks
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        ent_coef=0.01,  # Encourage exploration
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    return model

def evaluate_pricing_policy(model, env, n_episodes=50):
    """
    Evaluate trained model with comprehensive metrics including
    supply chain KPIs.
    """
    rewards = []
    price_volatilities = []
    demands_fulfilled = []
    inventory_turnovers = []
    spoilage_rates = []
    
    sample_prices, sample_demands = [], []
    sample_inventories, sample_expiry = [], []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_prices = []
        episode_demands = []
        episode_inventories = []
        episode_expiry = []
        episode_spoilage = 0
        initial_inventory = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = env.step(action)
            
            done = done_arr[0]
            info = info_arr[0]
            episode_reward += reward[0]
            
            price = info['price']
            demand = info['demand']
            inventory = info['inventory']
            expiry = info.get('days_to_expiry', 0)
            spoiled = info.get('spoiled_units', 0)
            
            if initial_inventory is None:
                initial_inventory = inventory + demand + spoiled
            
            episode_prices.append(price)
            episode_demands.append(demand)
            episode_inventories.append(inventory)
            episode_expiry.append(expiry)
            episode_spoilage += spoiled

        rewards.append(episode_reward)
        price_volatilities.append(np.std(episode_prices))
        demands_fulfilled.append(np.sum(episode_demands))
        
        # Supply chain metrics
        if initial_inventory and initial_inventory > 0:
            turnover = np.sum(episode_demands) / initial_inventory
            spoilage_rate = episode_spoilage / initial_inventory
        else:
            turnover = 0
            spoilage_rate = 0
            
        inventory_turnovers.append(turnover)
        spoilage_rates.append(spoilage_rate)
        
        if episode == n_episodes - 1:
            sample_prices = episode_prices
            sample_demands = episode_demands
            sample_inventories = episode_inventories
            sample_expiry = episode_expiry

    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_price_volatility': np.mean(price_volatilities),
        'avg_demand_fulfilled': np.mean(demands_fulfilled),
        'avg_inventory_turnover': np.mean(inventory_turnovers),
        'avg_spoilage_rate': np.mean(spoilage_rates),
        'sample_prices': sample_prices,
        'sample_demands': sample_demands,
        'sample_inventories': sample_inventories,
        'sample_expiry': sample_expiry
    }