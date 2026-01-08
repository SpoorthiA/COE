import numpy as np

def run_baseline_policy(env, policy_type="fixed", **kwargs):
    """
    Run baseline policies for comparison against RL agent.
    
    Policy types:
    - "fixed": Maintains constant price (no adjustment)
    - "random": Random price adjustments
    - "markdown": Rule-based markdown as expiry approaches
    - "adaptive": Sophisticated rule-based considering inventory, competition, expiry
    """
    obs = env.reset()
    done = False
    episode_reward = 0
    prices = []
    demands = []
    inventories = []
    expiries = []
    spoilage_total = 0
    initial_inventory = None

    while not done:
        if policy_type == "fixed":
            action = np.array([3])  # No change
        elif policy_type == "random":
            action = np.array([env.action_space.sample()])
        elif policy_type == "markdown":
            action = _markdown_action(obs, **kwargs)
        elif policy_type == "adaptive":
            action = _adaptive_action(obs, **kwargs)
        else:
            action = np.array([3])

        obs, reward, done_arr, info_arr = env.step(action)
        done = done_arr[0]
        info = info_arr[0]
        
        if initial_inventory is None:
            initial_inventory = info["inventory"] + info["demand"] + info.get("spoiled_units", 0)
        
        episode_reward += reward[0]
        prices.append(info["price"])
        demands.append(info["demand"])
        inventories.append(info["inventory"])
        expiries.append(info.get("days_to_expiry", 0))
        spoilage_total += info.get("spoiled_units", 0)

    return {
        "reward": episode_reward,
        "avg_price": np.mean(prices),
        "avg_price_volatility": np.std(prices),
        "total_demand": np.sum(demands),
        "prices": prices,
        "demands": demands,
        "inventories": inventories,
        "expiries": expiries,
        "spoilage": spoilage_total,
        "inventory_turnover": np.sum(demands) / initial_inventory if initial_inventory else 0
    }


def _markdown_action(obs, urgency_threshold=0.3, **kwargs):
    """
    Rule-based markdown policy for perishable goods.
    Reduces price aggressively as expiry approaches.
    """
    if len(obs.shape) > 1:
        obs = obs[0]
    
    # Get urgency score (last element if frame stacking, or specific index)
    base_obs_dim = 10
    if len(obs) > base_obs_dim:
        urgency_score = obs[-1]
    else:
        urgency_score = obs[-1] if len(obs) >= 10 else 0.0
    
    if urgency_score > urgency_threshold:
        return np.array([0])  # Maximum markdown (-15%)
    elif urgency_score > urgency_threshold * 0.5:
        return np.array([1])  # Moderate markdown (-10%)
    else:
        return np.array([3])  # Hold price


def _adaptive_action(obs, production_cost=5.0, **kwargs):
    """
    Sophisticated rule-based policy considering multiple factors.
    Represents industry best-practice heuristics.
    """
    if len(obs.shape) > 1:
        obs = obs[0]
    
    base_obs_dim = 10
    if len(obs) > base_obs_dim:
        frame_start = len(obs) - base_obs_dim
        current_price = obs[frame_start]
        competitor_price = obs[frame_start + 2]
        inventory_ratio = obs[frame_start + 8]
        urgency_score = obs[frame_start + 9]
    else:
        current_price = obs[0] if len(obs) > 0 else 10
        competitor_price = obs[2] if len(obs) > 2 else 10
        inventory_ratio = obs[8] if len(obs) > 8 else 0.5
        urgency_score = obs[9] if len(obs) > 9 else 0.0
    
    # Priority 1: Urgent markdown for near-expiry
    if urgency_score > 0.6:
        return np.array([0])
    
    # Priority 2: Competitive response
    if competitor_price > 0:
        price_diff_pct = (current_price - competitor_price) / competitor_price
        if price_diff_pct > 0.15:
            return np.array([1])
        elif price_diff_pct < -0.15 and inventory_ratio > 0.5:
            return np.array([5])
    
    # Priority 3: Inventory management
    if inventory_ratio > 0.7:
        return np.array([2])
    elif inventory_ratio < 0.2:
        return np.array([4])
    
    return np.array([3])


def run_multiple_baselines(env, n_episodes=10):
    """
    Run all baseline policies and return comparative results.
    """
    results = {}
    
    for policy in ["fixed", "random", "markdown", "adaptive"]:
        policy_results = []
        for _ in range(n_episodes):
            result = run_baseline_policy(env, policy_type=policy)
            policy_results.append(result)
        
        results[policy] = {
            "avg_reward": np.mean([r["reward"] for r in policy_results]),
            "std_reward": np.std([r["reward"] for r in policy_results]),
            "avg_price_volatility": np.mean([r["avg_price_volatility"] for r in policy_results]),
            "avg_demand": np.mean([r["total_demand"] for r in policy_results]),
            "avg_spoilage": np.mean([r["spoilage"] for r in policy_results]),
            "avg_turnover": np.mean([r["inventory_turnover"] for r in policy_results])
        }
    
    return results