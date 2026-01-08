# Dynamic Pricing System

This project implements a Reinforcement Learning-based dynamic pricing system.

## Project Structure
- `src/data_processing/`: Scripts to prepare data from Instacart and RetailRocket datasets.
- `src/models/`: RL Environment and Training scripts.
- `src/ui/`: Streamlit application for visualization.
- `Data/`: Contains the raw datasets.

## Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy gymnasium stable-baselines3 streamlit matplotlib shimmy
   ```

## Usage

### 1. Prepare Data
Generate synthetic metadata and demand patterns from the raw data.
```bash
python src/data_processing/prepare_data.py
```

### 2. Train Agent
Train the PPO agent on the environment.
```bash
python src/models/train_agent.py
```

### 3. Run UI
Launch the dashboard to interact with the pricing system.
```bash
streamlit run src/ui/app.py
```

## Methodology
- **Environment**: Simulates a market with a competitor. Demand is a function of price elasticity, competitor price, and day-of-week seasonality.
- **Agent**: Uses Proximal Policy Optimization (PPO) to learn the optimal pricing policy.
- **Data**: Uses Instacart order volume to estimate demand seasonality.
