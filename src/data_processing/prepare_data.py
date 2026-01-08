import pandas as pd
import numpy as np
import os

def prepare_data():
    print("Loading data...")
    # Define paths
    base_path = "Data/instacart-market-basket-analysis"
    orders_path = os.path.join(base_path, "orders.csv")
    order_products_path = os.path.join(base_path, "order_products__train.csv")
    products_path = os.path.join(base_path, "products.csv")

    # Load subset of data to speed up processing
    # We only need train set orders for demand
    orders = pd.read_csv(orders_path)
    orders = orders[orders['eval_set'] == 'train']
    
    order_products = pd.read_csv(order_products_path)
    products = pd.read_csv(products_path)

    print("Merging data...")
    # Merge orders with product details
    merged = orders.merge(order_products, on='order_id')
    
    # Calculate "Day Index"
    # Since we don't have absolute dates, we'll use order_dow and some accumulation if possible.
    # However, for simplicity in this simulation, we will treat each order as a demand signal 
    # and aggregate by 'order_id' as a proxy for time steps, or just aggregate by 'order_dow'.
    # Better approach for RL: Create a time series.
    # We will assume the dataset represents a snapshot. 
    # Let's aggregate by product to find the top products.
    
    top_products = merged['product_id'].value_counts().head(5).index.tolist()
    print(f"Top 5 products: {top_products}")
    
    # Filter for top products
    df_top = merged[merged['product_id'].isin(top_products)].copy()
    df_top = df_top.merge(products, on='product_id')
    
    # Generate Synthetic Metadata (Base Price, Elasticity)
    # This is needed because the dataset doesn't have prices, and we need ground truth for the environment.
    product_meta = []
    for pid in top_products:
        pname = products[products['product_id'] == pid]['product_name'].values[0]
        base_price = np.round(np.random.uniform(5, 25), 2)
        # Elasticity: typically negative. -1.5 to -3.0 means elastic.
        elasticity = np.round(np.random.uniform(-3.0, -1.1), 2) 
        product_meta.append({
            'product_id': pid,
            'product_name': pname,
            'base_price': base_price,
            'elasticity': elasticity
        })
    
    meta_df = pd.DataFrame(product_meta)
    
    # Save processed data
    os.makedirs("src/data", exist_ok=True)
    meta_df.to_csv("src/data/product_metadata.csv", index=False)
    
    # We also need a "Demand History" to seed the environment.
    # We'll create a synthetic daily demand history based on the real transaction counts
    # but smoothed out.
    
    # Group by product and 'order_dow' to get a weekly pattern
    weekly_demand = df_top.groupby(['product_id', 'order_dow']).size().reset_index(name='avg_demand')
    # Normalize demand to be reasonable per day (e.g., 10-100 units)
    weekly_demand['avg_demand'] = weekly_demand['avg_demand'] / weekly_demand['avg_demand'].max() * 50 + 10
    weekly_demand['avg_demand'] = weekly_demand['avg_demand'].astype(int)
    
    weekly_demand.to_csv("src/data/demand_patterns.csv", index=False)
    
    print("Data preparation complete.")
    print(meta_df)

if __name__ == "__main__":
    prepare_data()
