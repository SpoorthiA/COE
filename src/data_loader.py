import pandas as pd
import numpy as np

def load_and_preprocess_instacart(data_dir):
    try:
        orders_path = f"{data_dir}/orders.csv"
        order_products_prior_path = f"{data_dir}/order_products__prior.csv"
        
        orders_df = pd.read_csv(orders_path)
        order_products_prior_df = pd.read_csv(order_products_prior_path)

        orders_relevant = orders_df[orders_df['eval_set'].isin(['prior', 'train'])]
        merged_df = pd.merge(
            order_products_prior_df,
            orders_relevant[['order_id', 'order_hour_of_day', 'order_dow']],
            on='order_id',
            how='inner'
        )
        return merged_df.groupby(['product_id', 'order_hour_of_day', 'order_dow']).size().reset_index(name='demand')
    except FileNotFoundError:
        print(f"Warning: Instacart data not found in {data_dir}. Using empty DataFrame.")
        return pd.DataFrame(columns=['product_id', 'order_hour_of_day', 'order_dow', 'demand'])

def load_and_preprocess_retail(file_path):
    try:
        retail = pd.read_excel(file_path)
        retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
        retail['hour'] = retail['InvoiceDate'].dt.hour
        retail['dow'] = retail['InvoiceDate'].dt.dayofweek
        retail = retail[(retail['UnitPrice'] > 0) & (retail['Quantity'] > 0)].copy()
        return retail.groupby(['StockCode', 'hour', 'dow']).agg({
            'Quantity': 'sum',
            'UnitPrice': 'mean'
        }).reset_index()
    except FileNotFoundError:
        print(f"Warning: Retail data not found at {file_path}. Using empty DataFrame.")
        return pd.DataFrame(columns=['StockCode', 'hour', 'dow', 'Quantity', 'UnitPrice'])

def load_and_preprocess_retailrocket(file_path):
    try:
        events = pd.read_csv(file_path)
        events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
        events = events.dropna(subset=['timestamp'])
        events['hour'] = events['timestamp'].dt.hour
        events['dow'] = events['timestamp'].dt.dayofweek
        return events.groupby(['itemid', 'hour', 'dow']).size().reset_index(name='interactions')
    except FileNotFoundError:
        print(f"Warning: RetailRocket data not found at {file_path}. Using empty DataFrame.")
        return pd.DataFrame(columns=['itemid', 'hour', 'dow', 'interactions'])

class PricingDataset:
    def __init__(self, instacart_dir, retail_path, retailrocket_path):
        self.instacart_data = load_and_preprocess_instacart(instacart_dir)
        self.retail_data = load_and_preprocess_retail(retail_path)
        self.retailrocket_data = load_and_preprocess_retailrocket(retailrocket_path)
        self.price_elasticity_map = self._calculate_price_elasticity()
        self.demand_patterns = self._create_demand_patterns()

    def _calculate_price_elasticity(self):
        """
        Calculates price elasticity for each product using a log-log linear regression.
        
        NOTE: This calculation can only be performed on the 'Online Retail' dataset
        because it is the only one that contains both transaction-level price ('UnitPrice')
        and quantity ('Quantity') information. The other datasets lack price data.
        """
        elasticity_map = {}
        default_elasticity = -1.0
        
        if self.retail_data.empty:
            return {0: default_elasticity}

        for product in self.retail_data['StockCode'].unique():
            product_data = self.retail_data[self.retail_data['StockCode'] == product]
            
            # Check for sufficient data points and price variation to calculate elasticity
            if len(product_data) > 10 and product_data['UnitPrice'].nunique() > 1:
                # Use log-log model: log(Quantity) = intercept + elasticity * log(Price)
                price = np.log(product_data['UnitPrice'].clip(lower=0.01))
                quantity = np.log(product_data['Quantity'].clip(lower=0.01))
                
                try:
                    elasticity = np.polyfit(price, quantity, 1)[0]
                    # Clamp elasticity to a reasonable range
                    elasticity_map[product] = max(min(elasticity, -0.1), -2.5)
                except np.linalg.LinAlgError:
                    # Handle cases where the calculation fails
                    elasticity_map[product] = default_elasticity
            else:
                # Not enough data or price variation, assign default elasticity
                elasticity_map[product] = default_elasticity
        
        return elasticity_map if elasticity_map else {0: default_elasticity}

    def _create_demand_patterns(self):
        """
        Creates a normalized demand pattern for each hour of the day and day of the week
        by averaging the demand from all available datasets.
        """
        patterns = {}
        # Pre-group dataframes to speed up lookups inside the loop
        instacart_grouped = self.instacart_data.groupby(['order_hour_of_day', 'order_dow'])['demand'].mean()
        retail_grouped = self.retail_data.groupby(['hour', 'dow'])['Quantity'].mean()
        retailrocket_grouped = self.retailrocket_data.groupby(['hour', 'dow'])['interactions'].mean()

        all_demands = []
        # Collect all non-null average demands to compute a global max for scaling
        if not instacart_grouped.empty: all_demands.extend(instacart_grouped.tolist())
        if not retail_grouped.empty: all_demands.extend(retail_grouped.tolist())
        if not retailrocket_grouped.empty: all_demands.extend(retailrocket_grouped.tolist())
        
        global_max_demand = max(all_demands) if all_demands else 1.0

        for hour in range(24):
            for dow in range(7):
                demands = []
                # Efficiently lookup pre-calculated means
                if (hour, dow) in instacart_grouped.index: demands.append(instacart_grouped.loc[(hour, dow)])
                if (hour, dow) in retail_grouped.index: demands.append(retail_grouped.loc[(hour, dow)])
                if (hour, dow) in retailrocket_grouped.index: demands.append(retailrocket_grouped.loc[(hour, dow)])
                
                if demands:
                    # Normalize by the global max demand to get a consistent scale
                    patterns[(hour, dow)] = np.mean([d / global_max_demand for d in demands])
                else:
                    patterns[(hour, dow)] = 0.0 # No data for this slot, assume zero demand
        return patterns