import pandas as pd
import numpy as np

# Load original forecast dataset
df = pd.read_csv('D:/COE_project/new_data/clinker_demand_training_dataset_3960rows.csv')
df['date'] = pd.to_datetime(df['date'])
dates = sorted(df['date'].unique())

# ---------------------------------------------------------
# 1. Cost Parameters Dataset (Diesel & Rail Freight)
# ---------------------------------------------------------
diesel_prices, freight_rates = np.zeros(len(dates)), np.zeros(len(dates))
base_diesel, base_freight = 55.0, 1.45 

np.random.seed(42)
for i, d in enumerate(dates):
    month, year = pd.to_datetime(d).month, pd.to_datetime(d).year
    
    # Diesel Logic (Macro trends + Noise)
    if year <= 2017: trend = i * 0.15 
    elif 2018 <= year <= 2019: trend = i * 0.30 
    elif year == 2020 and month in [4, 5, 6]: trend = i * 0.20 - 8.0
    else: trend = i * 0.45 
        
    diesel_prices[i] = np.clip(base_diesel + trend + np.random.normal(0, 1.5), 50, 105)
    
    # Rail Freight Logic (Busy season vs Lean season)
    current_base = base_freight + (0.12 * ((year - 2015) // 2))
    fr = current_base if month in [7, 8, 9] else current_base * 1.15
    freight_rates[i] = fr + np.random.normal(0, 0.02)

pd.DataFrame({
    'date': dates,
    'diesel_price_inr_liter': np.round(diesel_prices, 2),
    'rail_freight_inr_ntkm': np.round(freight_rates, 3)
}).to_csv('mip_cost_parameters.csv', index=False)

# ---------------------------------------------------------
# 2. Demand & Supply Constraints
# ---------------------------------------------------------
# GU Demands (Directly from ML Forecast)
gu_df = df[df['unit_type'] == 'GU'][['date', 'unit_id', 'clinker_volume']]
gu_df = gu_df.rename(columns={'clinker_volume': 'demand_mt', 'unit_id': 'gu_id'})
gu_df.to_csv('mip_gu_demands.csv', index=False)

# IU Capacities (Base volume + Buffer + Shocks)
iu_df = df[df['unit_type'] == 'IU'][['date', 'unit_id', 'clinker_volume', 'month', 'year']].copy()
iu_df['capacity_mt'] = (iu_df['clinker_volume'] * np.random.uniform(1.1, 1.3, len(iu_df))).astype(int)

# Apply Shocks (Monsoon & COVID)
monsoon_mask = iu_df['month'].isin([7, 8])
iu_df.loc[monsoon_mask, 'capacity_mt'] = (iu_df.loc[monsoon_mask, 'capacity_mt'] * 0.85).astype(int)
covid_mask = (iu_df['year'] == 2020) & (iu_df['month'] == 4)
iu_df.loc[covid_mask, 'capacity_mt'] = (iu_df.loc[covid_mask, 'capacity_mt'] * 0.20).astype(int)

iu_df[['date', 'unit_id', 'capacity_mt']].rename(columns={'unit_id': 'iu_id'}).to_csv('mip_iu_capacities.csv', index=False)

# ---------------------------------------------------------
# 3. Transportation Route Matrix
# ---------------------------------------------------------
routes = []
for iu in iu_df['unit_id'].unique():
    iu_region = iu.split('_')[1]
    for gu in gu_df['gu_id'].unique():
        gu_region = gu.split('_')[1]
        
        # Intra-region vs Inter-region distances
        distance = np.random.randint(50, 400) if iu_region == gu_region else np.random.randint(400, 1500)
        
        routes.append({
            'iu_id': iu,
            'gu_id': gu,
            'distance_km': distance,
            'rail_feasible': 1 if distance > 300 else 0, # Rail only viable for long hauls
            'road_feasible': 1
        })

pd.DataFrame(routes).to_csv('mip_transport_routes.csv', index=False)