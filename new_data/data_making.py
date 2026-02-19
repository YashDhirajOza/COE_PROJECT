import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)  # reproducible

# ================== PARAMETERS ==================
start_date = '2015-01-01'
end_date   = '2025-12-31'
regions    = ['North', 'South', 'East', 'West', 'Central']
n_units_per_region = 6          # 3 IU + 3 GU per region → 30 total units

# Base monthly clinker volume per unit (tonnes) - realistic for large plants/clusters
base_volume = {
    'North':    850_000,
    'South':    950_000,
    'East':     720_000,
    'West':     680_000,
    'Central':  650_000
}

# Seasonal factors (Jan-Dec) - strong monsoon dip
seasonal_base = [1.08, 1.12, 1.15, 1.05, 0.95, 0.72, 0.65, 0.68, 0.82, 1.02, 1.10, 1.15]

# Region rainfall multipliers (higher in East/South)
rain_mult = {'North': 0.9, 'South': 1.25, 'East': 1.4, 'West': 0.85, 'Central': 1.0}

# ================================================

dates = pd.date_range(start_date, end_date, freq='MS')
df_list = []

unit_id = 0
for region in regions:
    for utype in ['IU', 'GU']:
        for i in range(3):   # 3 of each type per region
            unit_id += 1
            unit_name = f"{utype}_{region}_{i+1}"
            
            # Base for this unit
            base = base_volume[region] * (0.92 if utype == 'GU' else 1.08)  # GUs slightly lower on average
            
            temp = pd.DataFrame({'date': dates})
            temp['unit_id'] = unit_name
            temp['region']   = region
            temp['unit_type'] = utype
            temp['month']    = temp['date'].dt.month
            temp['year']     = temp['date'].dt.year
            
            # Trend (6% CAGR)
            temp['trend'] = 1 + 0.06 * (temp['year'] - 2015)
            
            # Seasonality
            temp['seasonal'] = [seasonal_base[m-1] for m in temp['month']]
            
            # COVID shock (Apr-Jun 2020)
            temp['covid'] = np.where((temp['year'] == 2020) & temp['month'].isin([4,5,6]), 0.68, 1.0)
            
            # Exogenous features
            temp['gdp_growth'] = 6.5 + 1.5 * np.sin(2*np.pi*(temp['year']-2015)/8) + np.random.normal(0,1.2, len(temp))
            temp['gdp_growth'] = temp['gdp_growth'].clip( -8, 9)   # 2020 dip
            
            temp['rainfall_mm'] = (120 + 80 * np.sin(2*np.pi*(temp['month']-6)/12)) * rain_mult[region] * (1 + np.random.normal(0,0.15,len(temp)))
            
            temp['infra_index'] = 40 + 4.5 * (temp['year'] - 2015) + np.random.normal(0,3,len(temp))
            
            temp['coal_price'] = 6200 + 280 * (temp['year'] - 2015) + 800 * np.sin(2*np.pi*temp['year']/3) + np.random.normal(0,450,len(temp))
            
            temp['interest_rate'] = 7.5 - 0.4 * (temp['year'] - 2015) + np.random.normal(0,0.8,len(temp))
            
            temp['capacity_util'] = 62 + 0.8 * (temp['year'] - 2015) + np.random.normal(0,4,len(temp))
            temp['capacity_util'] = temp['capacity_util'].clip(48, 82)
            
            # Declining clinker intensity (more blended cements)
            temp['clinker_ratio'] = 0.73 - 0.0045 * (temp['year'] - 2015)
            
            # Final target - realistic & correlated
            temp['clinker_volume'] = (
                base 
                * temp['trend'] 
                * temp['seasonal'] 
                * temp['covid']
                * (1 + 0.035 * (temp['gdp_growth'] - 6.5))      # GDP effect
                * np.exp(-0.0018 * temp['rainfall_mm'])          # monsoon penalty
                * (1 + 0.012 * (temp['infra_index'] - 70))       # infra boost
                * (0.98 + 0.000015 * (8000 - temp['coal_price'])) # higher coal → slightly lower volume
            ) * (1 + np.random.normal(0, 0.055, len(temp)))      # noise
            
            # Add some more columns for richer training
            temp['clinker_volume_lag1'] = temp['clinker_volume'].shift(1)
            temp['clinker_volume_lag12'] = temp['clinker_volume'].shift(12)
            temp['month_sin'] = np.sin(2 * np.pi * temp['month'] / 12)
            temp['month_cos'] = np.cos(2 * np.pi * temp['month'] / 12)
            
            df_list.append(temp)
            print(f"Generated {unit_name} → {len(temp)} rows")

# Combine
df = pd.concat(df_list, ignore_index=True)

# Final cleaning & sorting
df = df.sort_values(['unit_id', 'date']).reset_index(drop=True)
df['clinker_volume'] = df['clinker_volume'].round(0).astype(int)

print("\n=== DATASET READY ===")
print(f"Shape: {df.shape} rows × columns")
print(df[['date','unit_id','region','clinker_volume']].head(10))

# Save
df.to_csv('clinker_demand_training_dataset_3960rows.csv', index=False)
print("\nSaved as 'clinker_demand_training_dataset_3960rows.csv'")