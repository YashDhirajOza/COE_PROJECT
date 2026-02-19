import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

# Load Data
costs_df = pd.read_csv('mip_cost_parameters.csv')
demands_df = pd.read_csv('mip_gu_demands.csv')
capacities_df = pd.read_csv('mip_iu_capacities.csv')
routes_df = pd.read_csv('mip_transport_routes.csv')

dates = costs_df['date'].unique()

TRUCK_CAPACITY_MT = 30
TRAIN_CAPACITY_MT = 3000

all_results = []
total_11yr_cost = 0.0

print(f"Starting Strict MILP optimization for {len(dates)} months...")
start_time = time.time()

# Pre-compute routes array
routes = []
for _, row in routes_df.iterrows():
    routes.append((row['iu_id'], row['gu_id'], row['distance_km'], row['road_feasible'], row['rail_feasible']))

# Solve month by month
for month_idx, target_date in enumerate(dates):
    # Costs
    costs = costs_df[costs_df['date'] == target_date].iloc[0]
    diesel_price = costs['diesel_price_inr_liter']
    rail_freight = costs['rail_freight_inr_ntkm']
    road_cost_per_mt_km = (diesel_price / 3.5) / 30.0

    # Supply and Demand dictionaries
    demands = demands_df[demands_df['date'] == target_date].set_index('gu_id')['demand_mt'].to_dict()
    capacities = capacities_df[capacities_df['date'] == target_date].set_index('iu_id')['capacity_mt'].to_dict()
    
    iu_list = list(capacities.keys())
    gu_list = list(demands.keys())

    # Build Variables
    variables = []
    for iu, gu, dist, road_f, rail_f in routes:
        if road_f == 1:
            variables.append({'iu': iu, 'gu': gu, 'mode': 'road', 
                              'cost': road_cost_per_mt_km * dist * TRUCK_CAPACITY_MT, 'cap': TRUCK_CAPACITY_MT})
        if rail_f == 1:
            variables.append({'iu': iu, 'gu': gu, 'mode': 'rail', 
                              'cost': rail_freight * dist * TRAIN_CAPACITY_MT, 'cap': TRAIN_CAPACITY_MT})

    n_vars = len(variables)
    c = np.array([v['cost'] for v in variables])

    # Build Matrices
    A_supply = np.zeros((len(iu_list), n_vars))
    b_supply = np.zeros(len(iu_list))
    for i, iu in enumerate(iu_list):
        b_supply[i] = capacities[iu]
        for j, var in enumerate(variables):
            if var['iu'] == iu: A_supply[i, j] = var['cap']

    A_demand = np.zeros((len(gu_list), n_vars))
    b_demand = np.zeros(len(gu_list))
    for i, gu in enumerate(gu_list):
        b_demand[i] = demands[gu]
        for j, var in enumerate(variables):
            if var['gu'] == gu: A_demand[i, j] = var['cap']

    A = np.vstack([A_supply, A_demand])
    lb = np.concatenate([np.zeros(len(iu_list)), b_demand])
    ub = np.concatenate([b_supply, np.full(len(gu_list), np.inf)])
    
    # 1 for integer constraint
    integrality = np.ones(n_vars)

    # Solve
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=Bounds(0, np.inf), integrality=integrality)

    if res.success:
        total_11yr_cost += res.fun
        for j, val in enumerate(res.x):
            if val > 0.5:
                v = variables[j]
                qty_mt = val * v['cap']
                all_results.append({
                    'Date': target_date,
                    'Supplying_IU': v['iu'],
                    'Receiving_GU': v['gu'],
                    'Mode': v['mode'],
                    'Units_Dispatched': int(val),
                    'Unit_Type': 'Trucks (30 MT)' if v['mode'] == 'road' else 'Trains (3000 MT)',
                    'Quantity_MT': qty_mt,
                    'Total_Cost_INR': round(val * v['cost'], 2)
                })
        
        if (month_idx + 1) % 24 == 0:
            print(f"[{month_idx + 1}/{len(dates)}] Solved up to {target_date}...")
    else:
        print(f"Warning: Optimization failed for {target_date} - {res.message}")

end_time = time.time()
master_df = pd.DataFrame(all_results)
master_df.to_csv('master_logistics_roadmap_132_months.csv', index=False)

print(f"\nOptimization Complete in {round(end_time - start_time, 2)} seconds.")
print(f"Total Routes Dispatched over 11 Years: {len(master_df)}")
print(f"Total 11-Year Logistics Cost: ₹ {total_11yr_cost:,.2f}")

# Quick summary of units
summary = master_df.groupby('Unit_Type')['Units_Dispatched'].sum().reset_index()
print("\nFleet Summary (11 Years):")
print(summary.to_string(index=False))