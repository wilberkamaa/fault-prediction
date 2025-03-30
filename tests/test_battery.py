#!/usr/bin/env python3
"""
Test script for battery system behavior
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import HybridSystemDataGenerator

# Set up directories
os.makedirs('output', exist_ok=True)

# Create data generator with a shorter timeframe for testing
print("Generating dataset...")
data_gen = HybridSystemDataGenerator(seed=42)

# Generate a small dataset for testing
df = data_gen.generate_dataset(
    start_date='2023-01-01',
    periods_years=0.1,  # ~36 days
    output_file='notebooks/synthetic_data_test.csv'
)

# Override battery data to force alternating charge/discharge patterns
hours = len(df)
# Create alternating charge/discharge pattern
battery_power = np.zeros(hours)
battery_soc = np.zeros(hours)

# Start at 50% SOC
battery_soc[0] = 0.5
last_soc = battery_soc[0]

# Create forced patterns
for i in range(1, hours):
    # Every 12 hours, switch between charging and discharging
    if i % 12 < 6:  # Charge for 6 hours
        power = -200 * (0.7 + 0.3 * np.random.random())  # Negative = charging
        # Calculate SOC increase (simplified)
        soc_change = abs(power) * 0.9 / 1000  # 90% efficiency, 1000 kWh capacity
        battery_soc[i] = min(0.95, last_soc + soc_change)
    else:  # Discharge for 6 hours
        power = 200 * (0.7 + 0.3 * np.random.random())  # Positive = discharging
        # Calculate SOC decrease (simplified)
        soc_change = power / (0.9 * 1000)  # 90% efficiency, 1000 kWh capacity
        battery_soc[i] = max(0.15, last_soc - soc_change)
    
    # Add some random variation
    battery_power[i] = power + np.random.normal(0, 10)
    battery_soc[i] += np.random.normal(0, 0.01)
    battery_soc[i] = np.clip(battery_soc[i], 0.15, 0.95)
    
    last_soc = battery_soc[i]

# Override the dataframe values
df['battery_power'] = battery_power
df['battery_soc'] = battery_soc

# Calculate statistics
min_soc = df['battery_soc'].min() * 100
max_soc = df['battery_soc'].max() * 100
mean_soc = df['battery_soc'].mean() * 100
std_soc = df['battery_soc'].std() * 100

min_power = df['battery_power'].min()
max_power = df['battery_power'].max()
mean_power = df['battery_power'].mean()
std_power = df['battery_power'].std()

print(f"\nBattery Statistics:")
print(f"Min SOC: {min_soc:.2f}%")
print(f"Max SOC: {max_soc:.2f}%")
print(f"Mean SOC: {mean_soc:.2f}%")
print(f"SOC Standard Deviation: {std_soc:.2f}%")

print(f"\nBattery Power Statistics:")
print(f"Min Power: {min_power:.2f} kW")
print(f"Max Power: {max_power:.2f} kW")
print(f"Mean Power: {mean_power:.2f} kW")
print(f"Power Standard Deviation: {std_power:.2f} kW")

# Plot battery SOC
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df['battery_soc'], color='blue')
plt.title('Battery State of Charge (SOC)')
plt.xlabel('Time')
plt.ylabel('SOC (0-1)')
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('output/battery_soc.png')

# Plot battery power
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df['battery_power'], color='green')
plt.title('Battery Power (Positive = Discharge, Negative = Charge)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.grid(True)
plt.savefig('output/battery_power.png')

# Plot power balance
plt.figure(figsize=(12, 6))
# Create a stacked area chart
plt.stackplot(df.index, 
              df['solar_power'], 
              df['battery_power'].clip(lower=0),  # Only positive values (discharge)
              df['generator_power'], 
              df['grid_power'].clip(lower=0),  # Only positive values (import)
              labels=['Solar', 'Battery Discharge', 'Generator', 'Grid Import'],
              colors=['yellow', 'green', 'red', 'blue'],
              alpha=0.7)

# Plot negative values as separate lines
plt.plot(df.index, -df['battery_power'].clip(upper=0), color='darkgreen', 
         label='Battery Charge', linestyle='--')
plt.plot(df.index, -df['grid_power'].clip(upper=0), color='darkblue', 
         label='Grid Export', linestyle='--')

# Plot load demand
plt.plot(df.index, df['load_demand'], color='black', label='Load Demand', linewidth=2)

plt.title('Power Balance')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('output/power_balance.png')

print("\nDone! Check output directory for plots.")
