"""
Test script to visualize power distribution in the hybrid energy system
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import HybridSystemDataGenerator
from tests.test_fault import FaultAnalyzer

# Set up directories
os.makedirs('output_', exist_ok=True)
# Create data generator
print("Generating dataset...")
data_gen = HybridSystemDataGenerator(seed=42)

# Generate a small dataset for testing (0.1 years = ~36 days)
df = data_gen.generate_dataset(
    start_date='2023-01-01',
    periods_years=0.9,
    output_file='notebooks/power_distribution_test.csv'
)

# Calculate statistics for each power source
sources = ['solar', 'battery', 'grid', 'generator']
stats = {}

for source in sources:
    power_col = f'{source}_power'
    if power_col in df.columns:
        stats[source] = {
            'min': df[power_col].min(),
            'max': df[power_col].max(),
            'mean': df[power_col].mean(),
            'std': df[power_col].std(),
            'zero_percent': (df[power_col] == 0).mean() * 100,
            'positive_percent': (df[power_col] > 0).mean() * 100
        }

# Print statistics
print("\nPower Source Statistics:")
for source, stat in stats.items():
    print(f"\n{source.capitalize()} Power:")
    print(f"  Min: {stat['min']:.2f} kW")
    print(f"  Max: {stat['max']:.2f} kW")
    print(f"  Mean: {stat['mean']:.2f} kW")
    print(f"  Std Dev: {stat['std']:.2f} kW")
    print(f"  Zero %: {stat['zero_percent']:.2f}%")
    print(f"  Active %: {stat['positive_percent']:.2f}%")

# Plot power distribution over time
plt.figure(figsize=(15, 10))

# Create a stacked area chart for positive values
positive_data = {}
for source in sources:
    power_col = f'{source}_power'
    if power_col in df.columns:
        positive_data[source] = df[power_col].clip(lower=0)

# Define colors for each source
colors = {
    'solar': 'gold',
    'battery': 'purple',
    'grid': 'green',
    'generator': 'red'
}

# Plot positive values (generation)
plt.stackplot(df.index, 
              [positive_data[s] for s in sources if s in positive_data],
              labels=[f"{s.capitalize()} output_" for s in sources if s in positive_data],
              colors=[colors[s] for s in sources if s in positive_data],
              alpha=0.7)

# Plot negative values as separate lines (consumption/charging)
for source in ['battery', 'grid']:
    power_col = f'{source}_power'
    if power_col in df.columns:
        negative_values = -df[power_col].clip(upper=0)
        if negative_values.max() > 0:  # Only plot if there are negative values
            plt.plot(df.index, negative_values, 
                    color=colors[source], linestyle='--',
                    label=f"{source.capitalize()} Charging/Export")

# Plot load demand
plt.plot(df.index, df['load_demand'], color='black', label='Load Demand', linewidth=2)

plt.title('Power Distribution in Hybrid Energy System', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('output_/power_distribution.png', dpi=300, bbox_inches='tight')

# Plot daily patterns - extract a typical week
typical_week = df.iloc[24*7:24*14]  # Second week of data

plt.figure(figsize=(15, 8))
for source in sources:
    power_col = f'{source}_power'
    if power_col in typical_week.columns:
        plt.plot(typical_week.index, typical_week[power_col], 
                 label=f"{source.capitalize()} Power", 
                 color=colors[source])

plt.plot(typical_week.index, typical_week['load_demand'], 
         color='black', label='Load Demand', linewidth=2)

plt.title('Power Sources and Load Demand - Typical Week', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('output_/typical_week_power.png', dpi=300, bbox_inches='tight')

# Plot power source contributions as pie chart
avg_contributions = {}
total_positive = 0

for source in sources:
    power_col = f'{source}_power'
    if power_col in df.columns:
        # Only count positive contributions (generation, not consumption)
        positive_only = df[power_col].clip(lower=0)
        avg_contributions[source] = positive_only.sum()
        total_positive += positive_only.sum()

# Convert to percentages
if total_positive > 0:
    for source in avg_contributions:
        avg_contributions[source] = (avg_contributions[source] / total_positive) * 100

plt.figure(figsize=(10, 10))
plt.pie(
    [avg_contributions[s] for s in sources if s in avg_contributions],
    labels=[f"{s.capitalize()}: {avg_contributions[s]:.1f}%" for s in sources if s in avg_contributions],
    colors=[colors[s] for s in sources if s in avg_contributions],
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    explode=[0.05 if s == 'solar' else 0 for s in sources if s in avg_contributions]
)
plt.title('Average Power Contribution by Source', fontsize=16)
plt.axis('equal')
plt.savefig('output_/power_contribution_pie.png', dpi=300, bbox_inches='tight')

# Create a visualization of the power dispatch priority
plt.figure(figsize=(12, 8))

# Create a horizontal bar chart showing priority
priorities = {
    'solar': 1,
    'battery': 2,
    'grid': 3,
    'generator': 4
}

# Create bars for each source with their mean power
plt.barh(
    [priorities[s] for s in sources],
    [stats[s]['mean'] for s in sources],
    color=[colors[s] for s in sources],
    alpha=0.7,
    height=0.6
)

# Add source labels
plt.yticks(
    [priorities[s] for s in sources],
    [f"Priority {priorities[s]}: {s.capitalize()}" for s in sources]
)

# Add mean power values as text
for i, source in enumerate(sources):
    plt.text(
        stats[source]['mean'] + 20, 
        priorities[source],
        f"{stats[source]['mean']:.1f} kW",
        va='center'
    )

plt.title('Power Dispatch Priority and Average output_', fontsize=16)
plt.xlabel('Average Power output_ (kW)', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.savefig('output_/power_dispatch_priority.png', dpi=300, bbox_inches='tight')

# Create a 24-hour profile for each source (averaged across all days)
hourly_profiles = {}
for source in sources:
    power_col = f'{source}_power'
    if power_col in df.columns:
        hourly_profiles[source] = df.groupby(df.index.hour)[power_col].mean()

# Also get load demand hourly profile
hourly_profiles['load'] = df.groupby(df.index.hour)['load_demand'].mean()

plt.figure(figsize=(14, 8))
for source in sources:
    plt.plot(
        hourly_profiles[source].index,
        hourly_profiles[source].values,
        label=f"{source.capitalize()} Power",
        color=colors[source],
        linewidth=2
    )

plt.plot(
    hourly_profiles['load'].index,
    hourly_profiles['load'].values,
    label='Load Demand',
    color='black',
    linewidth=2
)

plt.title('Average 24-Hour Power Profile by Source', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Average Power (kW)', fontsize=12)
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=10)
plt.savefig('output_/daily_power_profile.png', dpi=300, bbox_inches='tight')

# Create a weekly comparison plot
plt.figure(figsize=(16, 8))
for i in range(4):
    week = df.iloc[24*7*i:24*7*(i+1)]
    plt.subplot(1, 4, i+1)
    for source in sources:
        power_col = f'{source}_power'
        if power_col in week.columns:
            plt.plot(
                week.index,
                week[power_col],
                label=f"{source.capitalize()} Power",
                color=colors[source]
            )
    plt.plot(
        week.index,
        week['load_demand'],
        label='Load Demand',
        color='black'
    )
    plt.title(f"Week {i+1}", fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Power (kW)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output_/weekly_comparison.png', dpi=300, bbox_inches='tight')

# Create a FaultAnalyzer instance
fa = FaultAnalyzer(df)

# Plot time series of parameters with fault periods highlighted
fa.plot_time_series(['solar_power', 'battery_power', 'grid_power', 'generator_power', 'load_demand'])
plt.savefig('output_/time_series.png', dpi=300, bbox_inches='tight')

# Compare distribution of grid_voltage during LINE_SHORT_CIRCUIT faults
fa.compare_distributions('grid_voltage', 'LINE_SHORT_CIRCUIT')
print(fa.get_fault_events())
print(fa.fault_statistics())
fa.plot_fault_timeline()


print("\nDone! Check output_ directory for plots.")
