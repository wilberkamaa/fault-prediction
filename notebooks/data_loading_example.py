"""
Example of how to correctly load or generate data
"""
import sys
import os
import pandas as pd

# Add the project root to the path
sys.path.append('..')

# Import the data generator
from src.data_generator import HybridSystemDataGenerator

# Set random seed for reproducibility
RANDOM_SEED = 42

# Check if dataset already exists
dataset_path = 'data/gen_data_test.csv'
generate_new_data = False  # Set to True to generate new data

if generate_new_data or not os.path.exists(dataset_path):
    print("Generating new dataset...")
    data_gen = HybridSystemDataGenerator(seed=RANDOM_SEED)
    # Generate dataset for 1 year
    df = data_gen.generate_dataset(
        start_date='2023-01-01',
        periods_years=1,
        output_file=dataset_path
    )
    print(f"Dataset generated and saved to {dataset_path}")
else:
    print(f"Loading existing dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    # If the index is a datetime, parse it
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    print("Dataset loaded successfully")

# Now you can use df for your analysis
print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {df.columns.tolist()[:5]}...")
