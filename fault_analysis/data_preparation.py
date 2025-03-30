"""
Data preparation module for fault analysis in hybrid energy systems.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_dataset(file_path):
    """Load the dataset from CSV or Parquet file."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def explore_faults(df):
    """Explore fault distribution and characteristics."""
    # Count fault occurrences
    fault_counts = df['fault_type'].value_counts()
    total_faults = (df['fault_type'] != 'NO_FAULT').sum()
    fault_rate = total_faults / len(df) * 100
    
    print(f"Total records: {len(df)}")
    print(f"Total fault events: {total_faults} ({fault_rate:.2f}%)")
    print("\nFault type distribution:")
    print(fault_counts)
    
    # Plot fault distribution
    plt.figure(figsize=(12, 6))
    fault_counts[fault_counts.index != 'NO_FAULT'].plot(kind='bar', color='crimson')
    plt.title('Fault Type Distribution')
    plt.ylabel('Count')
    plt.xlabel('Fault Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/fault_distribution.png', dpi=300, bbox_inches='tight')
    
    # Plot fault severity distribution
    plt.figure(figsize=(12, 6))
    fault_data = df[df['fault_type'] != 'NO_FAULT']
    sns.histplot(data=fault_data, x='fault_severity', hue='fault_type', 
                 multiple='stack', bins=20, alpha=0.7)
    plt.title('Fault Severity Distribution by Type')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/fault_severity_distribution.png', dpi=300, bbox_inches='tight')
    
    return fault_counts, fault_rate

def create_time_features(df):
    """Create time-based features for the dataset."""
    # Add hour of day
    df['hour'] = df.index.hour
    
    # Add day of week
    df['day_of_week'] = df.index.dayofweek
    
    # Add month
    df['month'] = df.index.month
    
    # Add is_weekend flag
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def create_lag_features(df, columns, lag_periods=[1, 3, 6, 12, 24]):
    """Create lag features for specified columns."""
    for col in columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN values from lag creation
    df = df.dropna()
    
    return df

def create_window_features(df, columns, window_sizes=[6, 12, 24]):
    """Create rolling window statistical features."""
    for col in columns:
        for window in window_sizes:
            # Rolling mean
            df[f'{col}_mean_{window}h'] = df[col].rolling(window=window).mean()
            
            # Rolling standard deviation
            df[f'{col}_std_{window}h'] = df[col].rolling(window=window).std()
            
            # Rolling min/max
            df[f'{col}_min_{window}h'] = df[col].rolling(window=window).min()
            df[f'{col}_max_{window}h'] = df[col].rolling(window=window).max()
            
            # Rolling range (max - min)
            df[f'{col}_range_{window}h'] = (
                df[f'{col}_max_{window}h'] - df[f'{col}_min_{window}h']
            )
    
    # Drop rows with NaN values from window creation
    df = df.dropna()
    
    return df

def prepare_fault_dataset(df, target_fault_type=None, prediction_horizon=24):
    """
    Prepare dataset for fault prediction.
    
    Args:
        df: Input DataFrame
        target_fault_type: Specific fault type to predict, or None for any fault
        prediction_horizon: Hours ahead to predict fault
    
    Returns:
        X: Features DataFrame
        y: Target Series (1 if fault occurs within prediction horizon)
    """
    # Create target variable: 1 if fault occurs within next N hours
    if target_fault_type:
        # For specific fault type
        fault_mask = df['fault_type'] == target_fault_type
    else:
        # For any fault
        fault_mask = df['fault_type'] != 'NO_FAULT'
    
    # Create target: 1 if fault occurs within prediction horizon
    y = pd.Series(0, index=df.index)
    for i in range(len(df) - prediction_horizon):
        if fault_mask.iloc[i:i+prediction_horizon].any():
            y.iloc[i] = 1
    
    # Select features (exclude fault-related columns and target)
    fault_cols = [col for col in df.columns if col.startswith('fault_')]
    X = df.drop(columns=fault_cols)
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale numerical features
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # One-hot encode categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse=False, drop='first')
        cat_encoded_train = encoder.fit_transform(X_train[cat_cols])
        cat_encoded_test = encoder.transform(X_test[cat_cols])
        
        # Create DataFrames with encoded features
        cat_encoded_cols = encoder.get_feature_names_out(cat_cols)
        cat_encoded_train_df = pd.DataFrame(
            cat_encoded_train, 
            columns=cat_encoded_cols, 
            index=X_train.index
        )
        cat_encoded_test_df = pd.DataFrame(
            cat_encoded_test, 
            columns=cat_encoded_cols, 
            index=X_test.index
        )
        
        # Drop original categorical columns and add encoded ones
        X_train = X_train.drop(columns=cat_cols).join(cat_encoded_train_df)
        X_test = X_test.drop(columns=cat_cols).join(cat_encoded_test_df)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Generate or load dataset
    from src.data_generator import HybridSystemDataGenerator
    
    print("Generating dataset for fault analysis...")
    data_gen = HybridSystemDataGenerator(seed=42)
    df = data_gen.generate_dataset(
        start_date='2023-01-01',
        periods_years=1,
        output_file='notebooks/fault_analysis_data.csv'
    )
    
    # Alternatively, load existing dataset
    # df = load_dataset('notebooks/fault_analysis_data.csv')
    
    # Explore faults
    fault_counts, fault_rate = explore_faults(df)
    
    # Create features
    print("Creating features for fault prediction...")
    df = create_time_features(df)
    
    # Select columns for feature engineering
    system_cols = [
        'solar_power', 'solar_cell_temp', 'solar_efficiency',
        'battery_power', 'battery_soc', 'battery_temperature',
        'grid_power', 'grid_voltage', 'grid_frequency',
        'generator_power', 'generator_temperature', 'generator_fuel_level',
        'load_demand', 'weather_temperature', 'weather_humidity'
    ]
    
    # Create lag and window features
    df = create_lag_features(df, system_cols, lag_periods=[1, 3, 6, 12, 24])
    df = create_window_features(df, system_cols, window_sizes=[6, 12, 24])
    
    # Prepare dataset for fault prediction (any fault within next 24 hours)
    X, y = prepare_fault_dataset(df, target_fault_type=None, prediction_horizon=24)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    print(f"Prepared training set with shape: {X_train.shape}")
    print(f"Prepared test set with shape: {X_test.shape}")
    print(f"Positive samples in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Positive samples in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Save processed datasets
    X_train.to_csv('notebooks/X_train.csv')
    X_test.to_csv('notebooks/X_test.csv')
    y_train.to_csv('notebooks/y_train.csv')
    y_test.to_csv('notebooks/y_test.csv')
    
    print("Data preparation complete. Files saved to notebooks/ directory.")
