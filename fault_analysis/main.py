"""
Main script for fault analysis and prediction in hybrid energy systems.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import custom modules
from data_preparation import (
    load_dataset, explore_faults, create_time_features,
    create_lag_features, create_window_features,
    prepare_fault_dataset, split_and_scale_data
)
from model_training import (
    train_binary_classifier, evaluate_classifier,
    train_multiclass_classifier, evaluate_multiclass_classifier,
    save_model, load_model
)
from visualization import (
    plot_fault_timeline, plot_fault_heatmap,
    plot_system_parameters_during_faults, plot_feature_correlations,
    plot_feature_distributions, plot_dimensionality_reduction,
    plot_model_predictions_over_time, plot_early_detection_analysis
)

def main():
    """Main function to run the fault analysis pipeline."""
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    print("="*80)
    print("HYBRID ENERGY SYSTEM FAULT ANALYSIS")
    print("="*80)
    
    # Step 1: Generate or load dataset
    print("\n1. Data Generation/Loading")
    print("-"*40)
    
    # Check if dataset exists
    dataset_path = 'notebooks/fault_analysis_data.csv'
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        df = load_dataset(dataset_path)
    else:
        print("Generating new dataset...")
        from src.data_generator import HybridSystemDataGenerator
        
        data_gen = HybridSystemDataGenerator(seed=42)
        df = data_gen.generate_dataset(
            start_date='2023-01-01',
            periods_years=1,
            output_file=dataset_path
        )
    
    # Step 2: Exploratory Data Analysis
    print("\n2. Exploratory Data Analysis")
    print("-"*40)
    
    # Explore fault distribution
    fault_counts, fault_rate = explore_faults(df)
    
    # Visualize faults
    plot_fault_timeline(df)
    plot_fault_heatmap(df)
    plot_system_parameters_during_faults(df)
    
    # Step 3: Feature Engineering
    print("\n3. Feature Engineering")
    print("-"*40)
    
    # Create time features
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
    print("Creating lag features...")
    df_features = create_lag_features(df, system_cols, lag_periods=[1, 3, 6, 12, 24])
    
    print("Creating window features...")
    df_features = create_window_features(df_features, system_cols, window_sizes=[6, 12, 24])
    
    # Analyze feature correlations with faults
    plot_feature_correlations(df_features)
    
    # Plot feature distributions for top correlated features
    top_features = [
        'battery_soc', 'battery_power', 'grid_voltage',
        'solar_cell_temp', 'generator_temperature'
    ]
    plot_feature_distributions(df_features, top_features)
    
    # Step 4: Binary Fault Prediction (Any Fault)
    print("\n4. Binary Fault Prediction (Any Fault)")
    print("-"*40)
    
    # Prepare dataset for binary classification
    print("Preparing dataset for binary fault prediction...")
    prediction_horizon = 24  # Predict faults 24 hours ahead
    X, y = prepare_fault_dataset(
        df_features, 
        target_fault_type=None,  # Any fault
        prediction_horizon=prediction_horizon
    )
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Positive samples in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Positive samples in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Train binary classifier
    model_path = 'models/fault_prediction_rf.pkl'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model('fault_prediction_rf')
    else:
        print("Training Random Forest classifier...")
        model = train_binary_classifier(X_train, y_train, model_type='rf')
        save_model(model, 'fault_prediction_rf')
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_classifier(model, X_test, y_test, feature_names=X_train.columns)
    
    # Visualize model predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Plot predictions over time
    plot_model_predictions_over_time(
        df.iloc[X_test.index], 
        y_test, 
        y_pred, 
        y_prob
    )
    
    # Analyze early detection performance
    early_detection_metrics = plot_early_detection_analysis(
        df.iloc[X_test.index], 
        y_test, 
        y_pred, 
        prediction_horizon
    )
    
    # Visualize feature space
    print("\nGenerating dimensionality reduction visualizations...")
    plot_dimensionality_reduction(X_test, y_test, method='tsne')
    plot_dimensionality_reduction(X_test, y_test, method='pca')
    
    # Step 5: Fault Type Prediction (Multi-class)
    print("\n5. Fault Type Prediction (Multi-class)")
    print("-"*40)
    
    # Prepare dataset for multi-class classification
    # Only include samples where faults occur
    fault_mask = df_features['fault_type'] != 'NO_FAULT'
    if fault_mask.sum() > 0:
        # Extract features and target
        X_fault = X[fault_mask]
        y_fault = df_features.loc[fault_mask, 'fault_type']
        
        # Split data
        X_train_fault, X_test_fault, y_train_fault, y_test_fault = train_test_split(
            X_fault, y_fault, test_size=0.2, random_state=42
        )
        
        print(f"Fault type training set shape: {X_train_fault.shape}")
        print(f"Fault type test set shape: {X_test_fault.shape}")
        print(f"Fault type distribution in training: {y_train_fault.value_counts()}")
        
        # Train multi-class classifier
        model_path = 'models/fault_type_prediction_rf.pkl'
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            fault_type_model = load_model('fault_type_prediction_rf')
        else:
            print("Training Random Forest classifier for fault type prediction...")
            fault_type_model = train_multiclass_classifier(X_train_fault, y_train_fault, model_type='rf')
            save_model(fault_type_model, 'fault_type_prediction_rf')
        
        # Evaluate model
        print("\nEvaluating fault type prediction model...")
        fault_type_metrics = evaluate_multiclass_classifier(
            fault_type_model, 
            X_test_fault, 
            y_test_fault, 
            feature_names=X_train.columns
        )
    else:
        print("No fault samples available for multi-class classification")
    
    # Step 6: Severity Prediction (Regression)
    # This could be implemented as an extension
    
    print("\nFault analysis complete! Check the 'output' directory for visualizations.")

if __name__ == "__main__":
    main()
