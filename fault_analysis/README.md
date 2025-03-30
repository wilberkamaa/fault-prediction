# Fault Analysis and Prediction for Hybrid Energy Systems

This module provides tools for analyzing and predicting faults in hybrid energy systems using machine learning techniques.

## Overview

The fault analysis module consists of several components:

1. **Data Preparation** - Loads and preprocesses data, creates features for fault prediction
2. **Model Training** - Trains and evaluates machine learning models for fault prediction
3. **Visualization** - Creates visualizations of fault patterns and model performance
4. **Main Pipeline** - Orchestrates the entire fault analysis workflow

## Features

- **Fault Detection** - Binary classification to predict if a fault will occur within a specified time horizon
- **Fault Type Classification** - Multi-class classification to identify the specific type of fault
- **Early Warning System** - Predicts faults hours before they occur
- **Feature Importance Analysis** - Identifies key system parameters that indicate potential faults
- **Interactive Visualizations** - Provides insights into fault patterns and system behavior

## Fault Types

The system can detect and predict several types of faults:

1. **LINE_SHORT_CIRCUIT** - Electrical short circuits in power lines
2. **LINE_PROLONGED_UNDERVOLTAGE** - Sustained low voltage conditions
3. **INVERTER_IGBT_FAILURE** - Failures in inverter IGBT components
4. **GENERATOR_FIELD_FAILURE** - Field winding failures in the diesel generator
5. **GRID_VOLTAGE_SAG** - Temporary drops in grid voltage
6. **GRID_OUTAGE** - Complete loss of grid power
7. **BATTERY_OVERDISCHARGE** - Battery discharge beyond safe limits

## Machine Learning Approach

### Feature Engineering

- **Time-based Features** - Hour of day, day of week, month, cyclical encoding
- **Lag Features** - Historical values of system parameters
- **Window Features** - Statistical aggregations over time windows
- **Domain-specific Features** - Engineered features based on system knowledge

### Models

- **Random Forest** - Primary model for fault detection and classification
- **Gradient Boosting** - Alternative model for comparison
- **Logistic Regression** - Baseline model for interpretability

### Evaluation Metrics

- **Classification Metrics** - Precision, recall, F1-score, ROC-AUC
- **Early Detection Rate** - Percentage of faults detected before occurrence
- **Average Detection Time** - How many hours in advance faults are detected

## Usage

Run the main pipeline:

```bash
python main.py
```

This will:
1. Generate or load a dataset
2. Perform exploratory data analysis
3. Engineer features for fault prediction
4. Train and evaluate binary fault prediction models
5. Train and evaluate fault type classification models
6. Generate visualizations and performance metrics

## Output

The pipeline generates several outputs:

- **Trained Models** - Saved in the `models/` directory
- **Visualizations** - Saved in the `output/` directory
- **Performance Metrics** - Printed to console and saved in reports

## Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly
- SHAP (for model interpretability)

## Future Enhancements

- Severity prediction (regression models)
- Time-to-failure prediction
- Anomaly detection for unknown fault types
- Online learning for continuous model updating
- Integration with real-time monitoring systems
