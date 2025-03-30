# Energy System Fault Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

A comprehensive solution for fault prediction in hybrid energy systems, combining synthetic data generation, machine learning models, and an interactive dashboard.

## 📋 Overview

This project provides tools for:

- **Synthetic Data Generation**: Create realistic time-series data for hybrid energy systems
- **Fault Detection**: Train and evaluate ML models (Random Forest and XGBoost) for fault prediction
- **Interactive Dashboard**: Visualize system data and predictions through a Streamlit interface

The system is designed to detect and predict faults in a hybrid energy system with components including:
- 1500 kW Solar PV System
- 1 MVA Diesel Generator
- 3 MWh Battery Storage
- 25 kV Grid Connection

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wilberkamau/fault-prediction.git
cd fault-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Generation

Generate synthetic data for a hybrid energy system:

```python
from src.data_generator import HybridSystemDataGenerator

# Initialize generator
generator = HybridSystemDataGenerator(seed=42)

# Generate 1 year of data
df = generator.generate_dataset(
    start_date='2023-01-01',
    periods_years=1,
    output_file='data/hybrid_system_data.parquet'
)
```

### Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard provides:
- System Overview visualization
- Fault Analysis and prediction
- Feature Importance analysis
- Performance Metrics evaluation

## 📊 Dashboard Features

### 1. System Overview
- Power distribution visualization
- Battery status monitoring
- Weather conditions display

### 2. Fault Analysis
- Fault probability over time
- Fault distribution visualization
- High-risk period identification

### 3. Feature Importance
- Feature importance visualization for both models
- Advanced feature analysis
- Top feature distribution and correlation analysis

### 4. Performance Metrics
- Confusion matrix visualization
- Precision, recall, and F1 score metrics
- Error analysis with false positive and false negative breakdowns

## 📁 Project Structure

```
energy-fault-prediction/
├── app.py                  # Streamlit dashboard application
├── data/                   # Data storage directory
├── docs/                   # Documentation
├── fault_analysis/         # Fault analysis modules
├── models/                 # Trained ML models
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Core source modules
└── tests/                  # Test modules
```

## 📚 Documentation

For detailed documentation, see:
- [Technical Documentation](docs/technical_documentation.md): Comprehensive technical details
- [API Reference](docs/api_reference.md): API documentation for all modules
- [User Guide](docs/user_guide.md): Guide for using the dashboard

## 🔍 Example Notebooks

The `notebooks/` directory contains example Jupyter notebooks:
- `Train_ml_models.ipynb`: Training ML models for fault detection
- `data_loading_example.py`: Example of loading and processing data
- `fault_detection_rf_xgb.py`: Example of using Random Forest and XGBoost models


## Acknowledgements

- Kenya Meteorological Department for weather patterns data
- IEEE 1547 for grid interconnection standards
- IEC 61724 for PV system performance monitoring guidelines
