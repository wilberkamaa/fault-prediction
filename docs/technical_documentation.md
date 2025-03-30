# Hybrid Energy System Fault Prediction

## Overview

This codebase provides a comprehensive solution for hybrid energy system fault prediction, consisting of:
- Synthetic data generation for a hybrid energy system (solar, battery, grid, diesel generator)
- Machine learning models for fault detection (Random Forest and XGBoost)
- Interactive Streamlit dashboard for visualization and prediction

The system is designed to detect and predict faults in a hybrid energy system located in Kenya, with components including:
- 1500 kW Solar PV System
- 1 MVA Diesel Generator
- 3 MWh Battery Storage
- 25 kV Grid Connection

## Project Structure

```
energy-fault-prediction/
├── app.py                  # Streamlit dashboard application
├── data/                   # Data storage directory
├── docs/                   # Documentation
├── fault_analysis/         # Fault analysis modules
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── main.py
│   ├── model_training.py
│   └── visualization.py
├── models/                 # Trained ML models
│   ├── feature_list.txt
│   ├── feature_scaler.pkl
│   ├── rf_fault_detection.pkl
│   └── xgboost_fault_detection.json
├── notebooks/              # Jupyter notebooks for analysis
│   ├── Train_ml_models.ipynb
│   ├── data_loading_example.py
│   ├── fault_detection_rf_xgb.py
├── src/                    # Core source modules
│   ├── __init__.py
│   ├── battery_system.py
│   ├── data_generator.py
│   ├── diesel_generator.py
│   ├── fault_injection.py
│   ├── grid_connection.py
│   ├── load_profile.py
│   ├── solar_pv.py
│   ├── validation.py
│   └── weather.py
└── tests/                  # Test modules
```

## Installation & Setup

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.10.0
scikit-learn>=1.0.0
xgboost>=1.5.0
plotly>=5.0.0
joblib>=1.1.0
shap>=0.40.0
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the Streamlit dashboard
streamlit run app.py
```

## System Components

### 1. Data Generation (src/)

The `src` package contains modules for generating synthetic data for the hybrid energy system:

- **weather.py**: Simulates weather conditions (temperature, humidity, etc.)
- **solar_pv.py**: Models solar PV generation based on weather
- **battery_system.py**: Simulates battery storage behavior
- **grid_connection.py**: Models grid connection and reliability
- **diesel_generator.py**: Simulates diesel generator operation
- **load_profile.py**: Generates realistic load profiles
- **fault_injection.py**: Injects realistic faults into the system
- **validation.py**: Validates generated data
- **data_generator.py**: Main class that orchestrates all components

#### Proper Import Structure

The `src` package is structured as a proper Python package. To use it in your code:

```python
# Method 1: Import the entire src package
import src
# Then use: src.data_generator.HybridSystemDataGenerator

# Method 2: Import specific modules
from src import data_generator
# Then use: data_generator.HybridSystemDataGenerator

# Method 3: Import specific classes directly
from src.data_generator import HybridSystemDataGenerator
# Then use: HybridSystemDataGenerator directly
```

#### Example Data Generation

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

### 2. Fault Analysis (fault_analysis/)

The `fault_analysis` package contains modules for analyzing and predicting faults:

- **data_preparation.py**: Prepares data for ML models
- **model_training.py**: Trains and evaluates ML models
- **visualization.py**: Visualizes fault patterns and model results
- **main.py**: Main entry point for fault analysis

### 3. Machine Learning Models (models/)

Two main models are used for fault prediction:

- **Random Forest**: Ensemble learning method using decision trees
- **XGBoost**: Gradient boosting framework optimized for performance

The models are trained to detect various fault types in the hybrid energy system, with features derived from system parameters.

### 4. Streamlit Dashboard (app.py)

The Streamlit dashboard provides an interactive interface for:

- Visualizing system data and fault patterns
- Making real-time fault predictions
- Analyzing feature importance
- Evaluating model performance

The dashboard includes:
- System Overview tab
- Fault Analysis tab
- Feature Importance tab
- Performance Metrics tab

## System Components

### 1. Solar PV System
- Capacity: 1500 kW
- High-efficiency panels (23% base efficiency)
- Modern inverters (85% system efficiency)
- Advanced thermal design (NOCT: 42°C)
- Improved cleaning schedule
- Expected to provide 30-50% of daily load
- **Priority Level: 1 (Highest)** - Maximized usage when available

### 2. Battery Energy Storage
- Capacity: 3 MWh (2 hours of solar capacity)
- Maximum power: 750 kW (50% of solar capacity)
- Modern lithium-ion technology
- High round-trip efficiency (95%)
- Low self-discharge (0.05% per hour)
- Extended cycle life
- Usable capacity: 90% of rated
- Dynamic SOC range (15%-95%)
- Balanced charging/discharging cycles
- **Priority Level: 2** - Used after solar to meet demand or store excess solar

### 3. Grid Connection
- Voltage: 25 kV
- Reliability: 98% (base)
- Seasonal reliability factors
  - Long rains: 95%
  - Short rains: 97%
  - Dry season: 99%
- Bi-directional power flow (import/export)
- Peak hour limitations (30% reduction during peak hours)
- **Priority Level: 3** - Used after solar and battery

### 4. Diesel Generator
- Capacity: 1 MVA
- Fuel tank: 2000 liters
- Minimum load: 40% (improved efficiency)
- Minimum runtime: 2 hours once started
- Modern engine with improved efficiency
- Maintenance interval: 750 hours
- **Priority Level: 4 (Lowest)** - Last resort, only used when other sources insufficient

## Power Dispatch Strategy

The system implements a hierarchical power dispatch strategy to optimize renewable energy usage and minimize operational costs:

1. **Solar PV (First Priority)**
   - Always used when available
   - Excess solar production can charge batteries or be exported to grid

2. **Battery Storage (Second Priority)**
   - Charges during solar surplus
   - Discharges to meet demand when solar is insufficient
   - Maintains dynamic SOC range (15%-95%)
   - Follows daily charge/discharge patterns based on time of day

3. **Grid Connection (Third Priority)**
   - Supplies remaining power needs after solar and battery
   - Reduced usage during peak hours (6 PM - 10 PM)
   - Can accept excess power from solar when batteries are full

4. **Diesel Generator (Last Resort)**
   - Only activates when all other sources are insufficient
   - Requires minimum load (40%) for efficient operation
   - Runs for minimum of 2 consecutive hours once started
   - Automatically activates during grid outages with significant load

## Data Structure

### Column Naming Convention
All columns follow a consistent naming pattern:
- Weather data: `weather_*` (e.g., `weather_temperature`, `weather_cloud_cover`)
- Solar PV: `solar_*` (e.g., `solar_power`, `solar_irradiance`)
- Grid: `grid_*` (e.g., `grid_power`, `grid_voltage`)
- Battery: `battery_*` (e.g., `battery_power`, `battery_soc`)
- Generator: `generator_*` (e.g., `generator_power`, `generator_fuel_rate`)
- Load: `load_*` (e.g., `load_demand`, `load_reactive_power`)
- Faults: `fault_*` (e.g., `fault_type`, `fault_severity`)

### Key Parameters
1. **Weather Parameters**
   - `weather_temperature`: Ambient temperature (°C)
   - `weather_cloud_cover`: Cloud cover ratio (0-1)
   - `weather_humidity`: Relative humidity (%)
   - `weather_wind_speed`: Wind speed (m/s)

2. **Solar PV Parameters**
   - `solar_irradiance`: Solar irradiance (W/m²)
   - `solar_cell_temp`: PV cell temperature (°C)
   - `solar_power`: Power output (kW)

3. **Grid Parameters**
   - `grid_voltage`: Grid voltage (V)
   - `grid_frequency`: Grid frequency (Hz)
   - `grid_power`: Power exchange with grid (kW, positive=import)
   - `grid_available`: Grid availability status (bool)
   - `grid_power_quality`: Power quality metric (0-1)

4. **Battery Parameters**
   - `battery_power`: Power exchange (kW, positive=discharge)
   - `battery_soc`: State of charge (%)
   - `battery_voltage`: Battery voltage (V)
   - `battery_temperature`: Battery temperature (°C)

5. **Generator Parameters**
   - `generator_power`: Power output (kW)
   - `generator_fuel_rate`: Fuel consumption rate (L/h)
   - `generator_temperature`: Engine temperature (°C)
   - `generator_runtime`: Cumulative runtime (h)

6. **Load Parameters**
   - `load_demand`: Active power demand (kW)
   - `load_reactive_power`: Reactive power (kVAR)
   - `load_power_factor`: Power factor (0-1)

7. **Fault Parameters**
   - `fault_type`: Type of fault (string)
   - `fault_severity`: Fault severity (0-1)
   - `fault_duration`: Duration in hours (float)
   - `fault_occurred`: Binary indicator of fault (0/1)

## Machine Learning Pipeline

### Data Preparation
```python
# Load dataset
df = pd.read_parquet('data/hybrid_system_data.parquet')

# Define features and target
features = [col for col in df.columns if col not in ['fault_occurred', 'fault_type']]
target = 'fault_occurred'

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Model Training
```python
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
```

### Model Evaluation
```python
# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate
print(classification_report(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_xgb))
```

## Streamlit Dashboard

The Streamlit dashboard (`app.py`) provides an interactive interface for exploring the data and making predictions. Key features include:

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

### Running the Dashboard
```bash
streamlit run app.py
```

## Known Limitations and Future Improvements

1. **Current Limitations**
   - Limited error handling in data generation
   - No model monitoring or retraining capabilities
   - Missing data validation in some components
   - Limited test coverage
   - Hardcoded configurations

2. **Planned Improvements**
   - Implement automated model retraining
   - Add more sophisticated fault models
   - Enhance error handling and logging
   - Implement configuration management
   - Add real-time data processing capabilities
   - Extend test coverage

## References

1. Kenya Meteorological Department - Weather patterns
2. IEEE 1547 - Grid interconnection standards
3. IEC 61724 - PV system performance monitoring
4. Battery storage system standards (IEC 62619)
5. Scikit-learn documentation - ML pipeline
6. XGBoost documentation - Gradient boosting
7. Streamlit documentation - Interactive dashboards
