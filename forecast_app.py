import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Function to engineer features
def engineer_features(df, target, lags=24):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    for lag in range(1, lags + 1):
        df[f'{target}_lag{lag}'] = df[target].shift(lag)
    
    df[f'{target}_roll_mean'] = df[target].rolling(window=24).mean()
    df[f'{target}_roll_std'] = df[target].rolling(window=24).std()
    
    if target == 'load_demand':
        df['hour_demand'] = df['hour'] * df[f'{target}_lag1']
    
    return df.dropna()

# Rolling forecast function with corrections
def rolling_forecast(model, features, data, steps, historical_means, hourly_weather_averages):
    forecasts = []
    current_data = data.copy()
    target = next((feat.split('_lag')[0] for feat in features if '_lag' in feat), None)
    
    if len(data) < 24:
        raise ValueError("Initial data must contain at least 24 rows.")
    if not target:
        raise ValueError("No lag feature found in features list.")
    
    for step in range(steps):
        # Prepare input features for prediction
        X_future = current_data[features].iloc[-1].values.reshape(1, -1)
        if np.any(np.isnan(X_future)):
            raise ValueError("NaN values detected in input features for prediction.")
        
        pred = model.predict(X_future)[0]
        # Debug first prediction
        if step == 0:
            st.write(f"First prediction: {pred}, Input features: {X_future}")
        
        # Adjust prediction based on historical hourly mean
        new_time = current_data.index[-1] + pd.Timedelta(hours=1)
        hour = new_time.hour
        if hour in historical_means and historical_means[hour] > 0:
            avg_pred = current_data[target].mean()
            if avg_pred > 0:
                pred = pred * (historical_means[hour] / avg_pred)
        
        forecasts.append(pred)
        
        # Create new row for next iteration
        new_row = pd.DataFrame(index=[new_time], columns=current_data.columns)
        
        # Set target for the new row
        new_row[target] = pred
        
        # Populate time-based features
        new_row['hour'] = new_time.hour
        new_row['day_of_week'] = new_time.dayofweek
        new_row['month'] = new_time.month
        
        # Populate lagged values
        last_24 = current_data[target].tail(24).tolist() + [pred]
        for lag in range(1, 25):
            new_row[f'{target}_lag{lag}'] = last_24[-lag]
        
        # Populate rolling statistics
        rolling_window = current_data[target].tail(24).tolist()  # Use only historical/predicted up to current step
        new_row[f'{target}_roll_mean'] = np.mean(rolling_window)
        new_row[f'{target}_roll_std'] = np.std(rolling_window)
        
        # Populate load_demand specific feature
        if target == 'load_demand':
            new_row['hour_demand'] = new_row['hour'] * new_row['load_demand_lag1']
        
        # Populate weather-related features using hourly averages with noise
        for col in ['weather_temperature', 'weather_humidity', 'weather_wind_speed']:
            hour_avg = hourly_weather_averages[new_time.hour][col]
            new_row[col] = hour_avg * (1 + np.random.uniform(-0.1, 0.1))
        
        # Fill any remaining NaN values with column means from current_data
        for col in features:
            if col not in new_row or pd.isna(new_row[col].iloc[0]):
                new_row[col] = current_data[col].mean()
        
        # Append new row and keep last 24 rows
        current_data = pd.concat([current_data, new_row]).tail(24)
    
    return forecasts

# Load dataset and models
@st.cache_data
def load_data_and_models():
    df = pd.read_parquet('notebooks/data/gen_data_testv2.parquet')
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    df = df[df['fault_occurred'] == False].fillna(method='ffill')
    
    df = engineer_features(df, 'load_demand')
    df = engineer_features(df, 'solar_power')
    
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    test_df = df.iloc[train_size + val_size:]
    
    # Load Random Forest models
    rf_demand = joblib.load('models/rf_demand.pkl')
    rf_solar = joblib.load('models/rf_solar.pkl')
    
    # Load XGBoost models
    xgb_demand = XGBRegressor()
    xgb_demand.load_model('models/xgb_demand.json')
    xgb_solar = XGBRegressor()
    xgb_solar.load_model('models/xgb_solar.json')
    
    models = {
        'load_demand': {
            'RF': rf_demand,
            'XGBoost': xgb_demand
        },
        'solar_power': {
            'RF': rf_solar,
            'XGBoost': xgb_solar
        }
    }
    
    return test_df, models

# Define features
demand_features = [
    'hour', 'day_of_week', 'month', 'weather_temperature', 'weather_humidity', 'weather_wind_speed',
    'load_demand_roll_mean', 'load_demand_roll_std', 'hour_demand'
] + [f'load_demand_lag{i}' for i in range(1, 25)]

solar_features = [
    'hour', 'day_of_week', 'month', 'weather_temperature', 'weather_humidity', 'weather_wind_speed',
    'solar_power_roll_mean', 'solar_power_roll_std'
] + [f'solar_power_lag{i}' for i in range(1, 25)]

# Streamlit app
st.title("Energy Forecasting Dashboard")
st.markdown("Explore the performance of Random Forest and XGBoost models for load demand and solar power forecasting.")

# Load data and models
try:
    test_df, models = load_data_and_models()
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# Compute historical hourly means for load demand and solar power
historical_means_load = test_df.groupby('hour')['load_demand'].mean().to_dict()
historical_means_solar = test_df.groupby('hour')['solar_power'].mean().to_dict()

# Compute hourly weather averages
weather_cols = ['weather_temperature', 'weather_humidity', 'weather_wind_speed']
hourly_weather_averages = test_df.groupby('hour')[weather_cols].mean().to_dict(orient='index')

# Sidebar
st.sidebar.header("Options")
target = st.sidebar.selectbox("Select Target", ["load_demand", "solar_power"], index=0)
model_name = st.sidebar.selectbox("Select Model", ["RF", "XGBoost"], index=0)
horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 24, 24)
max_start = test_df.index.max() - pd.Timedelta(hours=horizon)

# Convert Timestamps to datetime.datetime for the slider, ensure 24 hours of prior data
min_time = test_df.index[24].to_pydatetime() if len(test_df) > 24 else test_df.index.min().to_pydatetime()
max_time = max_start.to_pydatetime()
default_time = min_time

# Create the slider with datetime objects
start_time = st.sidebar.slider(
    "Select Starting Time",
    min_value=min_time,
    max_value=max_time,
    value=default_time,
    format="YYYY-MM-DD HH:mm"
)

# Select model and features
model = models[target][model_name]
features = demand_features if target == 'load_demand' else solar_features
historical_means = historical_means_load if target == 'load_demand' else historical_means_solar

# Overall performance
st.header("Overall One-Step Ahead Performance")
X_test = test_df[features]
y_test = test_df[target]
test_pred = model.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, test_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
    "R2": r2_score(y_test, test_pred)
}
st.write("Metrics:", metrics)

# Plot one-step ahead forecast with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=test_df.index[-168:], y=y_test[-168:],
    mode='lines', name='Actual', line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=test_df.index[-168:], y=test_pred[-168:],
    mode='lines', name='Predicted', line=dict(color='orange')
))
fig.update_layout(
    title=f"One-Step Ahead Forecast - {model_name} ({target.replace('_', ' ').title()})",
    xaxis_title="Time",
    yaxis_title="Value",
    legend=dict(x=0, y=1),
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Interactive forecast
st.header("Interactive Multi-Step Forecast")
try:
    start_idx = test_df.index.get_loc(start_time)
    if start_idx < 24:
        st.error("Starting time too early for 24-hour context.")
    else:
        initial_data = test_df.iloc[start_idx - 24:start_idx]
        forecasts = rolling_forecast(model, features, initial_data, horizon, historical_means, hourly_weather_averages)
        forecast_dates = pd.date_range(start=start_time, periods=horizon, freq='H')
        actuals = test_df[target].loc[start_time:forecast_dates[-1]]
        
        # Plot multi-step forecast with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecasts,
            mode='lines', name='Forecast', line=dict(dash='dash', color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=actuals.index, y=actuals,
            mode='lines', name='Actual', line=dict(color='blue')
        ))
        fig.update_layout(
            title=f"{horizon}-Hour Forecast - {model_name} ({target.replace('_', ' ').title()})",
            xaxis_title="Time",
            yaxis_title="Value",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if len(actuals) == horizon:
            forecast_metrics = {
                "MAE": mean_absolute_error(actuals, forecasts),
                "RMSE": np.sqrt(mean_squared_error(actuals, forecasts)),
                "R2": r2_score(actuals, forecasts)
            }
            st.write("Forecast Metrics:", forecast_metrics)
except Exception as e:
    st.error(f"Error generating forecast: {e}")

# Feature importance
st.header("Feature Importance")
if hasattr(model, 'best_estimator_'):
    importances = model.best_estimator_.feature_importances_
else:
    importances = model.feature_importances_
fig = px.bar(
    x=features, y=importances,
    labels={'x': 'Features', 'y': 'Importance'},
    title=f"Feature Importance - {model_name} ({target.replace('_', ' ').title()})"
)
fig.update_layout(
    xaxis_tickangle=45,
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)
