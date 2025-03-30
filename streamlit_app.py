import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import requests
import xgboost as xgb
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Patch numpy to fix the np.int deprecation issue
if not hasattr(np, 'int'):
    np.int = int

# Now import shap after the patch
import shap

# Set page configuration
st.set_page_config(
    page_title="Energy System Fault Prediction Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f8f9fa;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .fault-alert {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 5px solid #dc3545;
    }
    .no-fault {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Define constants and file paths
MODEL_FILES = {
    "rf_model": "rf_fault_detection.pkl",
    "xgb_model": "xgboost_fault_detection.json",
}

# Helper function to get the absolute path to a file in the repository
def get_repo_path(relative_path):
    """Get the absolute path to a file in the repository, works both locally and on Streamlit Cloud."""
    # Base directory is the directory where the script is located
    base_dir = Path(__file__).parent.absolute()
    return base_dir / relative_path

@st.cache_resource
def load_models():
    """Load ML models, scaler, and feature list from the repository."""
    models_dir = get_repo_path("models")
    models = {}
    
    # Load models
    for name, filename in MODEL_FILES.items():
        full_path = models_dir / filename
        if not os.path.exists(full_path):
            st.error(f"❌ Model file not found: {full_path}")
            continue
            
        try:
            if "xgb" in name:
                xgb_model = xgb.Booster()
                xgb_model.load_model(str(full_path))
                models[name] = xgb_model
                st.write(f"✅ Loaded XGBoost model from {filename}")
            else:
                models[name] = joblib.load(full_path)
                st.write(f"✅ Loaded {filename}")
        except Exception as e:
            st.write(f"❌ Error loading {filename}: {e}")
    
    # Load scaler
    try:
        scaler_path = models_dir / "feature_scaler.pkl"
        scaler = joblib.load(scaler_path)
        st.write(f"✅ Loaded feature scaler")
    except Exception as e:
        st.write(f"❌ Error loading feature scaler: {e}")
        return None
    
    # Load feature list
    try:
        feature_list_path = models_dir / "feature_list.txt"
        with open(feature_list_path, "r") as f:
            feature_list = [line.strip() for line in f.readlines()]
        st.write(f"✅ Loaded feature list with {len(feature_list)} features")
    except Exception as e:
        st.write(f"❌ Error loading feature list: {e}")
        return None
    
    # Add scaler and feature list to the returned dictionary
    models["scaler"] = scaler
    models["feature_list"] = feature_list
    st.write("🔹 Available models:", models.keys())
    return models

@st.cache_data
def load_sample_data():
    """Load sample data from the repository."""
    # Try multiple possible data locations
    possible_paths = [
        get_repo_path("data/gen_data_test.parquet"),
        get_repo_path("notebooks/data/gen_data_test.parquet"),
        get_repo_path("data/gen_data_test.csv"),
        get_repo_path("notebooks/data/gen_data_test.csv")
    ]
    
    # Try to load from each path
    for data_path in possible_paths:
        if data_path.exists():
            st.write(f"✅ Loading data from {data_path}")
            
            # Load based on file extension
            if str(data_path).endswith('.parquet'):
                df = pd.read_parquet(data_path)
            elif str(data_path).endswith('.csv'):
                df = pd.read_csv(data_path)
                # If the index is a datetime, parse it
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
            
            return df
    
    # If no data found
    st.error("Sample data file not found. Please check the repository structure.")
    return None

# Make predictions
def predict_fault(data, model, scaler, feature_list, threshold=0.5):
    """
    Make fault predictions on data.
    
    Args:
        data: DataFrame with required features
        model: Trained model
        scaler: Fitted scaler
        feature_list: List of features used for training
        threshold: Probability threshold for classification
        
    Returns:
        Predictions and probabilities
    """
    # Ensure all required features are present
    missing_features = set(feature_list) - set(data.columns)
    if missing_features:
        st.warning(f"Missing required features: {missing_features}")
        # Fill missing features with zeros
        for feature in missing_features:
            data[feature] = 0
    
    # Extract features in the correct order
    X = data[feature_list]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
     # Make predictions based on the selected model
    if isinstance(model, xgb.Booster):  # XGBoost model
        # XGBoost requires DMatrix format for prediction
        dmatrix = xgb.DMatrix(X_scaled)
        probabilities = model.predict(dmatrix)  # probabilities are already in the correct format
    else:  # RandomForest model
        probabilities = model.predict_proba(X_scaled)[:, 1]  # probabilities for class 1
    
    #probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

# Function to generate system status metrics
def get_system_metrics(df):
    """Extract key system metrics from the dataframe."""
    latest_data = df.iloc[-1]
    
    metrics = {
        "Battery SoC (%)": latest_data.get("battery_soc", 0) * 100,
        "Solar Power (kW)": latest_data.get("solar_power", 0),
        "Grid Power (kW)": latest_data.get("grid_power", 0),
        "Generator Power (kW)": latest_data.get("generator_power", 0),
        "Load Demand (kW)": latest_data.get("load_demand", 0),
        "Battery Temperature (°C)": latest_data.get("battery_temperature", 0),
        "Grid Voltage (V)": latest_data.get("grid_voltage", 0),
        "Grid Available": latest_data.get("grid_available", False)
    }
    
    return metrics

# Function to create time series plot
def create_time_series_plot(df, columns, title, y_label):
    """Create a time series plot for selected columns."""
    fig = go.Figure()
    
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Main application
def main():
    # Load models and data
    models_dict = load_models()
    sample_data = load_sample_data()
    
    # Header
    st.markdown('<h1 class="main-header">Energy System Fault Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Add DataFrame info in a dropdown section
    if sample_data is not None:
        with st.expander("📊 View Dataset Information"):
            st.subheader("Dataset Overview")
            st.write(f"**Shape**: {sample_data.shape[0]} rows × {sample_data.shape[1]} columns")
            
            # Display basic info
            st.subheader("Data Types and Non-Null Values")
            buffer = io.StringIO()
            sample_data.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
            # Display descriptive statistics
            st.subheader("Descriptive Statistics")
            st.write(sample_data.describe())
            
            # Display first few rows
            st.subheader("Sample Data (First 5 Rows)")
            st.dataframe(sample_data.head())
    
    # Sidebar
    #st.sidebar.image("https://img.icons8.com/fluency/96/000000/energy.png", width=80)
    st.sidebar.title("Controls & Settings")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Random Forest", "XGBoost"],
        index=0
    )
    
    model_key = "rf_model" if selected_model == "Random Forest" else "xgb_model"
    
    # Prediction threshold
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for fault classification"
    )
    
    # Time range selection for sample data visualization
    if sample_data is not None:
        # Convert index to datetime if it's not already
        if not isinstance(sample_data.index, pd.DatetimeIndex):
            if 'datetime' in sample_data.columns:
                sample_data['datetime'] = pd.to_datetime(sample_data['datetime'])
                sample_data.set_index('datetime', inplace=True)
            else:
                # Create a synthetic datetime index
                sample_data.index = pd.date_range(start='2023-01-01', periods=len(sample_data), freq='H')
        
        min_date = sample_data.index.min().date()
        max_date = sample_data.index.max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, min(min_date + timedelta(days=7), max_date)),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = sample_data.loc[start_date:end_date].copy()
        else:
            filtered_data = sample_data.copy()
    else:
        filtered_data = None
    
    # Main content
    if filtered_data is not None and not filtered_data.empty:
        try:
            # Make predictions on the filtered data
            predictions, probabilities = predict_fault(
                filtered_data,
                models_dict[model_key],
                models_dict["scaler"],
                models_dict["feature_list"],
                threshold
            )
            
            # Add predictions to the dataframe
            filtered_data['predicted_fault'] = predictions
            filtered_data['fault_probability'] = probabilities
            
            # Dashboard layout with tabs
            tab1, tab2, tab3, tab4 = st.tabs(["System Overview", "Fault Analysis", "Feature Importance", "Performance Metrics"])
            
            # Tab 1: System Overview
            with tab1:
                # Power distribution
                st.markdown('<h2 class="sub-header">Power Distribution</h2>', unsafe_allow_html=True)
                
                power_cols = ['solar_power', 'battery_power', 'grid_power', 'generator_power', 'load_demand']
                power_fig = create_time_series_plot(
                    filtered_data,
                    power_cols,
                    "Power Distribution Over Time",
                    "Power (kW)"
                )
                st.plotly_chart(power_fig, use_container_width=True)
                
                # Weather conditions
                st.markdown('<h2 class="sub-header">Weather Conditions</h2>', unsafe_allow_html=True)
                
                weather_cols = ['weather_temperature', 'weather_humidity', 'weather_wind_speed', 'solar_irradiance']
                weather_fig = create_time_series_plot(
                    filtered_data,
                    weather_cols,
                    "Weather Conditions Over Time",
                    "Value"
                )
                st.plotly_chart(weather_fig, use_container_width=True)
            
            # Tab 2: Fault Analysis
            with tab2:
                st.markdown('<h2 class="sub-header">Fault Prediction Analysis</h2>', unsafe_allow_html=True)
                
                # Current fault status
                latest_prediction = predictions[-1]
                latest_probability = probabilities[-1]
                
                if latest_prediction == 1:
                    st.markdown(
                        f'<div class="fault-alert"><h3>⚠️ Fault Detected!</h3>'
                        f'<p>Fault probability: {latest_probability:.2%}</p></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="no-fault"><h3>✅ System Normal</h3>'
                        f'<p>Fault probability: {latest_probability:.2%}</p></div>',
                        unsafe_allow_html=True
                    )
                
                # Fault probability over time
                st.subheader("Fault Probability Over Time")
                
                fig = px.line(
                    filtered_data,
                    x=filtered_data.index,
                    y='fault_probability',
                    title="Fault Probability Trend"
                )
                
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold})",
                    annotation_position="bottom right"
                )
                
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Fault Probability",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Fault distribution
                st.subheader("Predicted Fault Distribution")
                
                if not filtered_data.empty and 'predicted_fault' in filtered_data.columns:
                    fault_counts = filtered_data['predicted_fault'].value_counts()
                    
                    if not fault_counts.empty:
                        # Create a mapping dictionary for the labels
                        label_map = {0: "No Fault", 1: "Fault"}
                        
                        # Create the pie chart
                        fig = px.pie(
                            values=fault_counts.values,
                            names=[label_map.get(idx, f"Unknown ({idx})") for idx in fault_counts.index],
                            title="Fault Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No fault data available for visualization.")
                else:
                    st.info("No data available for fault distribution visualization.")
                
                # High-risk periods
                st.subheader("High-Risk Periods")
                
                if not filtered_data.empty and 'fault_probability' in filtered_data.columns:
                    high_risk = filtered_data[filtered_data['fault_probability'] > threshold].copy()
                    
                    if not high_risk.empty:
                        # Group consecutive high-risk periods
                        high_risk['date'] = high_risk.index.date
                        high_risk['hour'] = high_risk.index.hour
                        
                        # Display as a heatmap by hour and date
                        try:
                            pivot = pd.pivot_table(
                                high_risk,
                                values='fault_probability',
                                index='date',
                                columns='hour',
                                aggfunc='mean'
                            )
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
                            plt.title("High-Risk Periods by Hour")
                            plt.xlabel("Hour of Day")
                            plt.ylabel("Date")
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error creating high-risk heatmap: {str(e)}")
                    else:
                        st.info("No high-risk periods detected in the selected timeframe.")
                else:
                    st.info("No data available for high-risk analysis.")
            
            # Tab 3: Feature Importance
            with tab3:
                st.markdown('<h2 class="sub-header">Feature Importance</h2>', unsafe_allow_html=True)
                
                # Get feature importance based on model type
                if selected_model == "Random Forest":
                    # Random Forest feature importance
                    if hasattr(models_dict[model_key], 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'Feature': models_dict["feature_list"],
                            'Importance': models_dict[model_key].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Display top 15 features
                        st.subheader(f"Feature Importance for {selected_model}")
                        
                        fig = px.bar(
                            feature_importance.head(15),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title=f"Top 15 Feature Importances from {selected_model}"
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional visualizations for feature importance
                        st.subheader("Advanced Feature Analysis")
                        
                        # For the top feature, create analysis visualizations
                        if len(feature_importance) > 0:
                            top_feature_name = feature_importance.iloc[0]['Feature']
                            
                            st.subheader(f"Analysis for Top Feature: {top_feature_name}")
                            
                            # Create tabs for different visualizations
                            viz_tab1, viz_tab2 = st.tabs(["Distribution", "Relationship with Fault Probability"])
                            
                            with viz_tab1:
                                # Create a histogram of the top feature
                                fig = px.histogram(
                                    filtered_data, 
                                    x=top_feature_name,
                                    title=f'Distribution of {top_feature_name}',
                                    nbins=30,
                                    color_discrete_sequence=['#3498db']
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with viz_tab2:
                                # Create a scatter plot of the top feature vs fault probability
                                fig = px.scatter(
                                    filtered_data,
                                    x=top_feature_name,
                                    y='fault_probability',
                                    title=f'Relationship between {top_feature_name} and Fault Probability',
                                    opacity=0.6,
                                    trendline="lowess",
                                    color='predicted_fault',
                                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance for top 5 features
                            st.subheader("Top 5 Features Analysis")
                            
                            # Get top 5 features
                            top_features = feature_importance.head(5)['Feature'].tolist()
                            
                            # Create correlation heatmap for top features
                            top_features_df = filtered_data[top_features + ['fault_probability']]
                            corr = top_features_df.corr()
                            
                            fig = px.imshow(
                                corr,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                title="Correlation Between Top Features and Fault Probability"
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Feature importance not available for the Random Forest model.")
                
                elif selected_model == "XGBoost":
                    # XGBoost feature importance using native methods
                    st.subheader(f"Feature Importance for {selected_model}")
                    
                    try:
                        # Create a DataFrame with the features
                        X = filtered_data[models_dict["feature_list"]]
                        
                        # Convert to DMatrix for XGBoost
                        dmatrix = xgb.DMatrix(X)
                        
                        # Get feature importance from XGBoost model
                        importance_type = 'gain'
                        feature_importance = models_dict[model_key].get_score(importance_type=importance_type, fmap='')
                        
                        # Map feature indices to actual feature names
                        feature_names = models_dict["feature_list"]
                        feature_map = {}
                        
                        # Create a mapping from f0, f1, etc. to actual feature names
                        for i, name in enumerate(feature_names):
                            feature_map[f'f{i}'] = name
                        
                        # Convert to DataFrame with proper feature names
                        importance_df = pd.DataFrame({
                            'Feature': [feature_map.get(f, f) for f in feature_importance.keys()],
                            'Importance': list(feature_importance.values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Display top 15 features
                        fig = px.bar(
                            importance_df.head(15),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title=f"Top 15 Feature Importances from {selected_model} (based on {importance_type})"
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional visualizations for feature importance
                        st.subheader("Advanced Feature Analysis")
                        
                        # For the top feature, create analysis visualizations
                        if len(importance_df) > 0:
                            top_feature_name = importance_df.iloc[0]['Feature']
                            
                            st.subheader(f"Analysis for Top Feature: {top_feature_name}")
                            
                            # Create tabs for different visualizations
                            viz_tab1, viz_tab2 = st.tabs(["Distribution", "Relationship with Fault Probability"])
                            
                            with viz_tab1:
                                # Create a histogram of the top feature
                                fig = px.histogram(
                                    filtered_data, 
                                    x=top_feature_name,
                                    title=f'Distribution of {top_feature_name}',
                                    nbins=30,
                                    color_discrete_sequence=['#3498db']
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with viz_tab2:
                                # Create a scatter plot of the top feature vs fault probability
                                fig = px.scatter(
                                    filtered_data,
                                    x=top_feature_name,
                                    y='fault_probability',
                                    title=f'Relationship between {top_feature_name} and Fault Probability',
                                    opacity=0.6,
                                    trendline="lowess",
                                    color='predicted_fault',
                                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance for top 5 features
                            st.subheader("Top 5 Features Analysis")
                            
                            # Get top 5 features
                            top_features = importance_df.head(5)['Feature'].tolist()
                            
                            # Create correlation heatmap for top features
                            top_features_df = filtered_data[top_features + ['fault_probability']]
                            corr = top_features_df.corr()
                            
                            fig = px.imshow(
                                corr,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                title="Correlation Between Top Features and Fault Probability"
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating feature importance visualizations: {str(e)}")
                
                # Feature correlation with fault probability (for both models)
                st.subheader("Feature Correlation with Fault Probability")
                
                # Calculate correlations
                correlations = filtered_data[models_dict["feature_list"]].corrwith(filtered_data['fault_probability'])
                top_correlations = correlations.abs().sort_values(ascending=False).head(15)
                
                corr_df = pd.DataFrame({
                    'Feature': top_correlations.index,
                    'Correlation': correlations[top_correlations.index]
                })
                
                fig = px.bar(
                    corr_df,
                    x='Correlation',
                    y='Feature',
                    orientation='h',
                    color='Correlation',
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    title="Top 15 Feature Correlations with Fault Probability"
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Performance Metrics
            with tab4:
                st.markdown('<h2 class="sub-header">Performance Metrics</h2>', unsafe_allow_html=True)
                
                # Create a simulated ground truth for demonstration purposes
                # In a real application, you would use actual labeled data
                st.info("Note: For demonstration purposes, we're using a simulated ground truth based on probability thresholds.")
                
                # Create simulated ground truth with different thresholds
                threshold_slider = st.slider(
                    "Adjust Ground Truth Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,  # Higher threshold for ground truth
                    step=0.05,
                    help="Adjust this to simulate different ground truth scenarios"
                )
                
                # Simulate ground truth (in a real app, you would use actual labeled data)
                filtered_data['simulated_truth'] = (filtered_data['fault_probability'] >= threshold_slider).astype(int)
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                
                cm = confusion_matrix(filtered_data['simulated_truth'], filtered_data['predicted_fault'])
                
                # Create labels for the confusion matrix
                labels = ['No Fault', 'Fault']
                
                # Create a more informative confusion matrix
                cm_fig = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels,
                    y=labels,
                    title="Confusion Matrix"
                )
                cm_fig.update_layout(height=400)
                st.plotly_chart(cm_fig, use_container_width=True)
                
                # Explain confusion matrix
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **True Negatives (TN)**: {cm[0][0]}  
                    No fault correctly predicted as no fault
                    """)
                    st.markdown(f"""
                    **False Positives (FP)**: {cm[0][1]}  
                    No fault incorrectly predicted as fault
                    """)
                with col2:
                    st.markdown(f"""
                    **False Negatives (FN)**: {cm[1][0]}  
                    Fault incorrectly predicted as no fault
                    """)
                    st.markdown(f"""
                    **True Positives (TP)**: {cm[1][1]}  
                    Fault correctly predicted as fault
                    """)
                
                # Precision, recall, and F1 score
                st.subheader("Precision, Recall, and F1 Score")
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    filtered_data['simulated_truth'], 
                    filtered_data['predicted_fault'], 
                    average='binary',
                    zero_division=0
                )
                accuracy = accuracy_score(filtered_data['simulated_truth'], filtered_data['predicted_fault'])
                
                # Create a metrics dataframe for better visualization
                metrics_df = pd.DataFrame({
                    "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
                    "Value": [precision, recall, f1, accuracy],
                    "Description": [
                        "Ability of the model to avoid false positives",
                        "Ability of the model to find all faults",
                        "Harmonic mean of precision and recall",
                        "Overall correctness of predictions"
                    ]
                })
                
                # Display metrics as a styled table
                st.dataframe(metrics_df, use_container_width=True)
                
                # Error analysis
                st.subheader("Error Analysis")
                
                # Identify errors
                filtered_data['error_type'] = 'Correct'
                filtered_data.loc[(filtered_data['simulated_truth'] == 0) & (filtered_data['predicted_fault'] == 1), 'error_type'] = 'False Positive'
                filtered_data.loc[(filtered_data['simulated_truth'] == 1) & (filtered_data['predicted_fault'] == 0), 'error_type'] = 'False Negative'
                
                errors = filtered_data[filtered_data['error_type'] != 'Correct']
                
                if not errors.empty:
                    # Error statistics
                    error_count = len(errors)
                    error_rate = error_count / len(filtered_data)
                    
                    st.write(f"Total errors: {error_count} ({error_rate:.2%} of data)")
                    
                    # Error distribution
                    error_dist = errors['error_type'].value_counts().reset_index()
                    error_dist.columns = ['Error Type', 'Count']
                    
                    fig = px.pie(
                        error_dist,
                        values='Count',
                        names='Error Type',
                        title="Distribution of Error Types",
                        color_discrete_sequence=['#e74c3c', '#3498db']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show examples of errors
                    error_tabs = st.tabs(["False Positives", "False Negatives"])
                    
                    with error_tabs[0]:  # False Positives
                        fp_errors = filtered_data[filtered_data['error_type'] == 'False Positive']
                        if not fp_errors.empty:
                            st.write(f"False Positives: {len(fp_errors)} instances")
                            st.write("Sample of False Positive errors (predicted fault when there was none):")
                            st.dataframe(fp_errors.head(5)[['fault_probability'] + models_dict["feature_list"][:5]])
                            
                            # Feature distribution in false positives
                            if len(fp_errors) >= 5:  # Only show if we have enough data
                                st.subheader("Feature Distribution in False Positives")
                                fp_feature = st.selectbox("Select feature to analyze in False Positives:", models_dict["feature_list"][:10])
                                
                                fig = px.histogram(
                                    filtered_data,
                                    x=fp_feature,
                                    color='error_type',
                                    barmode='overlay',
                                    opacity=0.7,
                                    color_discrete_map={
                                        'Correct': '#2ecc71',
                                        'False Positive': '#e74c3c',
                                        'False Negative': '#3498db'
                                    },
                                    title=f"Distribution of {fp_feature} by Prediction Result"
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No False Positive errors found.")
                    
                    with error_tabs[1]:  # False Negatives
                        fn_errors = filtered_data[filtered_data['error_type'] == 'False Negative']
                        if not fn_errors.empty:
                            st.write(f"False Negatives: {len(fn_errors)} instances")
                            st.write("Sample of False Negative errors (missed actual faults):")
                            st.dataframe(fn_errors.head(5)[['fault_probability'] + models_dict["feature_list"][:5]])
                            
                            # Feature distribution in false negatives
                            if len(fn_errors) >= 5:  # Only show if we have enough data
                                st.subheader("Feature Distribution in False Negatives")
                                fn_feature = st.selectbox("Select feature to analyze in False Negatives:", models_dict["feature_list"][:10])
                                
                                fig = px.histogram(
                                    filtered_data,
                                    x=fn_feature,
                                    color='error_type',
                                    barmode='overlay',
                                    opacity=0.7,
                                    color_discrete_map={
                                        'Correct': '#2ecc71',
                                        'False Positive': '#e74c3c',
                                        'False Negative': '#3498db'
                                    },
                                    title=f"Distribution of {fn_feature} by Prediction Result"
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No False Negative errors found.")
                else:
                    st.success("No errors detected with the current thresholds.")
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
    else:
        st.error("No data available for visualization. Please check if the sample data file exists.")

if __name__ == "__main__":
    main()
