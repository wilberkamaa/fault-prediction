"""
Visualization module for fault analysis in hybrid energy systems.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_fault_timeline(df):
    """
    Plot timeline of fault occurrences.
    
    Args:
        df: DataFrame with fault data
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Extract fault data
    fault_data = df[df['fault_type'] != 'NO_FAULT'].copy()
    fault_data['date'] = fault_data.index.date
    
    # Count faults by date and type
    fault_counts = fault_data.groupby(['date', 'fault_type']).size().reset_index(name='count')
    
    # Plot fault timeline
    plt.figure(figsize=(15, 8))
    for fault_type in fault_counts['fault_type'].unique():
        subset = fault_counts[fault_counts['fault_type'] == fault_type]
        plt.plot(subset['date'], subset['count'], 'o-', label=fault_type)
    
    plt.title('Fault Occurrences Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Fault Hours', fontsize=12)
    plt.legend(title='Fault Type')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/fault_timeline.png', dpi=300, bbox_inches='tight')
    
    # Create interactive version with Plotly
    fig = px.line(fault_counts, x='date', y='count', color='fault_type',
                 title='Fault Occurrences Over Time',
                 labels={'date': 'Date', 'count': 'Number of Fault Hours', 'fault_type': 'Fault Type'})
    fig.update_layout(legend_title_text='Fault Type')
    fig.write_html('output/fault_timeline_interactive.html')

def plot_fault_heatmap(df):
    """
    Create heatmap of fault occurrences by hour and day of week.
    
    Args:
        df: DataFrame with fault data
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Extract fault data
    fault_data = df[df['fault_type'] != 'NO_FAULT'].copy()
    fault_data['hour'] = fault_data.index.hour
    fault_data['day_of_week'] = fault_data.index.dayofweek
    
    # Count faults by hour and day of week
    fault_counts = fault_data.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(fault_counts, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Fault Count'})
    plt.title('Fault Occurrences by Hour and Day of Week', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
    plt.tight_layout()
    plt.savefig('output/fault_heatmap.png', dpi=300, bbox_inches='tight')
    
    # Create separate heatmaps for each fault type
    for fault_type in fault_data['fault_type'].unique():
        subset = fault_data[fault_data['fault_type'] == fault_type]
        counts = subset.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(counts, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Fault Count'})
        plt.title(f'{fault_type} Occurrences by Hour and Day of Week', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'output/fault_heatmap_{fault_type}.png', dpi=300, bbox_inches='tight')

def plot_system_parameters_during_faults(df, fault_type=None):
    """
    Plot system parameters before, during, and after fault occurrences.
    
    Args:
        df: DataFrame with system and fault data
        fault_type: Specific fault type to analyze, or None for all faults
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Extract fault events (transitions from no fault to fault)
    if fault_type:
        fault_mask = df['fault_type'] == fault_type
    else:
        fault_mask = df['fault_type'] != 'NO_FAULT'
    
    # Find fault start times
    fault_starts = []
    for i in range(1, len(df)):
        if not fault_mask.iloc[i-1] and fault_mask.iloc[i]:
            fault_starts.append(i)
    
    if not fault_starts:
        print(f"No fault events found for type: {fault_type}")
        return
    
    # Define parameters to plot for each fault type
    param_groups = {
        'LINE_SHORT_CIRCUIT': ['grid_voltage', 'grid_power', 'battery_power'],
        'INVERTER_IGBT_FAILURE': ['solar_power', 'solar_cell_temp', 'solar_efficiency'],
        'GENERATOR_FIELD_FAILURE': ['generator_power', 'generator_temperature', 'generator_fuel_level'],
        'BATTERY_OVERDISCHARGE': ['battery_power', 'battery_soc', 'battery_temperature'],
        'default': ['solar_power', 'battery_power', 'grid_power', 'generator_power', 'load_demand']
    }
    
    # Get parameters to plot based on fault type
    if fault_type and fault_type in param_groups:
        params = param_groups[fault_type]
    else:
        params = param_groups['default']
    
    # Plot parameters around fault events
    window_size = 24  # Hours before and after fault
    
    for fault_idx in fault_starts[:5]:  # Limit to first 5 faults for clarity
        if fault_idx < window_size or fault_idx + window_size >= len(df):
            continue
        
        # Extract window around fault
        start_idx = fault_idx - window_size
        end_idx = fault_idx + window_size
        window_data = df.iloc[start_idx:end_idx]
        
        # Create plot
        fig, axes = plt.subplots(len(params), 1, figsize=(15, 4*len(params)), sharex=True)
        fault_time = df.index[fault_idx]
        
        for i, param in enumerate(params):
            if param in window_data.columns:
                axes[i].plot(window_data.index, window_data[param])
                axes[i].set_ylabel(param)
                axes[i].grid(alpha=0.3)
                
                # Mark fault start
                axes[i].axvline(x=fault_time, color='r', linestyle='--', label='Fault Start')
                
                # Add fault duration shading if available
                if 'fault_duration' in df.columns:
                    duration = df['fault_duration'].iloc[fault_idx]
                    if duration > 0:
                        end_time = fault_time + pd.Timedelta(hours=duration)
                        axes[i].axvspan(fault_time, end_time, alpha=0.2, color='red')
        
        # Add fault details to title
        fault_type_str = df['fault_type'].iloc[fault_idx]
        severity = df['fault_severity'].iloc[fault_idx] if 'fault_severity' in df.columns else 'N/A'
        
        plt.suptitle(f'System Parameters Around Fault Event\nType: {fault_type_str}, Time: {fault_time}, Severity: {severity}', 
                    fontsize=16)
        plt.xlabel('Time')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save plot
        plt.savefig(f'output/fault_event_{fault_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create interactive version with Plotly for the first fault
    if fault_starts:
        fault_idx = fault_starts[0]
        if fault_idx >= window_size and fault_idx + window_size < len(df):
            # Extract window around fault
            start_idx = fault_idx - window_size
            end_idx = fault_idx + window_size
            window_data = df.iloc[start_idx:end_idx]
            
            # Create interactive plot
            fig = make_subplots(rows=len(params), cols=1, 
                               shared_xaxes=True, 
                               subplot_titles=params)
            
            fault_time = df.index[fault_idx]
            
            for i, param in enumerate(params):
                if param in window_data.columns:
                    fig.add_trace(
                        go.Scatter(x=window_data.index, y=window_data[param], name=param),
                        row=i+1, col=1
                    )
                    
                    # Add fault start line
                    fig.add_vline(x=fault_time, line_dash="dash", line_color="red",
                                 annotation_text="Fault Start", 
                                 row=i+1, col=1)
            
            # Add fault details to title
            fault_type_str = df['fault_type'].iloc[fault_idx]
            severity = df['fault_severity'].iloc[fault_idx] if 'fault_severity' in df.columns else 'N/A'
            
            fig.update_layout(
                title_text=f'System Parameters Around Fault Event<br>Type: {fault_type_str}, Time: {fault_time}, Severity: {severity}',
                height=300*len(params),
                width=1000
            )
            
            fig.write_html('output/fault_event_interactive.html')

def plot_feature_correlations(df, fault_type=None):
    """
    Plot correlation heatmap between system parameters and fault occurrence.
    
    Args:
        df: DataFrame with system and fault data
        fault_type: Specific fault type to analyze, or None for all faults
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create binary fault indicator
    if fault_type:
        df['fault_indicator'] = (df['fault_type'] == fault_type).astype(int)
    else:
        df['fault_indicator'] = (df['fault_type'] != 'NO_FAULT').astype(int)
    
    # Select system parameters
    system_params = [col for col in df.columns if col.startswith(('solar_', 'battery_', 'grid_', 'generator_', 'load_', 'weather_'))]
    
    # Add fault indicator
    columns_to_correlate = system_params + ['fault_indicator']
    
    # Calculate correlation matrix
    corr_matrix = df[columns_to_correlate].corr()
    
    # Sort by correlation with fault indicator
    fault_correlations = corr_matrix['fault_indicator'].drop('fault_indicator').sort_values(ascending=False)
    
    # Plot top correlations
    plt.figure(figsize=(12, 8))
    top_n = 20
    top_corr = fault_correlations.abs().nlargest(top_n)
    
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f'Top {top_n} Parameters Correlated with Fault Occurrence', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/fault_correlations.png', dpi=300, bbox_inches='tight')
    
    # Plot correlation heatmap for top correlated features
    plt.figure(figsize=(14, 12))
    top_features = list(top_corr.index) + ['fault_indicator']
    sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Top Features with Fault Indicator', fontsize=16)
    plt.tight_layout()
    plt.savefig('output/fault_correlation_heatmap.png', dpi=300, bbox_inches='tight')

def plot_feature_distributions(df, features, fault_type=None):
    """
    Plot distribution of features for fault vs. no-fault conditions.
    
    Args:
        df: DataFrame with system and fault data
        features: List of features to plot
        fault_type: Specific fault type to analyze, or None for all faults
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create fault mask
    if fault_type:
        fault_mask = df['fault_type'] == fault_type
    else:
        fault_mask = df['fault_type'] != 'NO_FAULT'
    
    # Create a new column for plotting
    df['condition'] = 'No Fault'
    df.loc[fault_mask, 'condition'] = 'Fault'
    
    # Plot distributions
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(12, 6))
            
            sns.histplot(data=df, x=feature, hue='condition', kde=True, 
                        element='step', stat='density', common_norm=False)
            
            plt.title(f'Distribution of {feature} During Fault vs. No-Fault Conditions', fontsize=16)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'output/feature_distribution_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create a multi-panel plot for the top features
    if len(features) > 0:
        n_cols = min(3, len(features))
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i, feature in enumerate(features):
            if i < len(axes) and feature in df.columns:
                sns.histplot(data=df, x=feature, hue='condition', kde=True, 
                           element='step', stat='density', common_norm=False, ax=axes[i])
                
                axes[i].set_title(feature)
                axes[i].grid(alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Feature Distributions During Fault vs. No-Fault Conditions', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('output/feature_distributions_panel.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_dimensionality_reduction(X, y, method='tsne'):
    """
    Plot dimensionality reduction visualization of the feature space.
    
    Args:
        X: Feature matrix
        y: Target vector (fault indicator)
        method: 'tsne' or 'pca'
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        model = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization of Fault Prediction Feature Space'
    else:  # pca
        model = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Fault Prediction Feature Space'
    
    # Fit and transform
    X_reduced = model.fit_transform(X)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Component 1': X_reduced[:, 0],
        'Component 2': X_reduced[:, 1],
        'Fault': y
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=plot_df, x='Component 1', y='Component 2', 
                   hue='Fault', palette=['blue', 'red'], alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'output/{method}_visualization.png', dpi=300, bbox_inches='tight')
    
    # Create interactive version with Plotly
    fig = px.scatter(plot_df, x='Component 1', y='Component 2', color='Fault',
                    color_discrete_sequence=['blue', 'red'],
                    title=title)
    
    fig.update_layout(
        width=900,
        height=700
    )
    
    fig.write_html(f'output/{method}_visualization_interactive.html')

def plot_model_predictions_over_time(df, y_true, y_pred, y_prob=None):
    """
    Plot model predictions over time.
    
    Args:
        df: Original DataFrame with timestamps
        y_true: True fault labels
        y_pred: Predicted fault labels
        y_prob: Predicted fault probabilities (optional)
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Timestamp': df.index,
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    if y_prob is not None:
        plot_df['Probability'] = y_prob
    
    # Plot predictions over time
    plt.figure(figsize=(15, 8))
    
    plt.plot(plot_df['Timestamp'], plot_df['Actual'], 'b-', label='Actual', alpha=0.7)
    plt.plot(plot_df['Timestamp'], plot_df['Predicted'], 'r-', label='Predicted', alpha=0.7)
    
    if 'Probability' in plot_df.columns:
        plt.plot(plot_df['Timestamp'], plot_df['Probability'], 'g-', label='Probability', alpha=0.5)
    
    plt.title('Fault Predictions Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Fault Indicator / Probability', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/predictions_timeline.png', dpi=300, bbox_inches='tight')
    
    # Create interactive version with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=plot_df['Timestamp'], y=plot_df['Actual'],
                           mode='lines', name='Actual', line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=plot_df['Timestamp'], y=plot_df['Predicted'],
                           mode='lines', name='Predicted', line=dict(color='red')))
    
    if 'Probability' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['Timestamp'], y=plot_df['Probability'],
                               mode='lines', name='Probability', line=dict(color='green')))
    
    fig.update_layout(
        title='Fault Predictions Over Time',
        xaxis_title='Time',
        yaxis_title='Fault Indicator / Probability',
        width=1000,
        height=600
    )
    
    fig.write_html('output/predictions_timeline_interactive.html')

def plot_early_detection_analysis(df, y_true, y_pred, prediction_horizon=24):
    """
    Analyze and visualize early fault detection performance.
    
    Args:
        df: Original DataFrame with timestamps
        y_true: True fault labels (1 if fault occurs within prediction_horizon)
        y_pred: Predicted fault labels
        prediction_horizon: Hours ahead for prediction
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Find actual fault events
    fault_mask = df['fault_type'] != 'NO_FAULT'
    fault_events = []
    
    for i in range(1, len(df)):
        if not fault_mask.iloc[i-1] and fault_mask.iloc[i]:
            fault_events.append(i)
    
    if not fault_events:
        print("No fault events found for early detection analysis")
        return
    
    # Analyze detection time for each fault event
    detection_times = []
    
    for fault_idx in fault_events:
        # Look back up to prediction_horizon hours
        start_idx = max(0, fault_idx - prediction_horizon)
        
        # Check when the model first predicted the fault
        for i in range(start_idx, fault_idx + 1):
            if y_pred[i] == 1:
                detection_time = fault_idx - i
                detection_times.append(detection_time)
                break
        else:
            # Fault not detected
            detection_times.append(-1)
    
    # Calculate early detection metrics
    detected_faults = [t >= 0 for t in detection_times]
    detection_rate = sum(detected_faults) / len(fault_events) * 100
    
    early_detections = [t > 0 for t in detection_times]
    early_detection_rate = sum(early_detections) / len(fault_events) * 100
    
    avg_detection_time = np.mean([t for t in detection_times if t >= 0]) if any(t >= 0 for t in detection_times) else 0
    
    # Print metrics
    print("\nEarly Detection Analysis:")
    print(f"Total fault events: {len(fault_events)}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"Early detection rate: {early_detection_rate:.2f}%")
    print(f"Average detection time: {avg_detection_time:.2f} hours before fault")
    
    # Plot detection time distribution
    plt.figure(figsize=(12, 6))
    
    bins = np.arange(-1.5, prediction_horizon + 1.5, 1)
    plt.hist([t for t in detection_times if t >= 0], bins=bins, alpha=0.7)
    
    plt.title('Distribution of Fault Detection Times', fontsize=16)
    plt.xlabel('Hours Before Fault Occurrence', fontsize=12)
    plt.ylabel('Number of Faults', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/detection_time_distribution.png', dpi=300, bbox_inches='tight')
    
    # Return metrics
    return {
        'detection_rate': detection_rate,
        'early_detection_rate': early_detection_rate,
        'avg_detection_time': avg_detection_time
    }

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('notebooks/fault_analysis_data.csv', index_col=0, parse_dates=True)
    
    # Plot fault timeline
    plot_fault_timeline(df)
    
    # Plot fault heatmap
    plot_fault_heatmap(df)
    
    # Plot system parameters during faults
    plot_system_parameters_during_faults(df)
    
    # Plot feature correlations
    plot_feature_correlations(df)
    
    # Plot feature distributions for top correlated features
    top_features = [
        'battery_soc', 'battery_power', 'grid_voltage',
        'solar_cell_temp', 'generator_temperature'
    ]
    plot_feature_distributions(df, top_features)
    
    print("Visualization complete. Check output directory for plots.")
