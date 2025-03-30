"""
Fault Analysis module for Hybrid Energy Systems.
This module provides tools for analyzing and predicting faults in hybrid energy systems.
"""

from .data_preparation import (
    load_dataset, explore_faults, create_time_features,
    create_lag_features, create_window_features,
    prepare_fault_dataset, split_and_scale_data
)

from .model_training import (
    train_binary_classifier, evaluate_classifier,
    train_multiclass_classifier, evaluate_multiclass_classifier,
    save_model, load_model
)

from .visualization import (
    plot_fault_timeline, plot_fault_heatmap,
    plot_system_parameters_during_faults, plot_feature_correlations,
    plot_feature_distributions, plot_dimensionality_reduction,
    plot_model_predictions_over_time, plot_early_detection_analysis
)

__all__ = [
    'load_dataset', 'explore_faults', 'create_time_features',
    'create_lag_features', 'create_window_features',
    'prepare_fault_dataset', 'split_and_scale_data',
    'train_binary_classifier', 'evaluate_classifier',
    'train_multiclass_classifier', 'evaluate_multiclass_classifier',
    'save_model', 'load_model',
    'plot_fault_timeline', 'plot_fault_heatmap',
    'plot_system_parameters_during_faults', 'plot_feature_correlations',
    'plot_feature_distributions', 'plot_dimensionality_reduction',
    'plot_model_predictions_over_time', 'plot_early_detection_analysis'
]
