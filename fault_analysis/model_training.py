"""
Model training module for fault prediction in hybrid energy systems.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
import shap

def train_binary_classifier(X_train, y_train, model_type='rf', param_grid=None):
    """
    Train a binary classifier for fault prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model ('rf', 'gb', or 'lr')
        param_grid: Parameter grid for hyperparameter tuning
    
    Returns:
        Trained model
    """
    # Define default parameter grids if not provided
    if param_grid is None:
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        elif model_type == 'lr':
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': [None, 'balanced']
            }
    
    # Initialize model
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError("Unsupported model type. Use 'rf', 'gb', or 'lr'.")
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform grid search with time series cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=tscv, scoring='f1',
        n_jobs=-1, verbose=1
    )
    
    # Fit model
    print(f"Training {model_type} model with grid search...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model

def evaluate_classifier(model, X_test, y_test, feature_names=None):
    """
    Evaluate a binary classifier and generate performance metrics and visualizations.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
    
    Returns:
        Dictionary of performance metrics
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('output/roc_curve.png', dpi=300, bbox_inches='tight')
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('output/pr_curve.png', dpi=300, bbox_inches='tight')
    
    # Plot feature importance if applicable
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
    
    # SHAP values for model interpretability
    try:
        plt.figure(figsize=(12, 8))
        explainer = shap.Explainer(model, X_test.iloc[:100])  # Use subset for efficiency
        shap_values = explainer(X_test.iloc[:100])
        
        shap.summary_plot(shap_values, X_test.iloc[:100], plot_size=(12, 8), show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('output/shap_importance.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Could not generate SHAP plot: {e}")
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }
    
    return metrics

def train_multiclass_classifier(X_train, y_train, model_type='rf', param_grid=None):
    """
    Train a multi-class classifier for fault type prediction.
    
    Args:
        X_train: Training features
        y_train: Training target (fault types)
        model_type: Type of model ('rf', 'gb')
        param_grid: Parameter grid for hyperparameter tuning
    
    Returns:
        Trained model
    """
    # Define default parameter grids if not provided
    if param_grid is None:
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
    
    # Initialize model
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type. Use 'rf' or 'gb'.")
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform grid search with time series cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=tscv, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )
    
    # Fit model
    print(f"Training {model_type} model for fault type prediction...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model

def evaluate_multiclass_classifier(model, X_test, y_test, feature_names=None):
    """
    Evaluate a multi-class classifier for fault type prediction.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test target (fault types)
        feature_names: List of feature names
    
    Returns:
        Dictionary of performance metrics
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Fault Type Prediction')
    plt.savefig('output/fault_type_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Plot feature importance if applicable
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Top 20 Feature Importances - Fault Type Prediction')
        plt.tight_layout()
        plt.savefig('output/fault_type_feature_importance.png', dpi=300, bbox_inches='tight')
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def save_model(model, model_name):
    """Save trained model to file."""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name}.pkl')
    print(f"Model saved to models/{model_name}.pkl")

def load_model(model_name):
    """Load trained model from file."""
    model_path = f'models/{model_name}.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")

if __name__ == "__main__":
    # Load preprocessed data
    X_train = pd.read_csv('notebooks/X_train.csv', index_col=0)
    X_test = pd.read_csv('notebooks/X_test.csv', index_col=0)
    y_train = pd.read_csv('notebooks/y_train.csv', index_col=0).squeeze()
    y_test = pd.read_csv('notebooks/y_test.csv', index_col=0).squeeze()
    
    print(f"Loaded training data: {X_train.shape}")
    print(f"Loaded test data: {X_test.shape}")
    
    # Train binary classifier
    model = train_binary_classifier(X_train, y_train, model_type='rf')
    
    # Evaluate model
    metrics = evaluate_classifier(model, X_test, y_test, feature_names=X_train.columns)
    
    # Save model
    save_model(model, 'fault_prediction_rf')
