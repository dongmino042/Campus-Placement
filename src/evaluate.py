"""
Model evaluation module for Campus Placement prediction project.
Handles model evaluation with various metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data with comprehensive metrics.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test (array-like): Test features
        y_test (array-like): True labels
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics


def print_evaluation_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 80)
    print(f"EVALUATION METRICS: {metrics['model_name']}")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 80)


def print_classification_report(model, X_test, y_test, target_names=None):
    """
    Print detailed classification report.
    
    Args:
        model: Trained model
        X_test (array-like): Test features
        y_test (array-like): True labels
        target_names (list, optional): Names of target classes
    """
    y_pred = model.predict(X_test)
    
    print("\n" + "-" * 80)
    print("CLASSIFICATION REPORT")
    print("-" * 80)
    
    if target_names is None:
        target_names = ['Not Placed', 'Placed']
    
    print(classification_report(y_test, y_pred, target_names=target_names))


def plot_confusion_matrix(model, X_test, y_test, model_name="Model", save_path=None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model: Trained model
        X_test (array-like): Test features
        y_test (array-like): True labels
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name="Model", save_path=None):
    """
    Plot ROC curve for model predictions.
    
    Args:
        model: Trained model
        X_test (array-like): Test features
        y_test (array-like): True labels
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models and return a summary dataframe.
    
    Args:
        models_dict (dict): Dictionary of {model_name: model} pairs
        X_test (array-like): Test features
        y_test (array-like): True labels
        
    Returns:
        pd.DataFrame: Comparison dataframe with all metrics
    """
    results = []
    
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model_name')
    df_results = df_results.sort_values('roc_auc', ascending=False)
    
    return df_results


def plot_model_comparison(df_results, save_path=None):
    """
    Plot comparison of multiple models across different metrics.
    
    Args:
        df_results (pd.DataFrame): Results dataframe from compare_models
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        df_results[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(df_results[metric]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Remove the extra subplot
    fig.delaxes(axes[1, 2])
    
    plt.suptitle('Model Comparison Across Metrics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def get_feature_importance(model, feature_names, top_n=10):
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names (list): List of feature names
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: DataFrame with feature importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        print(f"Model {type(model).__name__} does not support feature importance.")
        return None


def plot_feature_importance(importance_df, model_name="Model", save_path=None):
    """
    Plot feature importance.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with feature importance
        model_name (str): Name of the model
        save_path (str, optional): Path to save the plot
    """
    if importance_df is None:
        return
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Evaluation module loaded successfully!")
    print("This module provides functions for model evaluation.")
