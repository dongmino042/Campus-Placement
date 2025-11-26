"""
Model training module for Campus Placement prediction project.
Handles training of multiple models with cross-validation and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
from pathlib import Path
from data import RANDOM_SEED


def get_models_path():
    """
    Get the path to the models directory.
    
    Returns:
        Path: Path object pointing to the models directory
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    return models_dir


def train_logistic_regression(X_train, y_train, cv=5, random_seed=RANDOM_SEED):
    """
    Train Logistic Regression with hyperparameter tuning.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        cv (int): Number of cross-validation folds
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    print("\n" + "=" * 80)
    print("Training Logistic Regression")
    print("=" * 80)
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    }
    
    # Initialize model
    lr = LogisticRegression(random_state=random_seed)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        lr, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV ROC AUC Score: {cv_score:.4f}")
    
    return best_model, best_params, cv_score


def train_random_forest(X_train, y_train, cv=5, random_seed=RANDOM_SEED):
    """
    Train Random Forest with hyperparameter tuning.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        cv (int): Number of cross-validation folds
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    print("\n" + "=" * 80)
    print("Training Random Forest")
    print("=" * 80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=random_seed)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV ROC AUC Score: {cv_score:.4f}")
    
    return best_model, best_params, cv_score


def train_xgboost(X_train, y_train, cv=5, random_seed=RANDOM_SEED):
    """
    Train XGBoost with hyperparameter tuning.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        cv (int): Number of cross-validation folds
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    print("\n" + "=" * 80)
    print("Training XGBoost")
    print("=" * 80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Initialize model
    xgb = XGBClassifier(
        random_state=random_seed,
        eval_metric='logloss'
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        xgb, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV ROC AUC Score: {cv_score:.4f}")
    
    return best_model, best_params, cv_score


def train_all_models(X_train, y_train, cv=5, random_seed=RANDOM_SEED):
    """
    Train all models (Logistic Regression, Random Forest, XGBoost).
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        cv (int): Number of cross-validation folds
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing all trained models and their metadata
    """
    models = {}
    
    # Train Logistic Regression
    lr_model, lr_params, lr_score = train_logistic_regression(X_train, y_train, cv, random_seed)
    models['logistic_regression'] = {
        'model': lr_model,
        'params': lr_params,
        'cv_score': lr_score
    }
    
    # Train Random Forest
    rf_model, rf_params, rf_score = train_random_forest(X_train, y_train, cv, random_seed)
    models['random_forest'] = {
        'model': rf_model,
        'params': rf_params,
        'cv_score': rf_score
    }
    
    # Train XGBoost
    xgb_model, xgb_params, xgb_score = train_xgboost(X_train, y_train, cv, random_seed)
    models['xgboost'] = {
        'model': xgb_model,
        'params': xgb_params,
        'cv_score': xgb_score
    }
    
    return models


def select_best_model(models):
    """
    Select the best model based on cross-validation score.
    
    Args:
        models (dict): Dictionary containing trained models
        
    Returns:
        tuple: (best_model_name, best_model, best_score)
    """
    best_model_name = None
    best_model = None
    best_score = -1
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (CV Scores)")
    print("=" * 80)
    
    for name, model_info in models.items():
        score = model_info['cv_score']
        print(f"{name:25s}: ROC AUC = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model_info['model']
            best_model_name = name
    
    print("=" * 80)
    print(f"Best Model: {best_model_name} (ROC AUC = {best_score:.4f})")
    print("=" * 80)
    
    return best_model_name, best_model, best_score


def save_model(model, filename, models_dir=None):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        filename (str): Filename for the saved model
        models_dir (Path, optional): Directory to save the model
        
    Returns:
        Path: Path to the saved model
    """
    if models_dir is None:
        models_dir = get_models_path()
    
    model_path = models_dir / filename
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model_path


def load_model(filename, models_dir=None):
    """
    Load a trained model from disk.
    
    Args:
        filename (str): Filename of the saved model
        models_dir (Path, optional): Directory containing the model
        
    Returns:
        Loaded model
    """
    if models_dir is None:
        models_dir = get_models_path()
    
    model_path = models_dir / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    return model


def save_all_models(models):
    """
    Save all trained models to disk.
    
    Args:
        models (dict): Dictionary containing trained models
        
    Returns:
        dict: Dictionary mapping model names to file paths
    """
    saved_paths = {}
    
    for name, model_info in models.items():
        filename = f"{name}.pkl"
        path = save_model(model_info['model'], filename)
        saved_paths[name] = path
    
    return saved_paths


if __name__ == '__main__':
    print("Training module loaded successfully!")
    print("This module provides functions for model training and hyperparameter tuning.")
