#!/usr/bin/env python3
"""
run_models.py

Usage:
  python run_models.py --data data/Placement_Data_Full_Class.csv --outdir outputs

Requirements:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost

What it does:
  - Preprocess dataset
  - Train classification models to predict status (Placed / Not Placed)
  - Train regression models to predict salary (only on Placed records)
  - Save best models and plots to output directory
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgboost_available = True
except ImportError:
    xgboost_available = False


# Minimum samples required for regression training
MIN_REGRESSION_SAMPLES = 5


def load_data(path):
    """Load CSV data with error handling."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}. Please ensure the file exists.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The data file is empty: {path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {path}: {str(e)}")


def preprocess(df, for_regression=False):
    # Copy to avoid modifying original
    data = df.copy()
    # Drop sl_no
    if 'sl_no' in data.columns:
        data = data.drop(columns=['sl_no'])
    # Trim whitespace in string columns
    for c in data.select_dtypes(include='object').columns:
        data[c] = data[c].str.strip()
    # Map status to binary
    if 'status' in data.columns:
        data['status_bin'] = data['status'].map({'Placed':1, 'Not Placed':0})
    # For regression, filter placed only and ensure salary numeric
    if for_regression:
        data = data[data['status'] == 'Placed'].copy()
        data = data.dropna(subset=['salary'])
    # Select features and target
    # Identify numeric and categorical
    numeric_cols = ['ssc_p','hsc_p','degree_p','etest_p','mba_p']
    cat_cols = [c for c in ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'] if c in data.columns]
    # Some datasets have 'salary' numeric, ensure dtype
    if 'salary' in data.columns:
        data['salary'] = pd.to_numeric(data['salary'], errors='coerce')
    return data, numeric_cols, cat_cols


def create_feature_matrix(df, numeric_cols, cat_cols):
    """Helper function to create feature matrix with numeric and categorical features."""
    X_num = df[numeric_cols]
    X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X


def prepare_X_y_classification(df, numeric_cols, cat_cols):
    y = df['status_bin'].astype(int)
    X = create_feature_matrix(df, numeric_cols, cat_cols)
    return X, y


def prepare_X_y_regression(df, numeric_cols, cat_cols):
    y = df['salary'].astype(float)
    X = create_feature_matrix(df, numeric_cols, cat_cols)
    return X, y


def evaluate_classification(models, X_test, y_test, outdir):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, preds, digits=4))
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        
        results[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'confusion_{name}.png'))
        plt.close()
    
    return results


def evaluate_regression(models, X_test, y_test, outdir):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"\n--- {name} Regression Metrics ---")
        print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")
        
        results[name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2)
        }
        
        # Scatter plot
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_test, y=preds)
        maxv = max(y_test.max(), preds.max())*1.05
        minv = min(y_test.min(), preds.min())*0.95
        plt.plot([minv, maxv], [minv, maxv], '--', color='gray')
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.title(f'Predicted vs Actual Salary ({name})')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'pred_vs_actual_{name}.png'))
        plt.close()
    
    return results


def create_results_summary(clf_results, reg_results, dataset_info, outdir):
    """Create a comprehensive text summary of results."""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("CAMPUS PLACEMENT PREDICTION - RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"\n{'-' * 80}")
    summary_lines.append("DATASET INFORMATION")
    summary_lines.append(f"{'-' * 80}")
    summary_lines.append(f"Total samples: {dataset_info['total_samples']}")
    summary_lines.append(f"Placed students: {dataset_info['placed_count']} ({dataset_info['placed_percentage']:.1f}%)")
    summary_lines.append(f"Not placed students: {dataset_info['not_placed_count']} ({dataset_info['not_placed_percentage']:.1f}%)")
    
    if clf_results:
        summary_lines.append(f"\n{'-' * 80}")
        summary_lines.append("CLASSIFICATION MODELS (Placement Prediction)")
        summary_lines.append(f"{'-' * 80}")
        
        # Find best model
        best_model = max(clf_results.items(), key=lambda x: x[1]['accuracy'])
        
        for model_name, metrics in clf_results.items():
            is_best = model_name == best_model[0]
            marker = " ⭐ BEST" if is_best else ""
            summary_lines.append(f"\n{model_name}{marker}")
            summary_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            summary_lines.append(f"  Precision: {metrics['precision']:.4f}")
            summary_lines.append(f"  Recall:    {metrics['recall']:.4f}")
            summary_lines.append(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        summary_lines.append(f"\nBest Classification Model: {best_model[0]}")
        summary_lines.append(f"Best Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
    
    if reg_results:
        summary_lines.append(f"\n{'-' * 80}")
        summary_lines.append("REGRESSION MODELS (Salary Prediction)")
        summary_lines.append(f"{'-' * 80}")
        
        # Find best model (lowest MAE)
        best_model = min(reg_results.items(), key=lambda x: x[1]['mae'])
        
        for model_name, metrics in reg_results.items():
            is_best = model_name == best_model[0]
            marker = " ⭐ BEST" if is_best else ""
            summary_lines.append(f"\n{model_name}{marker}")
            summary_lines.append(f"  MAE:      ₹{metrics['mae']:,.2f}")
            summary_lines.append(f"  RMSE:     ₹{metrics['rmse']:,.2f}")
            summary_lines.append(f"  R² Score: {metrics['r2_score']:.4f}")
        
        summary_lines.append(f"\nBest Regression Model: {best_model[0]}")
        summary_lines.append(f"Lowest MAE: ₹{best_model[1]['mae']:,.2f}")
    
    summary_lines.append(f"\n{'=' * 80}")
    summary_lines.append("OUTPUTS GENERATED")
    summary_lines.append(f"{'=' * 80}")
    summary_lines.append(f"📊 Classification confusion matrices: confusion_*.png")
    summary_lines.append(f"📈 Regression scatter plots: pred_vs_actual_*.png")
    summary_lines.append(f"💾 Trained models: model_clf_*.joblib, model_reg_*.joblib")
    summary_lines.append(f"📄 Results summary: results_summary.txt")
    summary_lines.append(f"📋 Results JSON: results.json")
    summary_lines.append(f"📊 Results dashboard: results_dashboard.png")
    summary_lines.append(f"\n{'=' * 80}\n")
    
    summary_text = "\n".join(summary_lines)
    
    # Save to file
    with open(os.path.join(outdir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    # Also print to console
    print("\n" + summary_text)
    
    return summary_text


def create_results_json(clf_results, reg_results, dataset_info, outdir):
    """Create a JSON file with all results."""
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': dataset_info,
        'classification_results': clf_results,
        'regression_results': reg_results,
        'best_classification_model': max(clf_results.items(), key=lambda x: x[1]['accuracy'])[0] if clf_results else None,
        'best_regression_model': min(reg_results.items(), key=lambda x: x[1]['mae'])[0] if reg_results else None
    }
    
    with open(os.path.join(outdir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    return results_data


def create_results_dashboard(clf_results, reg_results, dataset_info, outdir):
    """Create a comprehensive results dashboard visualization."""
    # Determine layout based on what results we have
    has_clf = bool(clf_results)
    has_reg = bool(reg_results)
    
    if has_clf and has_reg:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Campus Placement Prediction - Results Dashboard', fontsize=16, fontweight='bold')
    elif has_clf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Campus Placement Prediction - Classification Results', fontsize=16, fontweight='bold')
        axes = [axes]
    elif has_reg:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Campus Placement Prediction - Regression Results', fontsize=16, fontweight='bold')
        axes = [axes]
    else:
        return
    
    # Flatten axes for easier indexing
    if has_clf and has_reg:
        ax1, ax2, ax3, ax4 = axes.flatten()
    else:
        ax1, ax2 = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    # Plot 1: Dataset overview
    ax1.axis('off')
    info_text = [
        'Dataset Overview',
        '─' * 40,
        f'Total Samples: {dataset_info["total_samples"]}',
        f'Placed: {dataset_info["placed_count"]} ({dataset_info["placed_percentage"]:.1f}%)',
        f'Not Placed: {dataset_info["not_placed_count"]} ({dataset_info["not_placed_percentage"]:.1f}%)',
    ]
    ax1.text(0.1, 0.9, '\n'.join(info_text), transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 2: Placement distribution pie chart
    labels = ['Placed', 'Not Placed']
    sizes = [dataset_info['placed_count'], dataset_info['not_placed_count']]
    colors = ['#66b3ff', '#ff9999']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Placement Status Distribution')
    
    if has_clf:
        # Plot 3: Classification metrics comparison
        model_names = list(clf_results.keys())
        accuracy_scores = [clf_results[m]['accuracy'] for m in model_names]
        f1_scores = [clf_results[m]['f1_score'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        if has_reg:
            ax3.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='skyblue')
            ax3.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Score')
            ax3.set_title('Classification Models Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(model_names, rotation=15, ha='right')
            ax3.legend()
            ax3.set_ylim(0, 1.1)
            ax3.grid(axis='y', alpha=0.3)
        else:
            # If no regression, use ax1 for metrics comparison
            pass
    
    if has_reg:
        # Plot 4: Regression metrics comparison
        model_names = list(reg_results.keys())
        mae_scores = [reg_results[m]['mae'] for m in model_names]
        r2_scores = [reg_results[m]['r2_score'] for m in model_names]
        
        if has_clf:
            ax4_twin = ax4.twinx()
            x = np.arange(len(model_names))
            
            bars1 = ax4.bar(x - 0.2, mae_scores, 0.4, label='MAE', color='orange', alpha=0.7)
            ax4.set_xlabel('Model')
            ax4.set_ylabel('MAE (₹)', color='orange')
            ax4.tick_params(axis='y', labelcolor='orange')
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names, rotation=15, ha='right')
            
            bars2 = ax4_twin.bar(x + 0.2, r2_scores, 0.4, label='R²', color='green', alpha=0.7)
            ax4_twin.set_ylabel('R² Score', color='green')
            ax4_twin.tick_params(axis='y', labelcolor='green')
            ax4_twin.set_ylim(-0.2, 1.0)
            
            ax4.set_title('Regression Models Comparison')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'results_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Results dashboard saved: {os.path.join(outdir, 'results_dashboard.png')}")





def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.data)
    print(f"Loaded data with shape: {df.shape}")
    
    # Collect dataset info
    placed_count = (df['status'] == 'Placed').sum()
    not_placed_count = (df['status'] == 'Not Placed').sum()
    total_samples = len(df)
    
    dataset_info = {
        'total_samples': int(total_samples),
        'placed_count': int(placed_count),
        'not_placed_count': int(not_placed_count),
        'placed_percentage': float(placed_count / total_samples * 100),
        'not_placed_percentage': float(not_placed_count / total_samples * 100)
    }

    # Basic histogram similar to your image
    if 'salary' in df.columns:
        plt.figure(figsize=(6,3.5))
        sns.histplot(df['salary'].dropna(), kde=True, bins=20)
        plt.title('Histogram: salary')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'hist_salary.png'))
        plt.close()

    # Classification
    data_class, numeric_cols, cat_cols = preprocess(df, for_regression=False)
    X_class, y_class = prepare_X_y_classification(data_class, numeric_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
    print(f"Classification train/test shapes: {X_train.shape} / {X_test.shape}")

    # Fit simple pipelines / models
    # Logistic Regression
    log = Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    log.fit(X_train, y_train)
    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models_clf = {'LogisticRegression': log, 'RandomForest': rf}
    # XGBoost if available
    if xgboost_available:
        xgb = XGBClassifier(n_estimators=200, eval_metric='logloss', verbosity=0, random_state=42)
        xgb.fit(X_train, y_train)
        models_clf['XGBoost'] = xgb

    clf_results = evaluate_classification(models_clf, X_test, y_test, args.outdir)
    # Save classification models
    for name, model in models_clf.items():
        joblib.dump(model, os.path.join(args.outdir, f'model_clf_{name}.joblib'))

    # Regression (salary) on placed only
    data_reg, numeric_cols_reg, cat_cols_reg = preprocess(df, for_regression=True)
    reg_results = {}
    
    if 'salary' not in data_reg.columns or data_reg.shape[0] < MIN_REGRESSION_SAMPLES:
        print(f"Not enough salary data to train regression model (need at least {MIN_REGRESSION_SAMPLES} samples). Skipping regression.")
    else:
        X_reg, y_reg = prepare_X_y_regression(data_reg, numeric_cols_reg, cat_cols_reg)
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        print(f"Regression train/test shapes: {Xr_train.shape} / {Xr_test.shape}")

        # Simple regression models
        rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_reg.fit(Xr_train, yr_train)
        models_reg = {'RandomForestRegressor': rf_reg}

        if xgboost_available:
            xgb_reg = XGBRegressor(n_estimators=200, verbosity=0, random_state=42)
            xgb_reg.fit(Xr_train, yr_train)
            models_reg['XGBoostRegressor'] = xgb_reg

        reg_results = evaluate_regression(models_reg, Xr_test, yr_test, args.outdir)

        # Save regression models
        for name, model in models_reg.items():
            joblib.dump(model, os.path.join(args.outdir, f'model_reg_{name}.joblib'))

    # Create comprehensive results summary
    print("\n" + "="*80)
    print("GENERATING RESULTS SUMMARY...")
    print("="*80)
    
    create_results_summary(clf_results, reg_results, dataset_info, args.outdir)
    create_results_json(clf_results, reg_results, dataset_info, args.outdir)
    create_results_dashboard(clf_results, reg_results, dataset_info, args.outdir)

    print(f"\n✅ All outputs saved to {args.outdir}")
    print(f"📄 Check 'results_summary.txt' for a detailed summary")
    print(f"📊 Check 'results_dashboard.png' for visual overview")
    print(f"📋 Check 'results.json' for programmatic access to results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Placement_Data_Full_Class.csv', help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory for models and plots')
    args = parser.parse_args()
    main(args)
