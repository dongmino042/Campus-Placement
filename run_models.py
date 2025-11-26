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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgboost_available = True
except Exception:
    xgboost_available = False


def load_data(path):
    df = pd.read_csv(path)
    return df


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


def prepare_X_y_classification(df, numeric_cols, cat_cols):
    y = df['status_bin'].astype(int)
    X_num = df[numeric_cols]
    X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X, y


def prepare_X_y_regression(df, numeric_cols, cat_cols):
    y = df['salary'].astype(float)
    X_num = df[numeric_cols]
    X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X, y


def evaluate_classification(models, X_test, y_test, outdir):
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, preds, digits=4))
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'confusion_{name}.png'))
        plt.close()


def evaluate_regression(models, X_test, y_test, outdir):
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"\n--- {name} Regression Metrics ---")
        print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")
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


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.data)
    print(f"Loaded data with shape: {df.shape}")

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
        xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        models_clf['XGBoost'] = xgb

    evaluate_classification(models_clf, X_test, y_test, args.outdir)
    # Save classification models
    for name, model in models_clf.items():
        joblib.dump(model, os.path.join(args.outdir, f'model_clf_{name}.joblib'))

    # Regression (salary) on placed only
    data_reg, numeric_cols_reg, cat_cols_reg = preprocess(df, for_regression=True)
    if 'salary' not in data_reg.columns or data_reg.shape[0] < 5:
        print("Not enough salary data to train regression model. Skipping regression.")
        return

    X_reg, y_reg = prepare_X_y_regression(data_reg, numeric_cols_reg, cat_cols_reg)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    print(f"Regression train/test shapes: {Xr_train.shape} / {Xr_test.shape}")

    # Simple regression models
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_reg.fit(Xr_train, yr_train)
    models_reg = {'RandomForestRegressor': rf_reg}

    if xgboost_available:
        xgb_reg = XGBRegressor(n_estimators=200, random_state=42)
        xgb_reg.fit(Xr_train, yr_train)
        models_reg['XGBoostRegressor'] = xgb_reg

    evaluate_regression(models_reg, Xr_test, yr_test, args.outdir)

    # Save regression models
    for name, model in models_reg.items():
        joblib.dump(model, os.path.join(args.outdir, f'model_reg_{name}.joblib'))

    print(f"All outputs saved to {args.outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Placement_Data_Full_Class.csv', help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory for models and plots')
    args = parser.parse_args()
    main(args)
