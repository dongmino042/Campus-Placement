# Campus Placement Prediction

A reproducible machine learning project for predicting campus placement outcomes using the "Factors Affecting Campus Placement" dataset from Kaggle.

## 📊 Project Overview

This project implements a complete machine learning pipeline to predict whether a student will be placed in campus recruitment based on various academic and personal factors. The project follows best practices for reproducibility, including:

- **Fixed random seed** for reproducible results
- **Comprehensive data preprocessing** pipeline
- **Multiple ML models** with hyperparameter tuning
- **Cross-validation** for robust model selection
- **Detailed evaluation** with multiple metrics
- **Well-structured codebase** with modular design

## 🎯 Problem Statement

Predict whether a student will be placed (`Placed`) or not placed (`Not Placed`) during campus recruitment based on features such as:
- Academic performance (SSC, HSC, Degree, MBA percentages)
- Board of education
- Specialization
- Work experience
- Employability test scores

## 📁 Project Structure

```
Campus-Placement/
├── data/                           # Data directory
│   ├── README.md                   # Data download instructions
│   └── .gitignore                  # Ignore CSV files
├── notebooks/                      # Jupyter notebooks
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   └── 02_Modeling.ipynb          # Model training and evaluation
├── src/                            # Source code modules
│   ├── __init__.py                # Package initialization
│   ├── data.py                    # Data loading utilities
│   ├── preprocess.py              # Data preprocessing
│   ├── train.py                   # Model training with CV and tuning
│   └── evaluate.py                # Model evaluation
├── models/                         # Saved models directory
│   ├── .gitignore                 # Ignore model files
│   └── best_model.pkl             # Best trained model (generated)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # License file
└── .gitignore                      # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dongmino042/Campus-Placement.git
cd Campus-Placement
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Download Dataset

Follow the instructions in `data/README.md` to download the dataset. You can either:

**Option 1: Manual Download**
- Visit [Kaggle dataset page](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- Download and extract the ZIP file
- Place `Placement_Data_Full_Class.csv` in the `data/` directory

**Option 2: Using Kaggle API**
```bash
pip install kaggle
# Set up Kaggle credentials (see data/README.md)
kaggle datasets download -d benroshan/factors-affecting-campus-placement
unzip factors-affecting-campus-placement.zip -d data/
```

## 📊 Usage

### Option 1: Using Jupyter Notebooks (Recommended for exploration)

1. **Exploratory Data Analysis:**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
This notebook provides comprehensive data analysis, visualizations, and insights.

2. **Model Training and Evaluation:**
```bash
jupyter notebook notebooks/02_Modeling.ipynb
```
This notebook trains multiple models, performs hyperparameter tuning, and evaluates performance.

### Option 2: Using Python Scripts

1. **Load and explore data:**
```bash
cd src
python data.py
```

2. **Preprocess data:**
```bash
python preprocess.py
```

3. **Train models:**
```python
from data import load_data
from preprocess import prepare_train_test_split
from train import train_all_models, select_best_model, save_model

# Load and preprocess
df = load_data()
X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(df)

# Train models with CV and hyperparameter tuning
models = train_all_models(X_train, y_train, cv=5)

# Select and save best model
best_model_name, best_model, best_score = select_best_model(models)
save_model(best_model, 'best_model.pkl')
```

4. **Evaluate models:**
```python
from evaluate import evaluate_model, print_evaluation_metrics

metrics = evaluate_model(best_model, X_test, y_test, best_model_name)
print_evaluation_metrics(metrics)
```

## 🤖 Models Implemented

The project implements and compares three machine learning models:

1. **Logistic Regression**
   - Hyperparameters: C, penalty, solver
   - Fast training, interpretable results

2. **Random Forest**
   - Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
   - Robust to overfitting, handles non-linear relationships

3. **XGBoost**
   - Hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma
   - State-of-the-art gradient boosting, excellent performance

All models are trained with:
- **5-fold cross-validation** for robust evaluation
- **Grid search** for hyperparameter tuning
- **ROC AUC** as the primary optimization metric

## 📈 Evaluation Metrics

Models are evaluated using comprehensive metrics:

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

Additional visualizations:
- Confusion matrix
- ROC curve
- Feature importance (for tree-based models)
- Model comparison charts

## 🔄 Reproducibility

The project ensures reproducibility through:

- **Fixed random seed** (`RANDOM_SEED = 42`) used throughout the pipeline
- **Deterministic preprocessing** with consistent train-test splits
- **Version-pinned dependencies** in `requirements.txt`
- **Stratified sampling** to maintain class distribution

## 📝 Key Features

- ✅ Comprehensive data preprocessing pipeline
- ✅ Multiple ML models with hyperparameter tuning
- ✅ Cross-validation for robust model selection
- ✅ Detailed evaluation with multiple metrics
- ✅ Visualization of results
- ✅ Feature importance analysis
- ✅ Model persistence (save/load)
- ✅ Clean, modular code structure
- ✅ Well-documented notebooks
- ✅ Reproducible results with fixed random seed

## 📊 Dataset Information

**Source**: [Kaggle - Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)

**Features**:
- Academic performance metrics (SSC, HSC, Degree, MBA percentages)
- Board of education
- Degree type and specialization
- Work experience
- Employability test scores
- Gender

**Target**: Placement Status (Placed/Not Placed)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Notebooks**: Jupyter
- **Model Persistence**: Joblib

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**dongmino042**

## 🙏 Acknowledgments

- Dataset provided by [Ben Roshan](https://www.kaggle.com/benroshan) on Kaggle
- Inspired by real-world campus placement challenges

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational purposes and demonstrates best practices in machine learning project structure, reproducibility, and model evaluation.
