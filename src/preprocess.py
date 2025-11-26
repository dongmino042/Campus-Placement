"""
Data preprocessing module for Campus Placement prediction project.
Handles data cleaning, feature engineering, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data import RANDOM_SEED


class PlacementDataPreprocessor:
    """
    Preprocessor for the Campus Placement dataset.
    Handles encoding, scaling, and feature transformation.
    """
    
    def __init__(self, random_seed=RANDOM_SEED):
        """
        Initialize the preprocessor.
        
        Args:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_encoder = LabelEncoder()
        
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and duplicates.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Remove serial number column as it's not a feature
        if 'sl_no' in df_clean.columns:
            df_clean = df_clean.drop('sl_no', axis=1)
        
        # Handle missing values in salary (only present for placed students)
        # We'll fill with 0 for not placed students
        if 'salary' in df_clean.columns:
            df_clean['salary'] = df_clean['salary'].fillna(0)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        print(f"Data cleaned: {len(df)} -> {len(df_clean)} rows")
        
        return df_clean
    
    def prepare_features_target(self, df, target_col='status', drop_salary=True):
        """
        Separate features and target variable.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            drop_salary (bool): Whether to drop salary column (to avoid data leakage)
            
        Returns:
            tuple: (X, y) features and target
        """
        df_prep = df.copy()
        
        # Drop salary to avoid data leakage (salary is only known after placement)
        if drop_salary and 'salary' in df_prep.columns:
            df_prep = df_prep.drop('salary', axis=1)
        
        # Separate features and target
        y = df_prep[target_col]
        X = df_prep.drop(target_col, axis=1)
        
        return X, y
    
    def encode_categorical_features(self, X, fit=True):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            X (pd.DataFrame): Input features
            fit (bool): Whether to fit the encoders (True for training, False for test)
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        # Identify categorical columns
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if fit:
                # Fit and transform for training data
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col])
            else:
                # Transform only for test data
                if col in self.label_encoders:
                    X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
        
        return X_encoded
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Input features
            fit (bool): Whether to fit the scaler (True for training, False for test)
            
        Returns:
            pd.DataFrame: Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame with original column names
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled
    
    def encode_target(self, y, fit=True):
        """
        Encode target variable (Placed/Not Placed -> 1/0).
        
        Args:
            y (pd.Series): Target variable
            fit (bool): Whether to fit the encoder
            
        Returns:
            np.ndarray: Encoded target
        """
        if fit:
            y_encoded = self.target_encoder.fit_transform(y)
        else:
            y_encoded = self.target_encoder.transform(y)
        
        return y_encoded
    
    def fit_transform(self, X, y):
        """
        Fit the preprocessor and transform the data (for training set).
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            
        Returns:
            tuple: (X_transformed, y_transformed)
        """
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X, fit=True)
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, fit=True)
        
        # Store feature names
        self.feature_names = X_scaled.columns.tolist()
        
        # Encode target
        y_encoded = self.encode_target(y, fit=True)
        
        print(f"Features transformed: {X_scaled.shape}")
        print(f"Target classes: {self.target_encoder.classes_}")
        
        return X_scaled, y_encoded
    
    def transform(self, X, y=None):
        """
        Transform the data using fitted preprocessor (for test set).
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            tuple or pd.DataFrame: Transformed features (and target if provided)
        """
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X, fit=False)
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, fit=False)
        
        if y is not None:
            # Encode target
            y_encoded = self.encode_target(y, fit=False)
            return X_scaled, y_encoded
        
        return X_scaled


def prepare_train_test_split(df, test_size=0.2, random_seed=RANDOM_SEED):
    """
    Prepare and split the data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of data to use for testing
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Initialize preprocessor
    preprocessor = PlacementDataPreprocessor(random_seed=random_seed)
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Prepare features and target
    X, y = preprocessor.prepare_features_target(df_clean)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\nTrain-Test Split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Fit and transform training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    return X_train_processed, X_test_processed, y_train_processed, y_test_processed, preprocessor


if __name__ == '__main__':
    # Test preprocessing
    from data import load_data
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(df)
        
        print("\n" + "=" * 80)
        print("Preprocessing completed successfully!")
        print("=" * 80)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
