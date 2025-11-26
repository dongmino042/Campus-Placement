"""
Data loading module for Campus Placement prediction project.
Handles loading and basic validation of the dataset.
"""

import pandas as pd
import os
from pathlib import Path


# Fixed random seed for reproducibility
RANDOM_SEED = 42


def get_data_path():
    """
    Get the path to the data directory.
    
    Returns:
        Path: Path object pointing to the data directory
    """
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    return data_dir


def load_data(filename='Placement_Data_Full_Class.csv'):
    """
    Load the campus placement dataset from CSV file.
    
    Args:
        filename (str): Name of the CSV file to load
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the data file is not found
    """
    data_path = get_data_path() / filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            f"Please download the dataset following instructions in data/README.md"
        )
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully from {data_path}")
    print(f"Shape: {df.shape}")
    
    return df


def get_basic_info(df):
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing basic dataset information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    return info


def display_data_info(df):
    """
    Display comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n" + "-" * 80)
    print("Column Information:")
    print("-" * 80)
    print(df.info())
    
    print("\n" + "-" * 80)
    print("Missing Values:")
    print("-" * 80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print("\n" + "-" * 80)
    print("Duplicate Rows:")
    print("-" * 80)
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\n" + "-" * 80)
    print("First Few Rows:")
    print("-" * 80)
    print(df.head())
    
    print("\n" + "-" * 80)
    print("Statistical Summary:")
    print("-" * 80)
    print(df.describe())
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Test the data loading
    try:
        df = load_data()
        display_data_info(df)
    except FileNotFoundError as e:
        print(f"Error: {e}")
