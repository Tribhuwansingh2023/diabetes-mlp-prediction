"""
Data Preprocessing Module for Pima Indians Diabetes Dataset.
Handles biological zero-value detection, imputation, outlier analysis, and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Biologically implausible zero-value columns
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

class DiabetesPreprocessor:
    def __init__(self, strategy='median', scaler_type='standard'):
        """
        strategy: 'median', 'mean', or 'knn'
        scaler_type: 'standard' or 'minmax'
        """
        self.strategy = strategy
        self.scaler_type = scaler_type
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        """Fit imputer and scaler on training data only to prevent leakage."""
        X_copy = X.copy()
        # Replace 0s with NaN in biological columns
        for col in ZERO_COLS:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(0, np.nan)
                
        if self.strategy in ['median', 'mean']:
            self.imputer = SimpleImputer(strategy=self.strategy)
        elif self.strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            
        imputed_arr = self.imputer.fit_transform(X_copy)
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            
        self.scaler.fit(imputed_arr)
        self.feature_names = list(X.columns)
        return self

    def transform(self, X):
        """Transform data using fitted imputer and scaler."""
        X_copy = X.copy()
        for col in ZERO_COLS:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(0, np.nan)
        imputed_arr = self.imputer.transform(X_copy)
        scaled_arr = self.scaler.transform(imputed_arr)
        return pd.DataFrame(scaled_arr, columns=self.feature_names, index=X.index)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def load_and_split_data(data_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads dataset and performs stratified 3-way split (Train 70%, Validation 15%, Test 15%).
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # First split: Separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: Separate validation set from train+val
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df
