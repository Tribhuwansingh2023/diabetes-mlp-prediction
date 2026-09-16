"""
Data Preprocessing Module for Pima Indians Diabetes Dataset.
Provides the authoritative DiabetesPreprocessor pipeline:
1. Suspicious biological zero-value replacement with NaN
2. Median imputation (fit on training data only)
3. Domain feature engineering on imputed data
4. Standard scaling (fit on training data only)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from feature_engineering import engineer_features, RAW_FEATURE_NAMES, ALL_FEATURE_NAMES

# Biologically implausible zero-value columns in the Pima dataset
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

class DiabetesPreprocessor:
    """
    Authoritative End-to-End Preprocessing Pipeline.
    Encapsulates missing-value detection, median imputation,
    feature engineering, and standardization.
    """
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None
        self.scaler = None
        self.raw_feature_names = list(RAW_FEATURE_NAMES)
        self.processed_feature_names = list(ALL_FEATURE_NAMES)
        self.feature_names_in_ = list(RAW_FEATURE_NAMES)
        self.feature_names_out_ = list(ALL_FEATURE_NAMES)
        self.n_features_in_ = len(RAW_FEATURE_NAMES)
        self.n_features_out_ = len(ALL_FEATURE_NAMES)
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit imputer on raw features and scaler on engineered features
        using training data ONLY to prevent data leakage.
        """
        X_df = pd.DataFrame(X, columns=self.raw_feature_names).copy()
        
        # Step 1: Replace biologically implausible zeros with NaN
        for col in ZERO_COLS:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(0, np.nan)
                
        # Step 2: Fit median imputer on raw features
        self.imputer = SimpleImputer(strategy=self.strategy)
        imputed_raw = self.imputer.fit_transform(X_df[self.raw_feature_names])
        df_imputed = pd.DataFrame(imputed_raw, columns=self.raw_feature_names, index=X_df.index)
        
        # Step 3: Compute domain feature engineering on imputed data
        df_engineered = engineer_features(df_imputed)
        self.processed_feature_names = list(df_engineered.columns)
        self.feature_names_out_ = list(df_engineered.columns)
        self.n_features_out_ = len(self.processed_feature_names)
        
        # Step 4: Fit scaler on all engineered features
        self.scaler = StandardScaler()
        self.scaler.fit(df_engineered.values)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw input DataFrame through the fitted pipeline.
        """
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("DiabetesPreprocessor must be fitted before calling transform().")
            
        X_df = pd.DataFrame(X, columns=self.raw_feature_names).copy()
        
        # Step 1: Replace zeros with NaN
        for col in ZERO_COLS:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(0, np.nan)
                
        # Step 2: Impute using fitted imputer
        imputed_raw = self.imputer.transform(X_df[self.raw_feature_names])
        df_imputed = pd.DataFrame(imputed_raw, columns=self.raw_feature_names, index=X_df.index)
        
        # Step 3: Feature engineering on imputed data
        df_engineered = engineer_features(df_imputed)
        
        # Step 4: Scale using fitted scaler
        scaled_arr = self.scaler.transform(df_engineered.values)
        
        return pd.DataFrame(scaled_arr, columns=self.processed_feature_names, index=X_df.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
        
    def get_feature_names_out(self):
        """Returns the list of processed feature names."""
        return list(self.processed_feature_names)

def load_and_split_data(data_path: str, test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads dataset and performs stratified 3-way split:
    - Train: 70% (536 samples)
    - Validation: 15% (116 samples)
    - Test: 15% (116 samples)
    """
    df = pd.read_csv(data_path)
    X = df[RAW_FEATURE_NAMES].copy()
    y = df['Outcome'].copy()
    
    # Split 1: Isolate untouched Test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Split 2: Separate Validation set (15% of total = 0.15 / 0.85 of train_val)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df
