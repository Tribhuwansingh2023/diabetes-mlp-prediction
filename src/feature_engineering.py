"""
Feature Engineering Module for Diabetes Prediction.
Implements domain-specific physiological indicators, non-linear interaction terms,
and logarithmic transformations on imputed biometric data.
"""

import numpy as np
import pandas as pd

RAW_FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

ENGINEERED_FEATURE_NAMES = [
    'BMI_Underweight', 'BMI_Normal', 'BMI_Overweight', 'BMI_Obese',
    'Glucose_Normal', 'Glucose_Prediabetes', 'Glucose_Diabetes',
    'Age_Young', 'Age_Middle', 'Age_Senior',
    'Insulin_Glucose_Ratio', 'Insulin_Resistance_Proxy',
    'Pregnancy_Age_Risk', 'BMI_Age_Interaction',
    'Log_Insulin', 'Log_DPF'
]

ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 16 engineered features from the 8 raw biometric features.
    
    IMPORTANT: This function expects that missing/zero values in biological
    variables (Glucose, BloodPressure, SkinThickness, Insulin, BMI) have already
    been properly handled/imputed before calling this function.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing at least the 8 raw feature columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 24 total feature columns (8 raw + 16 engineered).
    """
    df_feat = df.copy()
    
    # Ensure raw columns exist
    for col in RAW_FEATURE_NAMES:
        if col not in df_feat.columns:
            raise ValueError(f"Missing required raw feature column: '{col}'")
            
    # 1. WHO BMI Categories (Binary Indicators)
    df_feat['BMI_Underweight'] = (df_feat['BMI'] < 18.5).astype(float)
    df_feat['BMI_Normal'] = ((df_feat['BMI'] >= 18.5) & (df_feat['BMI'] < 25.0)).astype(float)
    df_feat['BMI_Overweight'] = ((df_feat['BMI'] >= 25.0) & (df_feat['BMI'] < 30.0)).astype(float)
    df_feat['BMI_Obese'] = (df_feat['BMI'] >= 30.0).astype(float)
    
    # 2. ADA Glucose Categories (Binary Indicators)
    df_feat['Glucose_Normal'] = (df_feat['Glucose'] < 100.0).astype(float)
    df_feat['Glucose_Prediabetes'] = ((df_feat['Glucose'] >= 100.0) & (df_feat['Glucose'] <= 125.0)).astype(float)
    df_feat['Glucose_Diabetes'] = (df_feat['Glucose'] >= 126.0).astype(float)
    
    # 3. Age Brackets (Binary Indicators)
    df_feat['Age_Young'] = (df_feat['Age'] < 30.0).astype(float)
    df_feat['Age_Middle'] = ((df_feat['Age'] >= 30.0) & (df_feat['Age'] <= 50.0)).astype(float)
    df_feat['Age_Senior'] = (df_feat['Age'] > 50.0).astype(float)
    
    # 4. Mathematical Interaction Terms
    # Insulin-to-Glucose ratio
    df_feat['Insulin_Glucose_Ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + 1e-5)
    
    # Insulin Resistance Proxy: (Glucose * Insulin) / 405.0
    # Note: This is an engineered interaction feature modeling glycemic-insulin product,
    # not a formal clinical diagnosis.
    df_feat['Insulin_Resistance_Proxy'] = (df_feat['Glucose'] * df_feat['Insulin']) / 405.0
    
    # Pregnancy to Age ratio
    df_feat['Pregnancy_Age_Risk'] = df_feat['Pregnancies'] / (df_feat['Age'] + 1e-5)
    
    # BMI and Age interaction
    df_feat['BMI_Age_Interaction'] = df_feat['BMI'] * df_feat['Age']
    
    # 5. Logarithmic Transformations for Right-Skewed Continuous Variables
    df_feat['Log_Insulin'] = np.log1p(np.maximum(0.0, df_feat['Insulin']))
    df_feat['Log_DPF'] = np.log1p(np.maximum(0.0, df_feat['DiabetesPedigreeFunction']))
    
    # Reorder columns explicitly to ensure strict consistency
    return df_feat[ALL_FEATURE_NAMES]
