"""
Feature Engineering Module for Diabetes Prediction.
Implements domain-specific indicators, non-linear interaction terms, and logarithmic transformations.
"""

import numpy as np
import pandas as pd

def engineer_features(df):
    """
    Applies comprehensive domain feature engineering:
    1. BMI Categories (WHO classification):
       - Underweight (<18.5)
       - Normal (18.5-24.9)
       - Overweight (25.0-29.9)
       - Obese (>=30.0)
    2. Glucose Clinical Categories (ADA guidelines):
       - Normal (<100 mg/dL)
       - Prediabetic (100-125 mg/dL)
       - Diabetic (>=126 mg/dL)
    3. Age Cohorts:
       - Young (<30)
       - Middle (30-50)
       - Senior (>50)
    4. Biological Interaction Ratios:
       - Insulin_Glucose_Ratio = Insulin / (Glucose + 1e-5)
       - Insulin_Resistance_Proxy = (Glucose * Insulin) / 405 (HOMA-IR proxy)
       - Pregnancy_Age_Risk = Pregnancies / (Age + 1e-5)
       - BMI_Age_Interaction = BMI * Age
    5. Logarithmic Transformations on Skewed Features:
       - Log_Insulin = log1p(Insulin)
       - Log_DPF = log1p(DiabetesPedigreeFunction)
    """
    df_feat = df.copy()
    
    # 1. BMI Categories
    df_feat['BMI_Underweight'] = (df_feat['BMI'] < 18.5).astype(int)
    df_feat['BMI_Normal'] = ((df_feat['BMI'] >= 18.5) & (df_feat['BMI'] < 25.0)).astype(int)
    df_feat['BMI_Overweight'] = ((df_feat['BMI'] >= 25.0) & (df_feat['BMI'] < 30.0)).astype(int)
    df_feat['BMI_Obese'] = (df_feat['BMI'] >= 30.0).astype(int)
    
    # 2. Glucose Categories
    df_feat['Glucose_Normal'] = (df_feat['Glucose'] < 100).astype(int)
    df_feat['Glucose_Prediabetes'] = ((df_feat['Glucose'] >= 100) & (df_feat['Glucose'] <= 125)).astype(int)
    df_feat['Glucose_Diabetes'] = (df_feat['Glucose'] > 125).astype(int)
    
    # 3. Age Groups
    df_feat['Age_Young'] = (df_feat['Age'] < 30).astype(int)
    df_feat['Age_Middle'] = ((df_feat['Age'] >= 30) & (df_feat['Age'] <= 50)).astype(int)
    df_feat['Age_Senior'] = (df_feat['Age'] > 50).astype(int)
    
    # 4. Clinical Interactions
    df_feat['Insulin_Glucose_Ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + 1e-5)
    df_feat['Insulin_Resistance_Proxy'] = (df_feat['Glucose'] * df_feat['Insulin']) / 405.0
    df_feat['Pregnancy_Age_Risk'] = df_feat['Pregnancies'] / (df_feat['Age'] + 1e-5)
    df_feat['BMI_Age_Interaction'] = df_feat['BMI'] * df_feat['Age']
    
    # 5. Log Transformations
    df_feat['Log_Insulin'] = np.log1p(np.maximum(0, df_feat['Insulin']))
    df_feat['Log_DPF'] = np.log1p(np.maximum(0, df_feat['DiabetesPedigreeFunction']))
    
    return df_feat
