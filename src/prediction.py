"""
Authoritative Prediction and Inference Engine for Diabetes Risk Classification.
Provides unified inference pipeline used by Streamlit, unit tests, and batch simulators.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from feature_engineering import RAW_FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

# Acceptable physiological parameter validation boundaries
PHYSIOLOGICAL_BOUNDS = {
    'Pregnancies': (0, 25),
    'Glucose': (0, 400),
    'BloodPressure': (0, 200),
    'SkinThickness': (0, 120),
    'Insulin': (0, 1000),
    'BMI': (0.0, 80.0),
    'DiabetesPedigreeFunction': (0.0, 3.5),
    'Age': (1, 120)
}

def validate_patient_input(patient_dict: dict) -> pd.DataFrame:
    """
    Validates patient input dictionary against required fields and physical boundaries.
    Returns single-row DataFrame.
    """
    if not isinstance(patient_dict, dict):
        raise TypeError(f"Expected dict, got {type(patient_dict)}")
        
    missing_keys = [k for k in RAW_FEATURE_NAMES if k not in patient_dict]
    if missing_keys:
        raise ValueError(f"Missing required raw feature keys: {missing_keys}")
        
    validated_dict = {}
    for col in RAW_FEATURE_NAMES:
        val = patient_dict[col]
        try:
            val_num = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{col}' must be numeric, got: {val}")
            
        min_b, max_b = PHYSIOLOGICAL_BOUNDS[col]
        if val_num < min_b or val_num > max_b:
            raise ValueError(f"Feature '{col}' value {val_num} out of valid physiological range [{min_b}, {max_b}]")
            
        validated_dict[col] = [val_num]
        
    return pd.DataFrame(validated_dict, columns=RAW_FEATURE_NAMES)

def load_artifacts():
    """
    Loads saved model, preprocessor, and metadata configuration.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor artifact not found at: {PREPROCESSOR_PATH}")
        
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
    # Verify compatibility
    if hasattr(preprocessor, 'n_features_out_') and hasattr(model, 'n_features_in_'):
        if preprocessor.n_features_out_ != model.n_features_in_:
            raise ValueError(
                f"Dimension mismatch: preprocessor outputs {preprocessor.n_features_out_} features, "
                f"but model expects {model.n_features_in_} features."
            )
            
    return model, preprocessor, config

def predict_patient(patient_data, threshold=0.50):
    """
    Authoritative inference function.
    
    Parameters:
    -----------
    patient_data : dict or pd.DataFrame
        Patient biometrics (8 raw features).
    threshold : float
        Decision threshold for positive class (default 0.50).
        
    Returns:
    --------
    dict containing:
      - 'predicted_class': int (0 or 1)
      - 'probability': float in [0.0, 1.0]
      - 'raw_input': pd.DataFrame
      - 'processed_features': pd.DataFrame
      - 'processed_feature_names': list of strings
      - 'raw_feature_count': int
      - 'processed_feature_count': int
      - 'threshold': float
    """
    if isinstance(patient_data, dict):
        df_raw = validate_patient_input(patient_data)
    elif isinstance(patient_data, pd.DataFrame):
        df_raw = patient_data[RAW_FEATURE_NAMES].copy()
    else:
        raise TypeError("patient_data must be a dict or pd.DataFrame")
        
    model, preprocessor, config = load_artifacts()
    
    # Process through authoritative pipeline (zero handling -> imputation -> feature engineering -> scaling)
    df_processed = preprocessor.transform(df_raw)
    
    # Generate probability from actual model
    probabilities = model.predict_proba(df_processed)[:, 1]
    prob = float(probabilities[0])
    pred = int(prob >= threshold)
    
    return {
        'predicted_class': pred,
        'probability': prob,
        'raw_input': df_raw,
        'processed_features': df_processed,
        'processed_feature_names': list(df_processed.columns),
        'raw_feature_count': int(df_raw.shape[1]),
        'processed_feature_count': int(df_processed.shape[1]),
        'threshold': float(threshold),
        'config': config
    }
