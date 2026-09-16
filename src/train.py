"""
Master Training, EDA, and Hyperparameter Tuning Pipeline for Diabetes Classification.
Executes the authoritative machine learning pipeline:
1. Exploratory Data Analysis & visual asset generation
2. Stratified Train (70%), Validation (15%), and Test (15%) partitioning
3. Preprocessing (median imputation + feature engineering + standardization) fit on Train
4. Baseline model benchmarking
5. Systematic Scikit-Learn MLPClassifier hyperparameter tuning
6. Final model retraining on combined Train + Validation
7. Final single evaluation on untouched Test set
8. Model persistence and metadata serialization
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocessing import DiabetesPreprocessor, load_and_split_data, ZERO_COLS
from feature_engineering import RAW_FEATURE_NAMES, ENGINEERED_FEATURE_NAMES, ALL_FEATURE_NAMES
from models import get_baseline_models, build_mlp_classifier
from evaluate import (
    evaluate_model, plot_confusion_matrices, plot_roc_curves,
    plot_precision_recall_curves, plot_mlp_learning_curves
)

# Set random seeds for strict reproducibility
np.random.seed(42)

DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_eda(df: pd.DataFrame):
    """Generates all exploratory data analysis visualizations."""
    print("\n--- 1. Generating Exploratory Data Analysis (EDA) Visualizations ---")
    
    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Outcome'].value_counts()
    percentages = df['Outcome'].value_counts(normalize=True) * 100
    bars = ax.bar(['Negative (0)', 'Positive (1)'], counts.values, color=['#2b5c8f', '#d9534f'], width=0.55)
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{count} ({pct:.1f}%)", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Outcome Target Class Distribution (Imbalance Analysis)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Patient Count', fontsize=10)
    ax.set_ylim(0, max(counts.values) + 80)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "target_distribution.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Univariate Distributions
    num_cols = [c for c in df.columns if c != 'Outcome']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='#1f77b4', bins=20)
        axes[idx].set_title(f"{col} (Skew: {df[col].skew():.2f})", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(f"{col} (mm Hg)" if col == "BloodPressure" else col, fontsize=10)
    plt.suptitle("Univariate Feature Distributions & Skewness", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "univariate_distributions.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Bivariate Boxplots by Outcome
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for idx, col in enumerate(num_cols):
        sns.boxplot(x='Outcome', y=col, hue='Outcome', data=df, ax=axes[idx], palette=['#6baed6', '#fc9272'], legend=False)
        axes[idx].set_title(f"{col} vs. Outcome", fontsize=11, fontweight='bold')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Negative (0)', 'Positive (1)'])
    plt.suptitle("Bivariate Boxplots: Clinical Predictors vs. Diabetes Outcome", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "bivariate_boxplots.png"), bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 10, 'weight': 'bold'})
    plt.title('Pearson Correlation Heatmap of Dataset Variables', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "correlation_heatmap.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Zero Value Summary
    zero_counts = {col: int((df[col] == 0).sum()) for col in ZERO_COLS}
    zero_pcts = {col: float((df[col] == 0).mean() * 100) for col in ZERO_COLS}
    print("Biological Zero Value Counts (Implausible Physiological Readings):")
    for col in ZERO_COLS:
        print(f"  - {col:15s}: {zero_counts[col]} zeros ({zero_pcts[col]:.1f}%)")

def train_and_tune_pipeline():
    # Step 1: Load and Split Data
    X_train, X_val, X_test, y_train, y_val, y_test, raw_df = load_and_split_data(
        DATA_PATH, test_size=0.15, val_size=0.15, random_state=42
    )
    run_eda(raw_df)
    
    print(f"\n--- 2. Dataset Partitions ---")
    print(f"Train Set:      {len(X_train)} samples ({len(X_train)/len(raw_df)*100:.1f}%)")
    print(f"Validation Set: {len(X_val)} samples ({len(X_val)/len(raw_df)*100:.1f}%)")
    print(f"Test Set:       {len(X_test)} samples ({len(X_test)/len(raw_df)*100:.1f}%) [Kept strictly untouched]")
    
    # Step 2: Fit Preprocessor on Training Data ONLY
    print(f"\n--- 3. Preprocessing Pipeline ---")
    preprocessor = DiabetesPreprocessor(strategy='median')
    preprocessor.fit(X_train)
    
    X_train_scaled = preprocessor.transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    processed_dim = X_train_scaled.shape[1]
    print(f"Raw Features:       {len(RAW_FEATURE_NAMES)} {RAW_FEATURE_NAMES}")
    print(f"Engineered Features:{len(ENGINEERED_FEATURE_NAMES)} {ENGINEERED_FEATURE_NAMES}")
    print(f"Processed Features: {processed_dim} total features (Calculated dynamically)")
    
    # Step 3: Train & Benchmark Baseline Classifiers
    print(f"\n--- 4. Benchmarking Baseline Classifiers (Evaluated on Test Set) ---")
    baseline_models = get_baseline_models(random_state=42)
    model_evaluations = {}
    timing_records = {}
    
    for name, model in baseline_models.items():
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        t_train = time.time() - t0
        timing_records[name] = t_train
        
        eval_res = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        eval_res['Training_Time_Sec'] = t_train
        model_evaluations[name] = eval_res
        print(f"[{name:28s}] Acc: {eval_res['Accuracy']:.4f} | Recall: {eval_res['Recall (Sensitivity)']:.4f} | F1: {eval_res['F1-Score']:.4f} | ROC-AUC: {eval_res['ROC-AUC']:.4f} ({t_train*1000:.1f}ms)")
        
    # Step 4: Systematic Hyperparameter Tuning for Scikit-Learn MLPClassifier
    print(f"\n--- 5. Systematic Hyperparameter Tuning for Scikit-Learn MLPClassifier ---")
    tuning_grid = [
        {'hidden_layer_sizes': (32, 16), 'alpha': 0.0001, 'learning_rate_init': 0.005, 'batch_size': 32, 'max_iter': 400},
        {'hidden_layer_sizes': (64, 32), 'alpha': 0.001,  'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 400},
        {'hidden_layer_sizes': (128, 64), 'alpha': 0.005, 'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 500},
        {'hidden_layer_sizes': (64, 32), 'alpha': 0.01,   'learning_rate_init': 0.002, 'batch_size': 64, 'max_iter': 400},
        {'hidden_layer_sizes': (128, 32), 'alpha': 0.001, 'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 500},
        {'hidden_layer_sizes': (64, 32, 16), 'alpha': 0.005, 'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 400}
    ]
    
    best_val_auc = -1.0
    best_config = None
    tuning_records = []
    
    for trial_idx, cfg in enumerate(tuning_grid):
        mlp_trial = build_mlp_classifier(
            hidden_layer_sizes=cfg['hidden_layer_sizes'],
            activation='relu',
            learning_rate_init=cfg['learning_rate_init'],
            alpha=cfg['alpha'],
            batch_size=cfg['batch_size'],
            max_iter=cfg['max_iter'],
            early_stopping=False,
            random_state=42
        )
        
        t0 = time.time()
        mlp_trial.fit(X_train_scaled, y_train)
        t_trial = time.time() - t0
        
        # Evaluate on VALIDATION set for hyperparameter selection
        val_eval = evaluate_model(mlp_trial, X_val_scaled, y_val, model_name=f"MLP_Trial_{trial_idx+1}")
        
        record = {
            'Trial': trial_idx + 1,
            'Hidden_Layers': str(cfg['hidden_layer_sizes']),
            'Activation': 'relu',
            'Learning_Rate': cfg['learning_rate_init'],
            'Alpha': cfg['alpha'],
            'Batch_Size': cfg['batch_size'],
            'Max_Iter': cfg['max_iter'],
            'Val_Accuracy': val_eval['Accuracy'],
            'Val_Recall': val_eval['Recall (Sensitivity)'],
            'Val_F1': val_eval['F1-Score'],
            'Val_ROC_AUC': val_eval['ROC-AUC'],
            'Training_Time_Sec': round(t_trial, 4)
        }
        tuning_records.append(record)
        print(f"Trial {trial_idx+1}: Layers={cfg['hidden_layer_sizes']} | Alpha={cfg['alpha']} | LR={cfg['learning_rate_init']} -> Val Acc: {val_eval['Accuracy']:.4f}, Val AUC: {val_eval['ROC-AUC']:.4f} ({t_trial*1000:.1f}ms)")
        
        if val_eval['ROC-AUC'] > best_val_auc:
            best_val_auc = val_eval['ROC-AUC']
            best_config = cfg
            
    df_tuning = pd.DataFrame(tuning_records)
    df_tuning.to_csv(os.path.join(RESULTS_DIR, "hyperparameter_tuning_results.csv"), index=False)
    df_tuning.to_csv(os.path.join(VIZ_DIR, "hyperparameter_tuning_results.csv"), index=False)
    
    print(f"\n--- 6. Optimal Hyperparameters Selected based on Validation Performance ---")
    print(f"Optimal Architecture: {best_config['hidden_layer_sizes']}")
    print(f"Optimal Alpha (L2):   {best_config['alpha']}")
    print(f"Optimal Learning Rate:{best_config['learning_rate_init']}")
    print(f"Best Validation AUC:  {best_val_auc:.4f}")
    
    # Step 5: Final Model Retraining on Combined Train + Validation Data
    print(f"\n--- 7. Retraining Final Authoritative Model on Combined Train + Validation Data ---")
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    
    # Fit authoritative production preprocessor on all 85% development data
    final_preprocessor = DiabetesPreprocessor(strategy='median')
    final_preprocessor.fit(X_train_val)
    
    X_train_val_scaled = final_preprocessor.transform(X_train_val)
    X_test_final_scaled = final_preprocessor.transform(X_test)
    
    final_mlp = build_mlp_classifier(
        hidden_layer_sizes=best_config['hidden_layer_sizes'],
        activation='relu',
        learning_rate_init=best_config['learning_rate_init'],
        alpha=best_config['alpha'],
        batch_size=best_config['batch_size'],
        max_iter=best_config['max_iter'],
        early_stopping=False,
        random_state=42
    )
    t0 = time.time()
    final_mlp.fit(X_train_val_scaled, y_train_val)
    t_final = time.time() - t0
    
    # Step 6: Single Final Evaluation on Untouched Test Set
    print(f"\n--- 8. Single Final Evaluation on Untouched Test Set (N={len(X_test)}) ---")
    final_mlp_eval = evaluate_model(final_mlp, X_test_final_scaled, y_test, model_name="Multilayer Perceptron (Optimized)")
    final_mlp_eval['Training_Time_Sec'] = t_final
    model_evaluations['Multilayer Perceptron (Optimized)'] = final_mlp_eval
    
    # Step 7: Save Comparison Table & Visualizations
    comp_rows = []
    for name, res in model_evaluations.items():
        comp_rows.append({
            'Model': name,
            'Accuracy': round(res['Accuracy'], 4),
            'Precision': round(res['Precision'], 4),
            'Recall (Sensitivity)': round(res['Recall (Sensitivity)'], 4),
            'Specificity': round(res['Specificity'], 4),
            'F1-Score': round(res['F1-Score'], 4),
            'ROC-AUC': round(res['ROC-AUC'], 4),
            'False Negatives (FN)': res['FN'],
            'False Positives (FP)': res['FP'],
            'True Positives (TP)': res['TP'],
            'True Negatives (TN)': res['TN'],
            'Training_Time_Sec': round(res.get('Training_Time_Sec', 0.0), 4)
        })
    df_comparison = pd.DataFrame(comp_rows)
    df_comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    df_comparison.to_csv(os.path.join(VIZ_DIR, "model_comparison_results.csv"), index=False)
    
    # Step 8: Plot Diagnostic Figures
    plot_mlp_learning_curves(final_mlp, os.path.join(VIZ_DIR, "mlp_learning_curves.png"))
    plot_confusion_matrices(model_evaluations, os.path.join(VIZ_DIR, "confusion_matrices.png"))
    plot_roc_curves(model_evaluations, y_test, os.path.join(VIZ_DIR, "roc_curves.png"))
    plot_precision_recall_curves(model_evaluations, y_test, os.path.join(VIZ_DIR, "precision_recall_curves.png"))
    
    # Step 9: Serialize Authoritative Artifacts & Metadata Configuration
    joblib.dump(final_mlp, os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    joblib.dump(final_preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    
    model_config = {
        'framework': 'scikit-learn',
        'model_type': 'MLPClassifier',
        'raw_features': RAW_FEATURE_NAMES,
        'raw_feature_count': len(RAW_FEATURE_NAMES),
        'engineered_features': ENGINEERED_FEATURE_NAMES,
        'engineered_feature_count': len(ENGINEERED_FEATURE_NAMES),
        'processed_feature_count': int(X_train_val_scaled.shape[1]),
        'processed_feature_names': list(final_preprocessor.processed_feature_names),
        'hidden_layer_sizes': list(best_config['hidden_layer_sizes']),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': best_config['learning_rate_init'],
        'alpha': best_config['alpha'],
        'batch_size': best_config['batch_size'],
        'max_iter': best_config['max_iter'],
        'early_stopping': False,
        'threshold': 0.50,
        'random_state': 42,
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'test_metrics': {
            'accuracy': final_mlp_eval['Accuracy'],
            'precision': final_mlp_eval['Precision'],
            'recall': final_mlp_eval['Recall (Sensitivity)'],
            'specificity': final_mlp_eval['Specificity'],
            'f1_score': final_mlp_eval['F1-Score'],
            'roc_auc': final_mlp_eval['ROC-AUC'],
            'tp': final_mlp_eval['TP'],
            'tn': final_mlp_eval['TN'],
            'fp': final_mlp_eval['FP'],
            'fn': final_mlp_eval['FN']
        }
    }
    
    with open(os.path.join(MODELS_DIR, "model_config.json"), 'w') as f:
        json.dump(model_config, f, indent=4)
        
    with open(os.path.join(RESULTS_DIR, "final_metrics.json"), 'w') as f:
        json.dump(model_config['test_metrics'], f, indent=4)
        
    print(f"\n================ FINAL MODEL COMPARISON TABLE ================")
    print(df_comparison[['Model', 'Accuracy', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'ROC-AUC', 'False Negatives (FN)', 'False Positives (FP)']].to_string(index=False))
    print(f"==============================================================")
    print(f"\nArtifacts successfully saved to {MODELS_DIR}/ and {RESULTS_DIR}/")

if __name__ == "__main__":
    train_and_tune_pipeline()
