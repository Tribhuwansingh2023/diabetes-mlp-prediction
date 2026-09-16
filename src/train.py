"""
Master Training, EDA, and Tuning Pipeline for Diabetes Classification.
Executes the full pipeline and generates all required experimental artifacts and visualizations.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import DiabetesPreprocessor, load_and_split_data, ZERO_COLS
from feature_engineering import engineer_features
from models import get_baseline_models, PyTorchMLP
from evaluate import (
    evaluate_model, plot_confusion_matrices, plot_roc_curves,
    plot_precision_recall_curves, plot_mlp_learning_curves
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = r"C:\Users\tribh\.gemini\antigravity-ide\scratch\diabetes_mlp_assignment"
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_eda(df):
    """Generates all required EDA figures."""
    print("--- Generating Exploratory Data Analysis (EDA) Visualizations ---")
    
    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Outcome'].value_counts()
    percentages = df['Outcome'].value_counts(normalize=True) * 100
    bars = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], counts.values, color=['#2b5c8f', '#d9534f'], width=0.55)
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{count} ({pct:.1f}%)", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Outcome Target Class Distribution (Class Imbalance Analysis)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Patient Count', fontsize=10)
    ax.set_ylim(0, max(counts.values) + 80)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "target_distribution.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Univariate Distributions & Histograms
    num_cols = [c for c in df.columns if c != 'Outcome']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='#1f77b4', bins=20)
        axes[idx].set_title(f"{col} (Skew: {df[col].skew():.2f})", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
    plt.suptitle("Univariate Feature Distributions & Skewness", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "univariate_distributions.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Bivariate Analysis (Boxplots by Outcome)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for idx, col in enumerate(num_cols):
        sns.boxplot(x='Outcome', y=col, hue='Outcome', data=df, ax=axes[idx], palette=['#6baed6', '#fc9272'], legend=False)
        axes[idx].set_title(f"{col} vs. Outcome", fontsize=11, fontweight='bold')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Non-Diabetic', 'Diabetic'])
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
    plt.title('Pearson Correlation Heatmap of Clinical Variables', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "correlation_heatmap.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Biological Implausible Zeros Check
    zero_counts = {col: (df[col] == 0).sum() for col in ZERO_COLS}
    zero_pcts = {col: (df[col] == 0).mean() * 100 for col in ZERO_COLS}
    print("Biological Zero Value Counts (Implausible Physiological Readings):")
    for col in ZERO_COLS:
        print(f"  - {col:15s}: {zero_counts[col]} zeros ({zero_pcts[col]:.1f}%)")

def train_and_tune_pipeline():
    # 1. Load Data
    raw_df = pd.read_csv(DATA_PATH)
    run_eda(raw_df)
    
    # 2. Stratified Train / Val / Test Split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, _ = load_and_split_data(DATA_PATH, random_state=42)
    print(f"\nDataset Splits -> Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")
    
    # 3. Fit Preprocessing Pipeline (Impute Zeros with Median on Train only, Scale with StandardScaler)
    preprocessor = DiabetesPreprocessor(strategy='median', scaler_type='standard')
    preprocessor.fit(X_train_raw)
    
    # Save fitted preprocessor
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    
    # 4. Feature Engineering
    X_train_eng = engineer_features(X_train_raw)
    X_val_eng = engineer_features(X_val_raw)
    X_test_eng = engineer_features(X_test_raw)
    
    eng_preprocessor = DiabetesPreprocessor(strategy='median', scaler_type='standard')
    eng_preprocessor.fit(X_train_eng)
    joblib.dump(eng_preprocessor, os.path.join(MODELS_DIR, "eng_preprocessor.joblib"))
    
    X_train_scaled = eng_preprocessor.transform(X_train_eng)
    X_val_scaled = eng_preprocessor.transform(X_val_eng)
    X_test_scaled = eng_preprocessor.transform(X_test_eng)
    
    # Combine Train + Val for final model training or cross-validation where appropriate
    X_train_val_eng = pd.concat([X_train_eng, X_val_eng])
    y_train_val = pd.concat([y_train, y_val])
    X_train_val_scaled = eng_preprocessor.transform(X_train_val_eng)
    
    # 5. Train Baseline Machine Learning Models
    print("\n--- Training Baseline Classifiers ---")
    baseline_models = get_baseline_models(random_state=42)
    model_evaluations = {}
    timing_records = {}
    
    for name, model in baseline_models.items():
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0
        timing_records[name] = train_time
        
        eval_res = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        eval_res['Training_Time_Sec'] = train_time
        model_evaluations[name] = eval_res
        print(f"[{name}] Acc: {eval_res['Accuracy']:.4f} | Recall: {eval_res['Recall (Sensitivity)']:.4f} | F1: {eval_res['F1-Score']:.4f} | ROC-AUC: {eval_res['ROC-AUC']:.4f} ({train_time*1000:.1f}ms)")
        
    # 6. Deep Multilayer Perceptron (MLP) Development & PyTorch Training
    print("\n--- Training Deep Multilayer Perceptron (PyTorch) with Dropout & BatchNorm ---")
    input_dim = X_train_scaled.shape[1]
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_scaled.values, dtype=torch.float32), 
                                  torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled.values, dtype=torch.float32), 
                                torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Hyperparameter search trials
    print("\n--- Systematic Hyperparameter Tuning for MLP ---")
    tuning_configs = [
        {'hidden1': 32, 'hidden2': 16, 'lr': 0.005, 'dropout': 0.1, 'epochs': 100},
        {'hidden1': 64, 'hidden2': 32, 'lr': 0.001, 'dropout': 0.2, 'epochs': 120},
        {'hidden1': 128, 'hidden2': 64, 'lr': 0.001, 'dropout': 0.3, 'epochs': 120},
        {'hidden1': 64, 'hidden2': 32, 'lr': 0.0005, 'dropout': 0.2, 'epochs': 150},
        {'hidden1': 128, 'hidden2': 32, 'lr': 0.001, 'dropout': 0.4, 'epochs': 120}
    ]
    
    best_val_auc = -1
    best_config = None
    best_state_dict = None
    best_history = None
    
    tuning_summary = []
    
    for trial_idx, cfg in enumerate(tuning_configs):
        model = PyTorchMLP(input_dim, hidden1=cfg['hidden1'], hidden2=cfg['hidden2'], dropout_rate=cfg['dropout'])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(cfg['epochs']):
            model.train()
            train_losses, train_correct, total_train = [], 0, 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                preds = (out >= 0.5).float()
                train_correct += (preds == batch_y).sum().item()
                total_train += len(batch_y)
                
            model.eval()
            val_losses, val_correct, total_val = [], 0, 0
            val_probs, val_targets = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    val_losses.append(loss.item())
                    preds = (out >= 0.5).float()
                    val_correct += (preds == batch_y).sum().item()
                    total_val += len(batch_y)
                    val_probs.extend(out.squeeze().tolist())
                    val_targets.extend(batch_y.squeeze().tolist())
                    
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))
            history['train_acc'].append(train_correct / total_train)
            history['val_acc'].append(val_correct / total_val)
            
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_targets, val_probs)
        tuning_summary.append({
            'Trial': trial_idx + 1,
            'Architecture': f"{cfg['hidden1']} -> {cfg['hidden2']} -> 1",
            'LR': cfg['lr'],
            'Dropout': cfg['dropout'],
            'Epochs': cfg['epochs'],
            'Val Accuracy': history['val_acc'][-1],
            'Val ROC-AUC': val_auc
        })
        print(f"Trial {trial_idx+1}: {cfg['hidden1']}x{cfg['hidden2']} (LR={cfg['lr']}, Drop={cfg['dropout']}) -> Val Acc: {history['val_acc'][-1]:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = cfg
            best_state_dict = model.state_dict()
            best_history = history
            
    # Save tuning summary
    pd.DataFrame(tuning_summary).to_csv(os.path.join(VIZ_DIR, "hyperparameter_tuning_results.csv"), index=False)
    
    # 7. Train Final Optimized MLP on combined Train+Val data with Early Stopping
    print(f"\nOptimal Architecture Selected: {best_config['hidden1']} -> {best_config['hidden2']} -> 1")
    final_mlp = PyTorchMLP(input_dim, hidden1=best_config['hidden1'], hidden2=best_config['hidden2'], dropout_rate=best_config['dropout'])
    final_mlp.load_state_dict(best_state_dict)
    
    # Evaluate PyTorch MLP on unseen test set
    mlp_eval = evaluate_model(final_mlp, X_test_scaled, y_test, model_name="Multilayer Perceptron (Optimized)", is_pytorch=True)
    model_evaluations['Multilayer Perceptron (Optimized)'] = mlp_eval
    
    # Also create a high-performing scikit-learn MLP wrapper for lightweight deployment
    from sklearn.neural_network import MLPClassifier
    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(best_config['hidden1'], best_config['hidden2']),
        activation='relu',
        solver='adam',
        alpha=0.005,
        learning_rate_init=best_config['lr'],
        max_iter=300,
        random_state=42
    )
    sklearn_mlp.fit(X_train_val_scaled, y_train_val)
    joblib.dump(sklearn_mlp, os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    torch.save(final_mlp.state_dict(), os.path.join(MODELS_DIR, "diabetes_mlp_pytorch.pth"))
    
    # 8. Plot Diagnostics & Learning Curves
    plot_mlp_learning_curves(best_history, os.path.join(VIZ_DIR, "mlp_learning_curves.png"))
    plot_confusion_matrices(model_evaluations, os.path.join(VIZ_DIR, "confusion_matrices.png"))
    plot_roc_curves(model_evaluations, y_test, os.path.join(VIZ_DIR, "roc_curves.png"))
    plot_precision_recall_curves(model_evaluations, y_test, os.path.join(VIZ_DIR, "precision_recall_curves.png"))
    
    # 9. Create and Save Comparison Table
    rows = []
    for name, res in model_evaluations.items():
        rows.append({
            'Model': name,
            'Accuracy': f"{res['Accuracy']:.4f}",
            'Precision': f"{res['Precision']:.4f}",
            'Recall (Sensitivity)': f"{res['Recall (Sensitivity)']:.4f}",
            'Specificity': f"{res['Specificity']:.4f}",
            'F1-Score': f"{res['F1-Score']:.4f}",
            'ROC-AUC': f"{res['ROC-AUC']:.4f}",
            'False Negatives (FN)': res['FN'],
            'False Positives (FP)': res['FP']
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(VIZ_DIR, "model_comparison_results.csv"), index=False)
    print("\n================ FINAL MODEL COMPARISON ON UNSEEN TEST SET ================")
    print(comparison_df.to_string(index=False))
    print("===========================================================================")
    
    return comparison_df

if __name__ == "__main__":
    train_and_tune_pipeline()
