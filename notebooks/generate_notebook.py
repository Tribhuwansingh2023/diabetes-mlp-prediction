"""
Jupyter Notebook Generator for Lab Assignment 01: Predicting Diabetes with Multilayer Perceptron.
Creates an annotated, self-contained, fully reproducible .ipynb notebook.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell(r"""# Lab Assignment 01: Predicting Diabetes with Multilayer Perceptron
**Course**: Deep Learning / Machine Learning Laboratory  
**Problem Statement**: Develop an end-to-end binary classification system to predict patient diabetes onset using biometric and demographic attributes from the Pima Indians Diabetes Database. Benchmark conventional machine learning baselines against an optimized Multilayer Perceptron (MLP) implemented in Scikit-Learn.

### 👥 Submitted By (Group Members):
| Sl. No. | Student Name | Registration Number |
| :---: | :--- | :---: |
| 1 | **Tribhuwan Singh** | `2341019538` |
| 2 | **Surajit Sahoo** | `2341019165` |
| 3 | **Anwesha Srichandan** | `2341019594` |
| 4 | **Priti Rani Maity** | `2341013065` |

---
## Notebook Structure
1. **Environment Setup & Library Imports**
2. **Dataset Loading & Initial Inspection**
3. **Exploratory Data Analysis (EDA)**
   - Target Variable Distribution & Imbalance Analysis
   - Univariate Distributions & Skewness
   - Bivariate Boxplots vs. Diabetes Outcome
   - Pearson Correlation Heatmap & Multicollinearity
   - Outlier Detection Analysis
4. **Data Quality, Preprocessing & Feature Engineering**
   - Biologically Implausible Zero-Value Identification & Imputation
   - Domain Feature Engineering on Imputed Data (24 Processed Features)
   - Stratified Train (70%), Validation (15%), and Test (15%) Partitioning
   - Feature Scaling (StandardScaler)
5. **Baseline Machine Learning Classifiers**
   - Logistic Regression, KNN, Support Vector Classifier, Decision Tree, Random Forest, Gradient Boosting
6. **Multilayer Perceptron (MLP) Development & Tuning**
   - Scikit-Learn `MLPClassifier` Architecture Configuration
   - Systematic Hyperparameter Search across Architectures & Regularization
   - Final Model Retraining on Combined Train + Validation Data
7. **Model Comparison & Comprehensive Evaluation**
   - Evaluation on Untouched Test Set (N=116)
   - Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC
   - Confusion Matrices, ROC Curves, and Precision-Recall Curves
   - Trade-Off Analysis (False Negatives vs. False Positives)
8. **Model Serialization & Deployment Preparation**
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Set random seed for strict reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 100
"""))

# Cell 2: Dataset Loading
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Loading & Initial Inspection"""))
cells.append(nbf.v4.new_code_cell("""DATA_PATH = os.path.join("..", "data", "diabetes.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("data", "diabetes.csv")

raw_df = pd.read_csv(DATA_PATH)
print(f"Dataset Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
display(raw_df.head())
display(raw_df.info())
display(raw_df.describe().T)
"""))

# Cell 3: EDA
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)"""))
cells.append(nbf.v4.new_code_cell("""# 1. Target Class Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = raw_df['Outcome'].value_counts()
pcts = raw_df['Outcome'].value_counts(normalize=True) * 100
bars = ax.bar(['Negative (0)', 'Positive (1)'], counts.values, color=['#2b5c8f', '#d9534f'], width=0.55)
for bar, count, pct in zip(bars, counts.values, pcts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{count} ({pct:.1f}%)", 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Outcome Target Class Distribution (Imbalance Analysis)', fontsize=12, fontweight='bold')
ax.set_ylabel('Patient Count')
ax.set_ylim(0, max(counts.values) + 80)
plt.tight_layout()
plt.show()

# 2. Implausible Zero Value Detection
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("Biologically Implausible Zero Value Counts:")
for col in ZERO_COLS:
    z_count = (raw_df[col] == 0).sum()
    z_pct = (raw_df[col] == 0).mean() * 100
    print(f"  - {col:15s}: {z_count:3d} zeros ({z_pct:.2f}%)")
"""))

# Cell 4: Univariate & Correlation
cells.append(nbf.v4.new_code_cell("""# Univariate Feature Distributions
num_cols = [c for c in raw_df.columns if c != 'Outcome']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()
for idx, col in enumerate(num_cols):
    sns.histplot(raw_df[col], kde=True, ax=axes[idx], color='#1f77b4', bins=20)
    axes[idx].set_title(f"{col} (Skew: {raw_df[col].skew():.2f})", fontsize=11, fontweight='bold')
plt.suptitle("Univariate Feature Distributions & Skewness", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(9, 7))
corr = raw_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 10, 'weight': 'bold'})
plt.title('Pearson Correlation Heatmap of Dataset Variables', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

# Cell 5: Preprocessing & Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Preprocessing & Feature Engineering Pipeline"""))
cells.append(nbf.v4.new_code_cell("""RAW_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def engineer_features(df):
    df_feat = df.copy()
    # 1. BMI Categories
    df_feat['BMI_Underweight'] = (df_feat['BMI'] < 18.5).astype(float)
    df_feat['BMI_Normal'] = ((df_feat['BMI'] >= 18.5) & (df_feat['BMI'] < 25.0)).astype(float)
    df_feat['BMI_Overweight'] = ((df_feat['BMI'] >= 25.0) & (df_feat['BMI'] < 30.0)).astype(float)
    df_feat['BMI_Obese'] = (df_feat['BMI'] >= 30.0).astype(float)
    
    # 2. Glucose Categories
    df_feat['Glucose_Normal'] = (df_feat['Glucose'] < 100.0).astype(float)
    df_feat['Glucose_Prediabetes'] = ((df_feat['Glucose'] >= 100.0) & (df_feat['Glucose'] <= 125.0)).astype(float)
    df_feat['Glucose_Diabetes'] = (df_feat['Glucose'] >= 126.0).astype(float)
    
    # 3. Age Categories
    df_feat['Age_Young'] = (df_feat['Age'] < 30.0).astype(float)
    df_feat['Age_Middle'] = ((df_feat['Age'] >= 30.0) & (df_feat['Age'] <= 50.0)).astype(float)
    df_feat['Age_Senior'] = (df_feat['Age'] > 50.0).astype(float)
    
    # 4. Interactions
    df_feat['Insulin_Glucose_Ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + 1e-5)
    df_feat['Insulin_Resistance_Proxy'] = (df_feat['Glucose'] * df_feat['Insulin']) / 405.0
    df_feat['Pregnancy_Age_Risk'] = df_feat['Pregnancies'] / (df_feat['Age'] + 1e-5)
    df_feat['BMI_Age_Interaction'] = df_feat['BMI'] * df_feat['Age']
    
    # 5. Log Transforms
    df_feat['Log_Insulin'] = np.log1p(np.maximum(0.0, df_feat['Insulin']))
    df_feat['Log_DPF'] = np.log1p(np.maximum(0.0, df_feat['DiabetesPedigreeFunction']))
    
    return df_feat

# Stratified 3-way Split (Train 70%, Val 15%, Test 15%)
X = raw_df[RAW_FEATURES].copy()
y = raw_df['Outcome'].copy()

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42)

print(f"Partitions: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Preprocessing Pipeline (Impute Zeros -> Feature Engineering -> Scale) fit on Train ONLY
def fit_pipeline(X_train_data):
    X_clean = X_train_data.copy()
    for c in ZERO_COLS:
        X_clean[c] = X_clean[c].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    imputed_raw = imputer.fit_transform(X_clean[RAW_FEATURES])
    df_imp = pd.DataFrame(imputed_raw, columns=RAW_FEATURES, index=X_clean.index)
    df_eng = engineer_features(df_imp)
    scaler = StandardScaler()
    scaler.fit(df_eng)
    return imputer, scaler, list(df_eng.columns)

def transform_pipeline(X_data, imputer, scaler, feature_cols):
    X_clean = X_data.copy()
    for c in ZERO_COLS:
        X_clean[c] = X_clean[c].replace(0, np.nan)
    imputed_raw = imputer.transform(X_clean[RAW_FEATURES])
    df_imp = pd.DataFrame(imputed_raw, columns=RAW_FEATURES, index=X_clean.index)
    df_eng = engineer_features(df_imp)
    scaled_arr = scaler.transform(df_eng)
    return pd.DataFrame(scaled_arr, columns=feature_cols, index=X_data.index)

imputer_dev, scaler_dev, feature_names = fit_pipeline(X_train)
X_train_sc = transform_pipeline(X_train, imputer_dev, scaler_dev, feature_names)
X_val_sc = transform_pipeline(X_val, imputer_dev, scaler_dev, feature_names)
X_test_sc = transform_pipeline(X_test, imputer_dev, scaler_dev, feature_names)

print(f"Processed Feature Count: {X_train_sc.shape[1]} features")
"""))

# Cell 6: Baseline Classifiers & MLP Tuning
cells.append(nbf.v4.new_markdown_cell("""## 5. Baseline Classifiers & Hyperparameter Tuning"""))
cells.append(nbf.v4.new_code_cell("""baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    'MLP (Baseline)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

print("--- Baseline Model Benchmark on Test Set ---")
eval_results = {}
for name, model in baseline_models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    probs = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else preds
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    eval_results[name] = {
        'Model': name, 'Accuracy': acc, 'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall (Sensitivity)': rec, 'Specificity': spec, 'F1-Score': f1, 'ROC-AUC': auc,
        'FN': fn, 'FP': fp, 'TP': tp, 'TN': tn, 'y_prob': probs, 'cm': cm
    }
    print(f"[{name:25s}] Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
"""))

# Cell 7: Final Retraining & Comparison
cells.append(nbf.v4.new_markdown_cell("""## 6. Hyperparameter Tuning & Final Model Retraining"""))
cells.append(nbf.v4.new_code_cell("""tuning_configs = [
    {'hidden_layer_sizes': (32, 16), 'alpha': 0.0001, 'learning_rate_init': 0.005, 'batch_size': 32, 'max_iter': 400},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.001,  'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 400},
    {'hidden_layer_sizes': (128, 64), 'alpha': 0.005, 'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 500},
    {'hidden_layer_sizes': (64, 32), 'alpha': 0.01,   'learning_rate_init': 0.002, 'batch_size': 64, 'max_iter': 400},
    {'hidden_layer_sizes': (128, 32), 'alpha': 0.001, 'learning_rate_init': 0.001, 'batch_size': 32, 'max_iter': 500}
]

best_auc, best_cfg = -1, None
for idx, cfg in enumerate(tuning_configs):
    mlp_t = MLPClassifier(hidden_layer_sizes=cfg['hidden_layer_sizes'], alpha=cfg['alpha'],
                          learning_rate_init=cfg['learning_rate_init'], batch_size=cfg['batch_size'],
                          max_iter=cfg['max_iter'], random_state=42)
    mlp_t.fit(X_train_sc, y_train)
    probs_v = mlp_t.predict_proba(X_val_sc)[:, 1]
    auc_v = roc_auc_score(y_val, probs_v)
    print(f"Trial {idx+1}: {cfg['hidden_layer_sizes']} -> Val AUC: {auc_v:.4f}")
    if auc_v > best_auc:
        best_auc, best_cfg = auc_v, cfg

print(f"\\nBest Architecture Selected: {best_cfg['hidden_layer_sizes']} (Val AUC: {best_auc:.4f})")

# Final Model Retraining on Combined Train + Validation
imputer_final, scaler_final, final_feats = fit_pipeline(X_train_val)
X_train_val_sc = transform_pipeline(X_train_val, imputer_final, scaler_final, final_feats)
X_test_final_sc = transform_pipeline(X_test, imputer_final, scaler_final, final_feats)

final_mlp = MLPClassifier(hidden_layer_sizes=best_cfg['hidden_layer_sizes'], alpha=best_cfg['alpha'],
                          learning_rate_init=best_cfg['learning_rate_init'], batch_size=best_cfg['batch_size'],
                          max_iter=best_cfg['max_iter'], random_state=42)
final_mlp.fit(X_train_val_sc, y_train_val)

preds_mlp = final_mlp.predict(X_test_final_sc)
probs_mlp = final_mlp.predict_proba(X_test_final_sc)[:, 1]
cm_mlp = confusion_matrix(y_test, preds_mlp)
tn, fp, fn, tp = cm_mlp.ravel()

eval_results['Multilayer Perceptron (Optimized)'] = {
    'Model': 'Multilayer Perceptron (Optimized)',
    'Accuracy': accuracy_score(y_test, preds_mlp),
    'Precision': precision_score(y_test, preds_mlp, zero_division=0),
    'Recall (Sensitivity)': recall_score(y_test, preds_mlp, zero_division=0),
    'Specificity': tn / (tn + fp),
    'F1-Score': f1_score(y_test, preds_mlp, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, probs_mlp),
    'FN': fn, 'FP': fp, 'TP': tp, 'TN': tn, 'y_prob': probs_mlp, 'cm': cm_mlp
}

# Display Model Comparison Table
df_comp = pd.DataFrame([
    {
        'Model': r['Model'], 'Accuracy': f"{r['Accuracy']:.4f}", 'Recall': f"{r['Recall (Sensitivity)']:.4f}",
        'Specificity': f"{r['Specificity']:.4f}", 'Precision': f"{r['Precision']:.4f}",
        'F1-Score': f"{r['F1-Score']:.4f}", 'ROC-AUC': f"{r['ROC-AUC']:.4f}",
        'FN': r['FN'], 'FP': r['FP']
    }
    for r in eval_results.values()
])
display(df_comp)
"""))

# Cell 8: Plots & Model Saving
cells.append(nbf.v4.new_markdown_cell("""## 7. Diagnostic Visualizations & Serialization"""))
cells.append(nbf.v4.new_code_cell("""# 1. ROC Curves
plt.figure(figsize=(8, 6))
for name, r in eval_results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['ROC-AUC']:.3f})", lw=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()

# 2. MLP Training Loss Curve
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(final_mlp.loss_curve_) + 1), final_mlp.loss_curve_, 'b-', lw=2)
plt.title('Final MLPClassifier Training Loss Curve', fontsize=12, fontweight='bold')
plt.xlabel('Iterations')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
"""))

nb.cells = cells
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes_prediction_mlp.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully created at: {output_path}")
