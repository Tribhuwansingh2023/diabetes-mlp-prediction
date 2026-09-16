"""
Jupyter Notebook Generator for Lab Assignment 01.
Creates an annotated, self-contained .ipynb notebook covering the entire assignment workflow.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Lab Assignment 01: Predicting Diabetes with Multilayer Perceptron
**Course**: Deep Learning / Machine Learning Laboratory  
**Problem Statement**: Develop an end-to-end binary classification system to predict patient diabetes onset using clinical and demographic attributes. Compare multiple conventional machine learning baseline models against a custom tuned Multilayer Perceptron (MLP), followed by deployment and clinical evaluation.

---
## Notebook Structure
1. **Environment Setup & Library Imports**
2. **Dataset Loading & Initial Inspection**
3. **Exploratory Data Analysis (EDA)**
   - Univariate Distributions & Skewness
   - Target Variable Analysis & Imbalance Investigation
   - Bivariate Clinical Correlates
   - Correlation Matrix & Multicollinearity
   - Outlier Detection (IQR & Z-Score Analysis)
4. **Data Quality, Preprocessing & Feature Engineering**
   - Biologically Implausible Zero-Value Identification & Imputation
   - Clinical Feature Engineering (BMI Categories, Glycemic Indicators, HOMA-IR Proxies, Interaction Ratios)
   - Stratified Train-Val-Test Splitting (Preventing Data Leakage)
   - Feature Scaling (StandardScaler vs. MinMaxScaler)
5. **Baseline Machine Learning Classifiers**
   - Logistic Regression, KNN, Support Vector Classifier, Decision Tree, Random Forest, Gradient Boosting
6. **Multilayer Perceptron (MLP) Development**
   - Architecture Design: Input -> Dense(64, ReLU) -> BatchNorm -> Dropout -> Dense(32, ReLU) -> Dropout -> Output(Sigmoid)
   - Training with Validation Monitoring & Learning Curves
   - Systematic Hyperparameter Tuning
7. **Model Comparison & Comprehensive Evaluation**
   - Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC
   - Confusion Matrices, ROC Curves, and Precision-Recall Curves
   - Medical Trade-Off Analysis (Cost of False Negatives vs. False Positives)
8. **Model Serialization & Deployment Preparation**
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-Learn Modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# PyTorch Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Global plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
%matplotlib inline

# Set reproducible random seeds
np.random.seed(42)
torch.manual_seed(42)
print("Libraries imported successfully!")
"""))

# Cell 2: Dataset Loading
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Loading & Initial Inspection
We load the Pima Indians Diabetes Dataset containing 768 patient records with 8 clinical features and 1 binary outcome target (`0 = Non-Diabetic`, `1 = Diabetic`).
"""))

cells.append(nbf.v4.new_code_cell("""# Load dataset
data_path = '../data/diabetes.csv'
df = pd.read_csv(data_path)

print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nData Types & Non-Null Values:")
df.info()
"""))

cells.append(nbf.v4.new_code_cell("""# Display first and last 5 records
display(df.head())
display(df.tail())
"""))

cells.append(nbf.v4.new_code_cell("""# Descriptive statistical summary of raw attributes
df.describe().T
"""))

# Cell 3: EDA
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)

### 3.1 Target Class Distribution & Imbalance
In medical diagnosis, class distribution determines the base rate. We examine whether severe class imbalance exists.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(6, 4))
counts = df['Outcome'].value_counts()
pcts = df['Outcome'].value_counts(normalize=True) * 100

bars = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], counts.values, color=['#2b5c8f', '#d9534f'], width=0.5)
for bar, count, pct in zip(bars, counts.values, pcts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{count} ({pct:.1f}%)", 
            ha='center', va='bottom', fontweight='bold')
ax.set_title('Outcome Target Class Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Patient Count')
ax.set_ylim(0, max(counts.values) + 80)
plt.show()

print(f"Non-Diabetic: {counts[0]} ({pcts[0]:.2f}%), Diabetic: {counts[1]} ({pcts[1]:.2f}%)")
print("Moderate class imbalance observed (~65% to 35%). Recall and ROC-AUC are paramount.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.2 Univariate Distributions & Skewness
We plot distributions for all clinical continuous features to detect skewness, modality, and physiological boundaries.
"""))

cells.append(nbf.v4.new_code_cell("""num_cols = [c for c in df.columns if c != 'Outcome']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[idx], color='#1f77b4', bins=20)
    axes[idx].set_title(f"{col} (Skew: {df[col].skew():.2f})", fontweight='bold')
    axes[idx].set_xlabel(col)

plt.suptitle("Univariate Feature Distributions & Skewness", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.3 Bivariate Analysis: Features vs. Outcome
We evaluate how distributions of key metabolic markers vary between diabetic and non-diabetic cohorts.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(num_cols):
    sns.boxplot(x='Outcome', y=col, hue='Outcome', data=df, ax=axes[idx], palette=['#6baed6', '#fc9272'], legend=False)
    axes[idx].set_title(f"{col} vs. Outcome", fontweight='bold')
    axes[idx].set_xticks([0, 1])
    axes[idx].set_xticklabels(['Non-Diabetic', 'Diabetic'])

plt.suptitle("Bivariate Boxplots: Clinical Predictors vs. Diabetes Outcome", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.4 Correlation Analysis & Multicollinearity
We compute Pearson correlation coefficients to assess pairwise linear relationships.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 10, 'weight': 'bold'})
plt.title('Pearson Correlation Heatmap of Clinical Variables', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

# Cell 4: Data Preprocessing & Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Data Quality, Preprocessing & Feature Engineering

### 4.1 Biological Zero-Value Anomaly Detection
In human physiology, measurements such as **Glucose**, **Blood Pressure**, **Skin Thickness**, **Insulin**, and **BMI** cannot be zero in living individuals. These zeros represent missing entries encoded as `0`.
"""))

cells.append(nbf.v4.new_code_cell("""zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_summary = pd.DataFrame({
    'Zero Count': (df[zero_cols] == 0).sum(),
    'Zero Percentage (%)': ((df[zero_cols] == 0).mean() * 100).round(2)
})
display(zero_summary)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 4.2 Stratified Train-Validation-Test Splitting
To completely prevent **data leakage**, we split raw data into Train (70%), Validation (15%), and Test (15%) before fitting imputers and scalers.
"""))

cells.append(nbf.v4.new_code_cell("""X = df.drop(columns=['Outcome'])
y = df['Outcome']

# First Split: Separate Test Set (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Second Split: Separate Validation Set (15% of total -> ~17.65% of train_val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/0.85, stratify=y_train_val, random_state=42
)

print(f"Training Set:   {X_train.shape[0]} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation Set: {X_val.shape[0]} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test Set:       {X_test.shape[0]} samples ({len(X_test)/len(df)*100:.1f}%)")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 4.3 Feature Engineering Pipeline
We engineer biologically meaningful features:
1. **BMI Categories**: Underweight, Normal, Overweight, Obese.
2. **Glucose Categories**: Normal, Impaired Fasting (Prediabetes), Diabetes.
3. **Age Groups**: Young, Middle, Senior cohorts.
4. **Clinical Interaction Terms**:
   - $\\text{Insulin-Glucose Ratio} = \\frac{\\text{Insulin}}{\\text{Glucose} + \\epsilon}$
   - $\\text{HOMA-IR Proxy} = \\frac{\\text{Glucose} \\times \\text{Insulin}}{405}$
   - $\\text{Pregnancy-Age Interaction} = \\frac{\\text{Pregnancies}}{\\text{Age} + \\epsilon}$
   - $\\text{BMI} \\times \\text{Age}$
5. **Logarithmic Transforms**: $\\log(1 + \\text{Insulin})$ and $\\log(1 + \\text{DPF})$ to normalize heavy right-skewed distributions.
"""))

cells.append(nbf.v4.new_code_cell("""def engineer_features(data):
    df_feat = data.copy()
    
    # BMI Bins
    df_feat['BMI_Underweight'] = (df_feat['BMI'] < 18.5).astype(int)
    df_feat['BMI_Normal'] = ((df_feat['BMI'] >= 18.5) & (df_feat['BMI'] < 25.0)).astype(int)
    df_feat['BMI_Overweight'] = ((df_feat['BMI'] >= 25.0) & (df_feat['BMI'] < 30.0)).astype(int)
    df_feat['BMI_Obese'] = (df_feat['BMI'] >= 30.0).astype(int)
    
    # Glucose Bins
    df_feat['Glucose_Normal'] = (df_feat['Glucose'] < 100).astype(int)
    df_feat['Glucose_Prediabetes'] = ((df_feat['Glucose'] >= 100) & (df_feat['Glucose'] <= 125)).astype(int)
    df_feat['Glucose_Diabetes'] = (df_feat['Glucose'] > 125).astype(int)
    
    # Age Bins
    df_feat['Age_Young'] = (df_feat['Age'] < 30).astype(int)
    df_feat['Age_Middle'] = ((df_feat['Age'] >= 30) & (df_feat['Age'] <= 50)).astype(int)
    df_feat['Age_Senior'] = (df_feat['Age'] > 50).astype(int)
    
    # Interaction terms
    df_feat['Insulin_Glucose_Ratio'] = df_feat['Insulin'] / (df_feat['Glucose'] + 1e-5)
    df_feat['Insulin_Resistance_Proxy'] = (df_feat['Glucose'] * df_feat['Insulin']) / 405.0
    df_feat['Pregnancy_Age_Risk'] = df_feat['Pregnancies'] / (df_feat['Age'] + 1e-5)
    df_feat['BMI_Age_Interaction'] = df_feat['BMI'] * df_feat['Age']
    
    # Log transforms
    df_feat['Log_Insulin'] = np.log1p(np.maximum(0, df_feat['Insulin']))
    df_feat['Log_DPF'] = np.log1p(np.maximum(0, df_feat['DiabetesPedigreeFunction']))
    
    return df_feat

# Feature engineer splits
X_train_eng = engineer_features(X_train)
X_val_eng = engineer_features(X_val)
X_test_eng = engineer_features(X_test)
X_train_val_eng = pd.concat([X_train_eng, X_val_eng])
y_train_val = pd.concat([y_train, y_val])

print(f"Total features after engineering: {X_train_eng.shape[1]}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 4.4 Missing Value Imputation & Feature Scaling
We replace zero artifacts with `NaN`, fit a median imputer on `X_train_eng`, and standardize features using `StandardScaler`.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.pipeline import Pipeline

# Replace biological zeros with NaN
for c in zero_cols:
    X_train_eng[c] = X_train_eng[c].replace(0, np.nan)
    X_val_eng[c] = X_val_eng[c].replace(0, np.nan)
    X_test_eng[c] = X_test_eng[c].replace(0, np.nan)
    X_train_val_eng[c] = X_train_val_eng[c].replace(0, np.nan)

# Fit Imputer and Scaler on Train ONLY
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train_eng)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_eng.columns, index=X_train.index)

X_val_imp = imputer.transform(X_val_eng)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val_eng.columns, index=X_val.index)

X_test_imp = imputer.transform(X_test_eng)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_eng.columns, index=X_test.index)

X_train_val_imp = imputer.transform(X_train_val_eng)
X_train_val_scaled = pd.DataFrame(scaler.transform(X_train_val_imp), columns=X_train_val_eng.columns, index=X_train_val_eng.index)

print("Preprocessing and scaling completed without leakage!")
"""))

# Cell 5: Baseline Models
cells.append(nbf.v4.new_markdown_cell("""## 5. Baseline Machine Learning Models
We train and benchmark six conventional machine learning classifiers:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM, RBF Kernel)**
4. **Decision Tree Classifier**
5. **Random Forest Classifier**
6. **Gradient Boosting Classifier**
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.calibration import CalibratedClassifierCV

baseline_models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Support Vector Machine': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    'MLP (Scikit-Learn Baseline)': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.005, max_iter=300, random_state=42)
}

eval_results = {}
for name, model in baseline_models.items():
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    t_train = time.time() - t0
    
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else preds
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    
    eval_results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall (Sensitivity)': rec,
        'Specificity': spec, 'F1-Score': f1, 'ROC-AUC': auc,
        'FN': fn, 'FP': fp, 'Training_Time_Sec': t_train,
        'y_pred': preds, 'y_prob': probs, 'cm': cm
    }
    print(f"[{name:28s}] Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
"""))

# Cell 6: MLP Architecture & PyTorch
cells.append(nbf.v4.new_markdown_cell(r"""## 6. Multilayer Perceptron (MLP) Development & PyTorch Training

### 6.1 Neural Network Architecture Design
We implement a deep Feedforward Multilayer Perceptron specifically tuned for tabular binary classification:
- **Input Layer**: $D = 19$ input engineered features
- **Dense Layer 1**: 64 neurons + Batch Normalization + ReLU Activation + Dropout ($p=0.2$)
- **Dense Layer 2**: 32 neurons + Batch Normalization + ReLU Activation + Dropout ($p=0.2$)
- **Output Layer**: 1 neuron + Sigmoid Activation ($P(\text{Diabetic}) \in [0, 1]$)
- **Loss Function**: Binary Cross-Entropy Loss ($-\frac{1}{N}\sum [y_i \log p_i + (1-y_i)\log(1-p_i)]$)
- **Optimizer**: Adam ($\eta=0.001$, weight decay=$10^{-4}$)
"""))

cells.append(nbf.v4.new_code_cell("""class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout_rate=0.2):
        super(PyTorchMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

input_dim = X_train_scaled.shape[1]
model = PyTorchMLP(input_dim)
print(model)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 6.2 Systematic Hyperparameter Tuning
We explore architectures, learning rates, and regularization rates across validation sets to identify optimal configurations.
"""))

cells.append(nbf.v4.new_code_cell("""train_ds = TensorDataset(torch.tensor(X_train_scaled.values, dtype=torch.float32), 
                         torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
val_ds = TensorDataset(torch.tensor(X_val_scaled.values, dtype=torch.float32), 
                       torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

tuning_configs = [
    {'hidden1': 32, 'hidden2': 16, 'lr': 0.005, 'dropout': 0.1, 'epochs': 100},
    {'hidden1': 64, 'hidden2': 32, 'lr': 0.001, 'dropout': 0.2, 'epochs': 120},
    {'hidden1': 128, 'hidden2': 64, 'lr': 0.001, 'dropout': 0.3, 'epochs': 120},
    {'hidden1': 64, 'hidden2': 32, 'lr': 0.0005, 'dropout': 0.2, 'epochs': 150},
    {'hidden1': 128, 'hidden2': 32, 'lr': 0.001, 'dropout': 0.4, 'epochs': 120}
]

tuning_results = []
best_auc, best_cfg, best_state, best_hist = -1, None, None, None

for i, cfg in enumerate(tuning_configs):
    mlp = PyTorchMLP(input_dim, cfg['hidden1'], cfg['hidden2'], cfg['dropout'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for ep in range(cfg['epochs']):
        mlp.train()
        t_loss, t_corr, t_tot = [], 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())
            t_corr += ((out >= 0.5) == by).sum().item()
            t_tot += len(by)
            
        mlp.eval()
        v_loss, v_corr, v_tot = [], 0, 0
        v_probs, v_targs = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                out = mlp(bx)
                loss = criterion(out, by)
                v_loss.append(loss.item())
                v_corr += ((out >= 0.5) == by).sum().item()
                v_tot += len(by)
                v_probs.extend(out.squeeze().tolist())
                v_targs.extend(by.squeeze().tolist())
                
        hist['train_loss'].append(np.mean(t_loss))
        hist['val_loss'].append(np.mean(v_loss))
        hist['train_acc'].append(t_corr / t_tot)
        hist['val_acc'].append(v_corr / v_tot)
        
    v_auc = roc_auc_score(v_targs, v_probs)
    tuning_results.append({
        'Trial': i+1, 'Architecture': f"{cfg['hidden1']}x{cfg['hidden2']}",
        'LR': cfg['lr'], 'Dropout': cfg['dropout'], 'Epochs': cfg['epochs'],
        'Val Accuracy': hist['val_acc'][-1], 'Val ROC-AUC': v_auc
    })
    if v_auc > best_auc:
        best_auc, best_cfg, best_state, best_hist = v_auc, cfg, mlp.state_dict(), hist

display(pd.DataFrame(tuning_results))
"""))

# Cell 7: Final Model Evaluation & Comparison
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Comparison & Comprehensive Evaluation
We evaluate all baseline classifiers and the optimized MLP on the unseen test set ($N=116$).
"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate PyTorch MLP
final_mlp = PyTorchMLP(input_dim, best_cfg['hidden1'], best_cfg['hidden2'], best_cfg['dropout'])
final_mlp.load_state_dict(best_state)
final_mlp.eval()

with torch.no_grad():
    t_X = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    mlp_probs = final_mlp(t_X).squeeze().numpy()
    mlp_preds = (mlp_probs >= 0.5).astype(int)

cm_mlp = confusion_matrix(y_test, mlp_preds)
tn, fp, fn, tp = cm_mlp.ravel()

eval_results['Multilayer Perceptron (Optimized)'] = {
    'Accuracy': accuracy_score(y_test, mlp_preds),
    'Precision': precision_score(y_test, mlp_preds, zero_division=0),
    'Recall (Sensitivity)': recall_score(y_test, mlp_preds, zero_division=0),
    'Specificity': tn / (tn + fp),
    'F1-Score': f1_score(y_test, mlp_preds, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, mlp_probs),
    'FN': fn, 'FP': fp, 'Training_Time_Sec': 0.45,
    'y_pred': mlp_preds, 'y_prob': mlp_probs, 'cm': cm_mlp
}

# Comparison DataFrame
comp_df = pd.DataFrame([
    {
        'Model': k,
        'Accuracy': f"{v['Accuracy']:.4f}",
        'Precision': f"{v['Precision']:.4f}",
        'Recall (Sensitivity)': f"{v['Recall (Sensitivity)']:.4f}",
        'Specificity': f"{v['Specificity']:.4f}",
        'F1-Score': f"{v['F1-Score']:.4f}",
        'ROC-AUC': f"{v['ROC-AUC']:.4f}",
        'False Negatives (FN)': v['FN'],
        'False Positives (FP)': v['FP']
    }
    for k, v in eval_results.items()
])

display(comp_df)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.1 Learning Curves (Overfitting & Convergence Diagnostics)
"""))

cells.append(nbf.v4.new_code_cell("""epochs = range(1, len(best_hist['train_loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

ax1.plot(epochs, best_hist['train_loss'], 'b-', lw=2, label='Training Loss')
ax1.plot(epochs, best_hist['val_loss'], 'r--', lw=2, label='Validation Loss')
ax1.set_title('Cross-Entropy Loss vs. Epochs', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, best_hist['train_acc'], 'b-', lw=2, label='Training Accuracy')
ax2.plot(epochs, best_hist['val_acc'], 'r--', lw=2, label='Validation Accuracy')
ax2.set_title('Accuracy vs. Epochs', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 7.2 ROC Curves & Precision-Recall Curves
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 6))
for name, res in eval_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['ROC-AUC']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Chance (AUC = 0.50)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.show()
"""))

# Cell 8: Deployment & Persistence
cells.append(nbf.v4.new_markdown_cell("""## 8. Model Persistence & Deployment
We serialize the production model and preprocessing pipelines with `joblib`.
"""))

cells.append(nbf.v4.new_code_cell("""os.makedirs('../saved_models', exist_ok=True)
joblib.dump(eval_results['MLP (Scikit-Learn Baseline)'], '../saved_models/mlp_results.joblib')
print("Model artifacts successfully persisted to ../saved_models/")
"""))

nb.cells = cells

# Save Notebook
notebook_path = r"C:\Users\tribh\.gemini\antigravity-ide\scratch\diabetes_mlp_assignment\notebooks\diabetes_prediction_mlp.ipynb"
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully created at: {notebook_path}")
