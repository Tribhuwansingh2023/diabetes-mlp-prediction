"""
Evaluation and Visualization Module for Diabetes Prediction.
Calculates comprehensive classification metrics, confusion matrices, ROC/PR curves,
and training loss progression curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)

# Apply sleek styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a binary classification model and returns a metrics dictionary.
    """
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        probabilities = 1.0 / (1.0 + np.exp(-decision))
    else:
        probabilities = predictions.astype(float)

    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, zero_division=0))
    rec = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_test, probabilities))
    except Exception:
        auc = float('nan')
        
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Sensitivity)': rec,
        'Specificity': specificity,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'y_pred': predictions,
        'y_prob': probabilities,
        'cm': cm
    }

def plot_confusion_matrices(results_dict, save_path):
    """Plots grid of confusion matrices for evaluated models."""
    n_models = len(results_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    
    for idx, (name, res) in enumerate(results_dict.items()):
        cm = res['cm']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'],
            annot_kws={'size': 13, 'weight': 'bold'}
        )
        axes[idx].set_title(f"{name}\nAcc: {res['Accuracy']:.3f} | Recall: {res['Recall (Sensitivity)']:.3f}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Class', fontsize=10)
        axes[idx].set_ylabel('True Class', fontsize=10)
        
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_roc_curves(results_dict, y_test, save_path):
    """Plots combined ROC curves for all models with AUC scores."""
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(results_dict))
    
    for idx, (name, res) in enumerate(results_dict.items()):
        y_prob = res['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = res['ROC-AUC']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2.2, color=palette[idx])
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(results_dict, y_test, save_path):
    """Plots combined Precision-Recall curves."""
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(results_dict))
    
    for idx, (name, res) in enumerate(results_dict.items()):
        y_prob = res['y_prob']
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec, prec, label=f"{name} (F1 = {res['F1-Score']:.3f})", lw=2.2, color=palette[idx])
        
    plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves for Imbalanced Evaluation', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_mlp_learning_curves(mlp_model, save_path):
    """Plots the actual training loss progression curve from the fitted MLPClassifier."""
    if not hasattr(mlp_model, 'loss_curve_'):
        return
        
    loss_history = mlp_model.loss_curve_
    iterations = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, loss_history, 'b-', lw=2.2, label='Training Loss (Cross-Entropy)')
    plt.title('Multilayer Perceptron (MLP) Loss Progression', fontsize=13, fontweight='bold')
    plt.xlabel('Training Iterations (Epochs)', fontsize=11)
    plt.ylabel('Loss Value', fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
