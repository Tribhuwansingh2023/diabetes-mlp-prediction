# 🩺 EndoPredict AI: Multilayer Perceptron for Diabetes Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep Feedforward Neural Network (MLP) & Clinical Risk Stratification Engine** for tabular binary diabetes classification with Batch Normalization, Dropout Regularization, Domain Feature Engineering, and Multi-Model Benchmarking.

---

## 📌 Project Overview
EndoPredict AI is an end-to-end Machine Learning & Deep Learning system developed for **Lab Assignment 01: Multilayer Perceptron for Diabetes Prediction**. It provides high-accuracy glycemic risk assessment, biomarker decomposition, explainable risk drivers, and cohort simulation.

### Key Highlights:
- **Deep Multilayer Perceptron (MLP):** Custom 3-layer architecture with Batch Normalization, ReLU, and Dropout ($p=0.2$).
- **19-Dimensional Feature Pipeline:** HOMA-IR / Insulin Resistance proxy, glycemic category flags, adiposity indicators, and clinical interaction ratios.
- **8-Algorithm Benchmark Suite:** Systematic evaluation against Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Baseline MLP, and Optimized Deep MLP.
- **Interactive Streamlit Suite:** Real-time probability estimation, animated SVG gauge meters, and synthetic cohort screening.

---

## 📁 Repository Structure

```
diabetes_mlp_assignment/
├── app/
│   └── app.py                          # Streamlit UI & Interactive Risk Assessor
├── data/
│   └── diabetes.csv                    # Pima Indians Diabetes Dataset
├── notebooks/
│   ├── diabetes_prediction_mlp.ipynb   # Complete Annotated Jupyter Lab Notebook
│   └── generate_notebook.py            # Automated Notebook Generator Script
├── saved_models/
│   ├── diabetes_model.pkl              # Production Scikit-Learn MLP Model
│   ├── eng_preprocessor.joblib         # 19-Feature Scaling & Imputation Pipeline
│   ├── preprocessor.joblib             # Baseline Scaler
│   └── diabetes_mlp_pytorch.pth        # Deep PyTorch Checkpoint State Dict
├── src/
│   ├── preprocessing.py                # Median Imputation & Feature Scaling
│   ├── feature_engineering.py          # HOMA-IR, Interaction Ratios & Log-Transforms
│   ├── models.py                       # PyTorch & Sklearn Model Architectures
│   ├── train.py                        # Master Training, Grid Search & Evaluation
│   └── evaluate.py                     # Metric Computation & Visualization Generator
├── visualizations/                     # ROC, PR, Confusion Matrices & Learning Curves
├── test_deployment.py                  # Automated Unit & Clinical Verification Suite
├── Lab_Assignment_01_Report.md         # Comprehensive Academic Research Report
├── requirements.txt                    # Project Dependencies
├── .gitignore                          # Clean Git Ignore Rules
└── README.md                           # Documentation
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/diabetes_mlp_assignment.git
cd diabetes_mlp_assignment
pip install -r requirements.txt
```

### 2. Run the Interactive Web App
```bash
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Re-train the Pipeline
```bash
python src/train.py
```

### 4. Run Automated Test Suite
```bash
python test_deployment.py
```

---

## 📊 Model Performance Matrix (Unseen Test Set)

| Model | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | FN | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.31% | 72.50% | 82.89% | 0.6905 | 0.7073 | 0.8444 | 11 | 13 |
| **K-Nearest Neighbors** | 81.90% | 77.50% | 84.21% | 0.7209 | 0.7470 | 0.8954 | 9 | 12 |
| **Support Vector Machine (SVM)** | 83.62% | 77.50% | 86.84% | 0.7561 | 0.7654 | 0.8808 | 9 | 10 |
| **MLP (Baseline Sklearn)** | 81.03% | 65.00% | 89.47% | 0.7647 | 0.7027 | 0.8625 | 14 | 8 |
| **Multilayer Perceptron (Optimized)** | **84.48%** | **77.50%** | **88.16%** | **0.7750** | **0.7750** | **0.8980** | **9** | **9** |
| **Decision Tree** | 87.93% | 82.50% | 90.79% | 0.8250 | 0.8250 | 0.9021 | 7 | 7 |
| **Random Forest** | 88.79% | 87.50% | 89.47% | 0.8140 | 0.8434 | 0.9431 | 5 | 8 |
| **Gradient Boosting** | 89.66% | 85.00% | 92.11% | 0.8500 | 0.8500 | 0.9579 | 6 | 6 |

---

## 🚀 Streamlit Community Cloud Deployment
1. Push this repository to your **GitHub** account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New app"**, select your repository, set the main file path to `app/app.py`, and click **"Deploy"**!

---

## 📄 License
This project is licensed under the MIT License.
