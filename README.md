# 🩺 EndoPredict AI: Multilayer Perceptron for Diabetes Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://endopredict-ai.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🌐 Live Deployment**: Access the production research dashboard at **[https://endopredict-ai.streamlit.app/](https://endopredict-ai.streamlit.app/)**
> 
> **Academic Multilayer Perceptron (MLP) Binary Classification System** for diabetes prediction using domain feature engineering, median zero-value imputation, and multi-model benchmarking.

---

## 📌 Project Overview
EndoPredict AI is a machine learning pipeline developed for **Lab Assignment 01: Predicting Diabetes with Multilayer Perceptron**. It provides automated preprocessing, non-linear feature engineering, systematic hyperparameter tuning, model persistence, and an interactive Streamlit user interface.

### Key Highlights:
- **Authoritative ML Pipeline:** Single unified inference engine (`src/prediction.py`) used identically across training, Streamlit UI, and unit tests.
- **Data-Leakage Free Preprocessing:** Biological zero-value detection on physiological continuous variables (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with training-fitted median imputation.
- **24-Dimensional Processed Features:** 8 raw biometric features + 16 engineered features (WHO BMI bins, ADA glucose categories, age cohorts, interaction ratios, and log transforms).
- **8-Algorithm Benchmark:** Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Baseline MLP, and Optimized MLP (`MLPClassifier`).
- **Zero Binary DLL Dependencies:** Pure HTML/CSS visualization tables avoiding Windows AppLocker PyArrow DLL blocks.

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
├── results/
│   ├── model_comparison.csv            # Unseen Test Set Metrics for All 8 Models
│   ├── hyperparameter_tuning_results.csv# Grid Search Logs Across MLP Configurations
│   └── final_metrics.json              # Serialized JSON Metrics for Best Model
├── saved_models/
│   ├── diabetes_model.pkl              # Production Scikit-Learn MLPClassifier
│   ├── preprocessor.joblib             # Authoritative 24-Feature Preprocessing Pipeline
│   └── model_config.json               # Serialized Metadata Configuration
├── src/
│   ├── preprocessing.py                # Median Imputation & Pipeline Definition
│   ├── feature_engineering.py          # Domain Feature Engineering Logic
│   ├── models.py                       # Baseline & Scikit-Learn MLP Architectures
│   ├── train.py                        # Master Training, Grid Search & Evaluation
│   ├── evaluate.py                     # Metric Computation & Plotting Functions
│   └── prediction.py                   # Unified Single-Source Inference Engine
├── visualizations/                     # ROC, PR, Confusion Matrices & Learning Curves
├── test_deployment.py                  # Automated Unit & Contract Test Suite
├── Lab_Assignment_01_Report.md         # Comprehensive Academic Research Report
├── requirements.txt                    # Project Dependencies
├── .gitignore                          # Clean Git Exclusions
└── README.md                           # Documentation
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Tribhuwansingh2023/diabetes-mlp-prediction.git
cd diabetes-mlp-prediction
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

## 📊 Model Performance Benchmark (Untouched Test Set, $N=116$)

| Algorithm | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | FN | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.31% | 72.50% | 82.89% | 0.6905 | 0.7073 | 0.8444 | 11 | 13 |
| **K-Nearest Neighbors** | 81.90% | 77.50% | 84.21% | 0.7209 | 0.7470 | 0.8954 | 9 | 12 |
| **Support Vector Machine (SVM)** | 83.62% | 77.50% | 86.84% | 0.7561 | 0.7654 | 0.8808 | 9 | 10 |
| **Decision Tree** | 87.93% | 82.50% | 90.79% | 0.8250 | 0.8250 | 0.9021 | 7 | 7 |
| **Random Forest** | 88.79% | 87.50% | 89.47% | 0.8140 | 0.8434 | 0.9431 | 5 | 8 |
| **Gradient Boosting** | 89.66% | 85.00% | 92.11% | 0.8500 | 0.8500 | 0.9579 | 6 | 6 |
| **MLP (Baseline)** | 81.03% | 65.00% | 89.47% | 0.7647 | 0.7027 | 0.8625 | 14 | 8 |
| **Multilayer Perceptron (Optimized)** | **82.76%** | **75.00%** | **86.84%** | **0.7500** | **0.7500** | **0.8681** | **10** | **10** |

---

## 🚀 Streamlit Community Cloud Deployment
1. Push this repository to your **GitHub** account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New app"**, select `Tribhuwansingh2023/diabetes-mlp-prediction`, set the main file path to `app/app.py`, and click **"Deploy"**!

## 👥 Team Members & Contributors
| Sl. No. | Name | Registration Number |
| :---: | :--- | :---: |
| 1 | **Tribhuwan Singh** | `2341019538` |
| 2 | **Surajit Sahoo** | `2341019165` |
| 3 | **Anwesha Srichandan** | `2341019594` |
| 4 | **Priti Rani Maity** | `2341013065` |

---

## ℹ️ Academic & Medical Disclaimer
This application was developed strictly for academic and educational evaluation as part of Lab Assignment 01. It is not certified for autonomous clinical diagnosis and must not replace professional medical evaluations.

