# Lab Assignment 01: Predicting Diabetes with Multilayer Perceptron

---

## 1. Project Title
**Diabetes Onset Prediction and Risk Stratification Using an Optimized Multilayer Perceptron (MLP) and Conventional Machine Learning Baselines**

---

## 2. Submitted By
- **Student Name**: Tribhuvan Singh
- **Registration Number**: [Enter Registration Number Here]

---

## 3. Executive Summary
This laboratory investigation presents an end-to-end Machine Learning pipeline designed to predict diabetes onset from physiological, metabolic, and demographic markers using the standard Pima Indians Diabetes Database. The workflow encompasses rigorous data quality audits, missing-value imputation (addressing biologically implausible zero-value measurements), domain-specific feature engineering (expanding 8 raw attributes into 24 processed predictive features), and strict data-leakage prevention via stratified train-validation-test partitioning. A systematic benchmark of seven baseline classifiers was conducted alongside an extensive hyperparameter search for an optimized Scikit-Learn Multilayer Perceptron (`MLPClassifier`). 

The final tuned MLP architecture (`64 -> 32 -> 1`, ReLU activation, Adam optimizer, $\alpha=0.001$, batch size=32) was retrained on combined development data (Train + Validation, $N=652$) and evaluated on an untouched test partition ($N=116$). The optimized MLP achieved an **Accuracy of 82.76%**, **Sensitivity (Recall) of 75.00%**, **Specificity of 86.84%**, **Precision of 75.00%**, **F1-Score of 75.00%**, and a **ROC-AUC of 0.8681**, successfully demonstrating neural non-linear pattern extraction on tabular biometric data.

---

## 4. Problem Statement & Objectives
- **Problem Statement**: Build a robust, reproducible binary classification system to estimate diabetes onset probability from biometric indicators while minimizing diagnostic false negatives.
- **Key Objectives**:
  1. Conduct comprehensive Exploratory Data Analysis (EDA) investigating feature distributions, skewness, class imbalance ($65.1\%$ vs $34.9\%$), and pairwise correlations.
  2. Implement a leak-free preprocessing pipeline that detects biologically implausible zeros in physiological measurements (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) and applies training-fitted median imputation.
  3. Formulate domain feature engineering producing 24 total processed features (WHO BMI categories, ADA glucose stages, age brackets, interaction ratios, and log transforms).
  4. Train and benchmark 7 baseline models: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, and Baseline MLP.
  5. Perform systematic hyperparameter tuning over hidden layer topologies, L2 regularization ($\alpha$), learning rates, and batch sizes.
  6. Deploy the authoritative production model into an interactive Streamlit dashboard with automated verification tests.

---

## 5. Dataset Description & Exploratory Data Analysis

### 5.1 Dataset Features
The Pima Indians Diabetes Database comprises $N = 768$ patient records across 8 raw biometric features and 1 binary outcome label:

| Raw Feature Name | Description | Data Type | Units / Valid Bounds |
| :--- | :--- | :--- | :--- |
| **Pregnancies** | Number of times pregnant | Integer | $0 - 17$ |
| **Glucose** | Plasma glucose concentration (2-hour oral glucose tolerance test) | Integer / Float | $\text{mg/dL}$ ($0 - 199$) |
| **BloodPressure** | Blood Pressure reading | Integer / Float | $\text{mm Hg}$ ($0 - 122$) |
| **SkinThickness** | Triceps skin fold thickness | Integer / Float | $\text{mm}$ ($0 - 99$) |
| **Insulin** | 2-Hour serum insulin | Integer / Float | $\mu\text{U/mL}$ ($0 - 846$) |
| **BMI** | Body Mass Index ($\text{weight in kg} / (\text{height in m})^2$) | Float | $\text{kg/m}^2$ ($0.0 - 67.1$) |
| **DiabetesPedigreeFunction**| Diabetes pedigree function (genetic score) | Float | $0.078 - 2.42$ |
| **Age** | Patient age | Integer | Years ($21 - 81$) |
| **Outcome** (Target) | Diabetes status ($0 = \text{Negative}, 1 = \text{Positive}$) | Binary | $0 \text{ or } 1$ |

### 5.2 Missing Value Analysis & Biologically Implausible Zeros
Continuous physiological measurements cannot be zero in living individuals. Zero values in these fields represent omitted or unrecorded diagnostic tests:

| Attribute | Zero Count ($N=768$) | Missing Percentage (%) | Imputation Strategy |
| :--- | :--- | :--- | :--- |
| **Glucose** | 5 | 0.65% | Median imputation (fit on training partition only) |
| **BloodPressure** | 35 | 4.56% | Median imputation (fit on training partition only) |
| **SkinThickness** | 227 | 29.56% | Median imputation (fit on training partition only) |
| **Insulin** | 374 | 48.70% | Median imputation (fit on training partition only) |
| **BMI** | 11 | 1.43% | Median imputation (fit on training partition only) |

---

## 6. Preprocessing & Feature Engineering Architecture

### 6.1 Correct Pipeline Sequencing (Preventing Data Leakage)
To prevent data contamination and invalid feature computations:
$$\text{Raw Input} \longrightarrow \text{Zero-to-NaN Conversion} \longrightarrow \text{Training Median Imputation} \longrightarrow \text{Feature Engineering} \longrightarrow \text{StandardScaler}$$

### 6.2 Feature Derivations (8 Raw + 16 Engineered = 24 Processed Features)
1. **WHO BMI Categories (4 binary flags)**: `BMI_Underweight` ($<18.5$), `BMI_Normal` ($18.5-24.9$), `BMI_Overweight` ($25.0-29.9$), `BMI_Obese` ($\ge 30.0$).
2. **ADA Glucose Categories (3 binary flags)**: `Glucose_Normal` ($<100$), `Glucose_Prediabetes` ($100-125$), `Glucose_Diabetes` ($\ge 126$).
3. **Age Cohorts (3 binary flags)**: `Age_Young` ($<30$), `Age_Middle` ($30-50$), `Age_Senior` ($>50$).
4. **Mathematical Interaction Terms (4 features)**:
   - `Insulin_Glucose_Ratio` = $\frac{\text{Insulin}}{\text{Glucose} + 10^{-5}}$
   - `Insulin_Resistance_Proxy` = $\frac{\text{Glucose} \times \text{Insulin}}{405.0}$ (engineered interaction model feature)
   - `Pregnancy_Age_Risk` = $\frac{\text{Pregnancies}}{\text{Age} + 10^{-5}}$
   - `BMI_Age_Interaction` = $\text{BMI} \times \text{Age}$
5. **Logarithmic Transforms (2 features)**:
   - `Log_Insulin` = $\ln(1 + \text{Insulin})$
   - `Log_DPF` = $\ln(1 + \text{DiabetesPedigreeFunction})$

---

## 7. Model Development & Hyperparameter Tuning

### 7.1 Dataset Partitions
- **Training Set ($70\%$, $N=536$)**: Used for fitting baseline models and training tuning trials.
- **Validation Set ($15\%$, $N=116$)**: Used exclusively for hyperparameter comparison and model selection.
- **Test Set ($15\%$, $N=116$)**: Held strictly untouched until final single evaluation.

### 7.2 Systematic Hyperparameter Tuning Results
A systematic grid search evaluated Scikit-Learn `MLPClassifier` architectures on the validation partition:

| Trial | Hidden Layer Topology | L2 Alpha ($\alpha$) | Initial Learning Rate ($\eta$) | Batch Size | Max Iterations | Validation Accuracy | Validation ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $(32, 16)$ | $0.0001$ | $0.005$ | 32 | 400 | 0.8534 | 0.8611 |
| **2 (Selected)** | **$(64, 32)$** | **$0.0010$** | **$0.001$** | **32** | **400** | **0.8621** | **0.8855** |
| 3 | $(128, 64)$ | $0.0050$ | $0.001$ | 32 | 500 | 0.8190 | 0.8735 |
| 4 | $(64, 32)$ | $0.0100$ | $0.002$ | 64 | 400 | 0.8362 | 0.8777 |
| 5 | $(128, 32)$ | $0.0010$ | $0.001$ | 32 | 500 | 0.8362 | 0.8813 |
| 6 | $(64, 32, 16)$ | $0.0050$ | $0.001$ | 32 | 400 | 0.8534 | 0.8820 |

**Selected Optimal Configuration**: Hidden layers $(64, 32)$, ReLU activation, Adam optimizer, $\alpha=0.001$, $\eta=0.001$, batch size $32$, achieving top Validation ROC-AUC ($0.8855$).

---

## 8. Final Experimental Results on Untouched Test Set ($N=116$)

The table below reflects actual empirical test outputs generated by the automated pipeline:

| Algorithm | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | False Negatives (FN) | False Positives (FP) |
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

## 9. Discussion & Clinical Trade-Offs

### 9.1 Evaluation of Multilayer Perceptrons on Tabular Data
The optimized MLP demonstrated solid generalization ($82.76\%$ accuracy, $0.8681$ ROC-AUC) on the 24-dimensional engineered feature space, outperforming linear Logistic Regression ($79.31\%$) and the default baseline MLP ($81.03\%$). While tree ensembles (Gradient Boosting and Random Forest) exhibited higher absolute metrics due to their innate inductive bias on orthogonal tabular decision boundaries, the MLP successfully extracted smooth continuous probability estimations without overfitting.

### 9.2 False Negatives vs. False Positives
In medical screening, **False Negatives (FN)** carry higher clinical risk than **False Positives (FP)**:
- An undiagnosed diabetic patient (FN) risks unmanaged microvascular and macrovascular damage.
- A false positive (FP) triggers secondary non-invasive confirmatory testing (e.g., formal Fasting Blood Glucose or HbA1c test).
The optimized MLP reduced false negative misses from 14 (baseline MLP) down to 10.

---

## 10. Deployment & Academic Disclaimer
The final model (`saved_models/diabetes_model.pkl`) and authoritative preprocessor (`saved_models/preprocessor.joblib`) are packaged with a unified inference engine (`src/prediction.py`) and deployed in an interactive Streamlit suite.

> **Academic & Clinical Disclaimer**: This project is an academic machine learning demonstration developed for educational evaluation. It is not certified for autonomous clinical diagnosis or medical treatment planning.
