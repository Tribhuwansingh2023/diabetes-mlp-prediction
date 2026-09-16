# CSE 4192: Machine Learning Projects with Python
## Laboratory Assignment 01: Research Project Report

---

# Diabetes Onset Prediction and Risk Stratification Using an Optimized Multilayer Perceptron (MLP) and Conventional Machine Learning Baselines

---

## 👥 Submitted By (Group Members)

| Sl. No. | Student Name | Registration Number | Department / Centre |
| :---: | :--- | :---: | :--- |
| 1 | **Tribhuwan Singh** | `2341019538` | Department of Computer Science & Engineering |
| 2 | **Surajit Sahoo** | `2341019165` | Department of Computer Science & Engineering |
| 3 | **Anwesha Srichandan** | `2341019594` | Department of Computer Science & Engineering |
| 4 | **Priti Rani Maity** | `2341013065` | Department of Computer Science & Engineering |

- **Course Code**: CSE 4192 – Machine Learning Projects with Python
- **Faculty / Mentor**: Dr. Gyana Ranjan Patra, Centre for AI & ML
- **Academic Term**: 2026 Academic Session

---

## 1. Executive Summary

Diabetes mellitus is a progressive metabolic disorder characterized by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both. Early identification of individuals at high risk for diabetes is critical to prevent irreversible microvascular (retinopathy, nephropathy, neuropathy) and macrovascular (cardiovascular disease, stroke) complications. 

This research project delivers a reproducible, data-leakage-free machine learning system designed to predict diabetes onset from physiological, metabolic, and demographic markers using the Pima Indians Diabetes Database ($N = 768$). The pipeline resolves the pervasive issue of biologically implausible zero-value measurements in continuous physiological attributes (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) through training-fitted median imputation. Domain feature engineering expands the 8 raw biometric indicators into a 24-dimensional feature representation incorporating WHO BMI classifications, ADA glycemic categories, metabolic interaction ratios, and log transforms.

A systematic benchmark of seven baseline machine learning algorithms (Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, Baseline MLP) was conducted alongside an extensive grid search for an optimized Scikit-Learn Multilayer Perceptron (`MLPClassifier`). The final tuned MLP architecture ($24 \to 64 \to 32 \to 1$, ReLU activation, Adam optimizer, $\alpha = 0.001$, batch size = 32) was retrained on the combined development set ($N = 652$) and evaluated on an untouched test partition ($N = 116$). The optimized MLP achieved a **Test Accuracy of 82.76%**, **Sensitivity (Recall) of 75.00%**, **Specificity of 86.84%**, **Precision of 75.00%**, **F1-Score of 75.00%**, and **ROC-AUC of 0.8681**, successfully demonstrating non-linear neural decision boundaries on tabular biomedical data.

---

## 2. Motivation

### 2.1 Background & Clinical Importance
According to the International Diabetes Federation (IDF), over 537 million adults globally live with diabetes, a figure projected to rise to 783 million by 2045. A significant proportion of diabetic individuals remain undiagnosed during early stages when lifestyle interventions and early therapeutics are most effective.

### 2.2 Limitations of Existing Diagnostic Approaches
Traditional clinical risk scoring tools rely heavily on static univariate thresholds (e.g., fasting plasma glucose $\ge 126$ mg/dL or $\text{HbA1c} \ge 6.5\%$). However, diabetes pathogenesis is governed by complex, multi-factorial interactions between genetic predisposition, adiposity distribution, subcutaneous fat storage, and pancreatic $\beta$-cell insulin compensation. Linear scoring tools and standard clinical cutoffs fail to capture these non-linear metabolic relationships.

### 2.3 Need for Research
Many existing machine learning implementations in the literature suffer from severe methodological flaws:
1. **Data Leakage**: Preprocessing transformations (imputation and standardization) are computed across the entire dataset prior to splitting, corrupting model evaluation.
2. **Zero-Value Mishandling**: Treating physiological zeros as true values or dropping incomplete records entirely, discarding critical sample variance.
3. **Black-Box Over-Reliance**: Implementing opaque deep networks without baseline comparisons, clinical sanity checks, or decision threshold calibration.

This project addresses these gaps by establishing a mathematically sound, leak-free preprocessing pipeline, expanding biomedical feature representations, and benchmarking an optimized MLP against standard classification baselines.

---

## 3. Problem Statement & Research Gap

```
┌─────────────────────────────────────────────────────────────┐
│                      CURRENT SITUATION                      │
│ High global diabetes prevalence; static univariate clinical │
│ cutoffs miss complex multi-factorial metabolic risk factors.│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        RESEARCH GAP                         │
│ Existing ML models often suffer from data leakage, ignore   │
│ biological zero artifacts, lack domain feature engineering, │
│ and omit multi-algorithm benchmarking against neural MLPs.  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      PROPOSED SOLUTION                      │
│ An authoritative, leak-free pipeline with median zero       │
│ imputation, 24-dim domain engineering, systematic MLP grid  │
│ search, 8-model benchmark, and interactive cloud deployment.│
└─────────────────────────────────────────────────────────────┘
```

- **Problem Statement**: Develop and evaluate an end-to-end binary classification system to accurately predict diabetes onset from biometric indicators while minimizing diagnostic false negatives (missed cases).
- **Core Research Question**: Can domain-specific feature engineering combined with a systematically tuned Multilayer Perceptron (MLP) effectively extract non-linear risk boundaries from tabular biomedical data while maintaining competitive sensitivity and specificity against tree-based ensembles?

---

## 4. Research Objectives (SMART Framework)

The project objectives were formulated according to the SMART (Specific, Measurable, Achievable, Relevant, Time-Bound) guidelines using precise action verbs:

1. **Analyze** the Pima Indians Diabetes dataset through comprehensive Exploratory Data Analysis (EDA), quantifying feature skewness, outlier prevalence, correlation structures, and target class imbalance ($65.1\%$ negative vs. $34.9\%$ positive).
2. **Implement** a leak-free preprocessing pipeline that identifies biologically implausible zero values in continuous variables (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) and imputes them using training-set medians.
3. **Design** domain-specific engineered features (expanding 8 raw features to 24 processed dimensions) capturing WHO adiposity bins, ADA glycemic stages, metabolic ratios, interaction terms, and log transformations.
4. **Develop** baseline classifiers across six distinct algorithmic families (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting) alongside a baseline Multilayer Perceptron.
5. **Optimize** the Multilayer Perceptron architecture using systematic Grid Search over hidden layer topologies, L2 regularization penalties ($\alpha$), initial learning rates, and batch sizes.
6. **Evaluate** and compare all models on an untouched test partition ($N = 116$) using six key performance metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC-AUC.
7. **Deploy** the authoritative trained pipeline into an interactive, enterprise-grade Streamlit web application supporting single-patient risk triage, counterfactual sensitivity simulation, and batch cohort screening.

---

## 5. Objective to Report Chapter Mapping

| Research Objective | Corresponding Report Chapter | Deliverables & Artifacts |
| :--- | :--- | :--- |
| **1. Data Analysis & Exploration** | Section 7: Dataset & EDA | Skewness tables, correlation heatmap, boxplots |
| **2. Leak-Free Preprocessing** | Section 8: Data Preprocessing | `DiabetesPreprocessor`, zero-masking logic |
| **3. Feature Engineering** | Section 9: Domain Feature Engineering | 16 mathematical feature formulas, 24-dim schema |
| **4. Baseline Model Development** | Section 10: Model Architecture & Baselines | 6 baseline classifier pipelines |
| **5. Hyperparameter Optimization** | Section 11: Hyperparameter Tuning | Grid search evaluation matrix, validation logs |
| **6. Performance Evaluation** | Section 12: Results & Comparative Analysis | Benchmark table ($N=116$), ROC/PR/Confusion matrices |
| **7. Production Deployment** | Section 13: Web Deployment & Verification | `app/app.py` Streamlit Cloud dashboard, test suite |

---

## 6. Literature Survey

A comprehensive survey of relevant literature on machine learning for diabetes prediction was conducted across major academic repositories (IEEE Xplore, ScienceDirect, Springer, Google Scholar):

| Author & Year | Methodology / Algorithm | Dataset Used | Reported Performance | Identified Limitations & Research Gaps |
| :--- | :--- | :--- | :---: | :--- |
| **Smith et al. (1988)** | ADAP Perceptron Neural Network | Pima Indians ($N=768$) | Accuracy: 76.0% | Early neural model; no systematic missing value imputation; high false negative rate. |
| **Kavakiotis et al. (2017)** | Systematic Survey of ML / Data Mining | Various Clinical Databases | Accuracy: 70% – 82% | Identified pervasive data leakage and lack of standardized evaluation protocols across studies. |
| **Sisodia & Sisodia (2018)** | Decision Tree, SVM, Naive Bayes | Pima Indians ($N=768$) | Naive Bayes: 76.3% | Treated zero values as valid measurements; omitted non-linear neural architectures. |
| **Zou et al. (2018)** | Decision Tree, Random Forest, Neural Net | Hospital Physical Exam Data | RF: 80.84% | Evaluated only raw features without domain feature engineering or interaction terms. |
| **Patil et al. (2021)** | Multilayer Perceptron & Ensembles | Pima Indians ($N=768$) | MLP: 79.1% | Omitted hyperparameter tuning for regularization ($\alpha$); no threshold calibration. |

### Research Gap Identification:
Prior studies frequently demonstrate one or more of the following critical deficiencies:
- Preprocessing steps applied globally across all samples before partitioning, causing data leakage.
- Treating physiological zeros (e.g., Blood Pressure = 0, Glucose = 0) as valid numerical readings.
- Relying exclusively on raw biometric features without engineering metabolic interaction terms.
- Evaluating models purely on Accuracy, ignoring the clinical hazard of False Negatives (missed diagnoses).

---

## 7. Dataset Description & Exploratory Data Analysis

### 7.1 Dataset Description
The investigation utilizes the **Pima Indians Diabetes Database** originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). The dataset comprises $N = 768$ female patients of Pima Indian heritage aged 21 years and older:

| Feature Name | Type | Description | Valid Biological Range | Missing / Zero Count |
| :--- | :---: | :--- | :---: | :---: |
| **Pregnancies** | Integer | Number of completed pregnancies | $[0, 17]$ | 0 (0 is valid) |
| **Glucose** | Float | Plasma glucose concentration (2h OGTT) | $44 - 199$ mg/dL | 5 ($0.65\%$) |
| **BloodPressure** | Float | Blood pressure reading | $24 - 122$ mm Hg | 35 ($4.56\%$) |
| **SkinThickness** | Float | Triceps skin fold thickness | $7 - 99$ mm | 227 ($29.56\%$) |
| **Insulin** | Float | 2-Hour post-load serum insulin | $14 - 846$ μU/mL | 374 ($48.70\%$) |
| **BMI** | Float | Body Mass Index (weight in $\text{kg}/\text{m}^2$) | $18.2 - 67.1$ | 11 ($1.43\%$) |
| **DiabetesPedigreeFunction** | Float | Genetic risk score based on family history | $0.078 - 2.42$ | 0 (None) |
| **Age** | Integer | Patient age in years | $21 - 81$ years | 0 (None) |
| **Outcome** (Target) | Binary | Diabetes diagnosis within 5 years | $\{0, 1\}$ | Class 0: 500 ($65.1\%$), Class 1: 268 ($34.9\%$) |

### 7.2 Exploratory Data Analysis Findings
1. **Target Imbalance**: 500 non-diabetic ($65.1\%$) vs. 268 diabetic ($34.9\%$). Class imbalance necessitates evaluation using Precision, Sensitivity, Specificity, F1-Score, and ROC-AUC rather than Accuracy alone.
2. **Distribution Skewness**: `Insulin` (skewness = 2.27) and `DiabetesPedigreeFunction` (skewness = 1.92) exhibit severe positive skewness, justifying logarithmic compression.
3. **Correlation Analysis**: `Glucose` exhibits the highest linear correlation with `Outcome` ($r = 0.49$), followed by `BMI` ($r = 0.31$) and `Age` ($r = 0.24$). Strong collinearity is observed between `Pregnancies` and `Age` ($r = 0.54$) and between `SkinThickness` and `BMI` ($r = 0.65$).

---

## 8. Data Preprocessing & Leak-Free Imputation

To eliminate data leakage, all statistical parameters (medians, means, standard deviations) are computed strictly on the training partition ($D_{\text{train}}$) and subsequently applied to transform validation and test sets.

```
Raw Patient Record [8 Features]
        │
        ▼
[Zero Detection] Detect biological zeros in Glucose, BP, Skin, Insulin, BMI -> Replace with NaN
        │
        ▼
[Train-Fitted Imputation] Impute NaNs using Training-Set Medians (Median_train)
        │
        ▼
[Feature Engineering] Compute 16 domain features on clean imputed values -> 24 Total Features
        │
        ▼
[Train-Fitted Scaling] Standardize all 24 features using StandardScaler (mu_train, sigma_train)
        │
        ▼
Standardized 24-Dimensional Feature Vector -> Passed to Model
```

### Imputation Statistics Computed on Training Data ($N = 537$):
- **Glucose**: $117.0$ mg/dL
- **BloodPressure**: $72.0$ mm Hg
- **SkinThickness**: $29.0$ mm
- **Insulin**: $125.0$ μU/mL
- **BMI**: $32.0$ $\text{kg}/\text{m}^2$

---

## 9. Domain-Specific Feature Engineering

To enhance the representational capacity of the MLP and baseline models, 16 domain-engineered features were formulated based on clinical literature (ADA, WHO):

| Category | Engineered Feature Name | Mathematical Definition / Logic | Clinical & Biological Justification |
| :--- | :--- | :--- | :--- |
| **WHO Adiposity Bins** | `BMI_Underweight`<br>`BMI_Normal`<br>`BMI_Overweight`<br>`BMI_Obese` | $\text{BMI} < 18.5$<br>$18.5 \le \text{BMI} < 25.0$<br>$25.0 \le \text{BMI} < 30.0$<br>$\text{BMI} \ge 30.0$ | Captures non-linear thresholds for metabolic syndrome and obesity-induced insulin resistance. |
| **ADA Glycemic Stages** | `Glucose_Normal`<br>`Glucose_Prediabetes`<br>`Glucose_Diabetes` | $\text{Glucose} < 100$<br>$100 \le \text{Glucose} < 126$<br>$\text{Glucose} \ge 126$ mg/dL | Reflects American Diabetes Association diagnostic boundaries for impaired fasting glucose. |
| **Age Cohorts** | `Age_Young`<br>`Age_Middle`<br>`Age_Senior` | $\text{Age} < 30$<br>$30 \le \text{Age} \le 50$<br>$\text{Age} > 50$ years | Encodes age-related decline in pancreatic $\beta$-cell secretory capacity. |
| **Metabolic Ratio** | `Insulin_Glucose_Ratio` | $\frac{\text{Insulin}}{\text{Glucose} + 10^{-5}}$ | Proxies $\beta$-cell compensation for ambient glycemic load. |
| **Interaction Term** | `Insulin_Resistance_Proxy` | $\frac{\text{Glucose} \times \text{Insulin}}{405.0}$ | Product interaction surrogate capturing combined glucotoxicity and hyperinsulinemia. |
| **Gestational Risk** | `Pregnancy_Age_Risk` | $\frac{\text{Pregnancies}}{\text{Age} + 10^{-5}}$ | Normalizes parity against chronological age to quantify cumulative gestational metabolic stress. |
| **Interaction Term** | `BMI_Age_Interaction` | $\text{BMI} \times \text{Age}$ | Models synergistic risk amplification between chronic obesity and advancing age. |
| **Non-Linear Transforms** | `Log_Insulin`<br>`Log_DPF` | $\ln(1 + \text{Insulin})$<br>$\ln(1 + \text{DPF})$ | Compresses severe right-skewed distributions to improve gradient stability. |

**Total Processed Input Dimension**: $8 \text{ (Raw)} + 16 \text{ (Engineered)} = \mathbf{24 \text{ Features}}$.

---

## 10. Model Architecture & Mathematical Formulation

### 10.1 Multilayer Perceptron Topology

```
┌────────────────────────────────────────────────────────┐
│      INPUT LAYER: 24 Standardized Processed Features   │
└───────────────────────────┬────────────────────────────┘
                            │ (W1 in R^(24 x 64), b1 in R^64)
                            ▼
┌────────────────────────────────────────────────────────┐
│   HIDDEN LAYER 1: 64 Neurons + ReLU Activation         │
│   z1 = W1*x + b1,   a1 = max(0, z1)                    │
└───────────────────────────┬────────────────────────────┘
                            │ (W2 in R^(64 x 32), b2 in R^32)
                            ▼
┌────────────────────────────────────────────────────────┐
│   HIDDEN LAYER 2: 32 Neurons + ReLU Activation         │
│   z2 = W2*a1 + b2,  a2 = max(0, z2)                    │
└───────────────────────────┬────────────────────────────┘
                            │ (W3 in R^(32 x 1), b3 in R^1)
                            ▼
┌────────────────────────────────────────────────────────┐
│   OUTPUT LAYER: 1 Neuron + Sigmoid Activation          │
│   z3 = W3*a2 + b3,  y_hat = sigma(z3) in [0, 1]        │
└────────────────────────────────────────────────────────┘
```

### 10.2 Trainable Parameters Calculation
- **Layer 1 ($24 \to 64$)**: $24 \times 64 = 1,536$ weights $+ 64$ biases $= \mathbf{1,600}$ parameters.
- **Layer 2 ($64 \to 32$)**: $64 \times 32 = 2,048$ weights $+ 32$ biases $= \mathbf{2,080}$ parameters.
- **Output Layer ($32 \to 1$)**: $32 \times 1 = 32$ weights $+ 1$ bias $= \mathbf{33}$ parameters.
- **Total Trainable Network Parameters**: $1,600 + 2,080 + 33 = \mathbf{3,713 \text{ Parameters}}$.

### 10.3 Loss Function & Regularization
The network minimizes the binary cross-entropy loss augmented with L2 weight regularization:

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\alpha}{2N} \sum_{l=1}^{L} \|\mathbf{W}^{(l)}\|_F^2$$

where $\hat{y}_i = \sigma(z_3^{(i)})$, $\alpha = 0.001$ is the L2 penalty, and optimization is governed by the **Adam optimizer** with adaptive moment estimation ($\beta_1 = 0.9, \beta_2 = 0.999$).

---

## 11. Experimental Setup & Hyperparameter Tuning

### 11.1 Experimental Environment Specifications
- **Hardware**: 12th Gen Intel Core i7 / AMD Ryzen 7, 16 GB DDR4 RAM, NVIDIA RTX GPU Acceleration.
- **Software Stack**: Python 3.12, Scikit-Learn 1.3+, Pandas 2.0+, NumPy 1.24+, SciPy 1.10+, Joblib 1.3+, Streamlit 1.28+, Plotly 5.15+, Matplotlib 3.7+, Seaborn 0.12+.
- **Data Partitioning**: Stratified split — 70% Train ($N=537$), 15% Validation ($N=115$), 15% Test ($N=116$).
- **Random Seed**: Fixed at `42` across all data splitters, initializers, and model seeds for strict reproducibility.

### 11.2 Hyperparameter Search Space & Validation Results

| Configuration | Hidden Layers | Activation | Optimizer | L2 Penalty ($\alpha$) | Initial LR ($\eta$) | Batch Size | Val Accuracy | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Config 1 | $(32, 16)$ | ReLU | Adam | $0.0001$ | $0.001$ | 32 | $78.26\%$ | $0.8352$ |
| Config 2 | $(32, 16)$ | ReLU | Adam | $0.01$ | $0.001$ | 64 | $79.13\%$ | $0.8410$ |
| Config 3 | $(64, 32)$ | ReLU | Adam | $0.0001$ | $0.001$ | 32 | $80.00\%$ | $0.8520$ |
| **Config 4 (Optimal)** | $\mathbf{(64, 32)}$ | **ReLU** | **Adam** | $\mathbf{0.001}$ | $\mathbf{0.001}$ | **32** | $\mathbf{81.74\%}$ | $\mathbf{0.8645}$ |
| Config 5 | $(64, 32)$ | ReLU | Adam | $0.01$ | $0.005$ | 64 | $79.13\%$ | $0.8460$ |
| Config 6 | $(128, 64)$ | ReLU | Adam | $0.001$ | $0.001$ | 32 | $80.87\%$ | $0.8590$ |
| Config 7 | $(128, 64, 32)$ | ReLU | Adam | $0.01$ | $0.001$ | 64 | $77.39\%$ | $0.8280$ |
| Config 8 | $(64, 32)$ | Tanh | Adam | $0.001$ | $0.001$ | 32 | $79.13\%$ | $0.8390$ |

**Optimal Selected Configuration**: Hidden Topology: `(64, 32)`, Activation: `relu`, Solver: `adam`, $\alpha = 0.001$, $\eta = 0.001$, Batch Size: `32`, Early Stopping: `True` (Patience = 10 epochs).

---

## 12. Results & Comparative Performance

The optimized MLP and seven baseline algorithms were evaluated on the **untouched test partition ($N = 116$, containing 76 Negative and 40 Positive cases)**:

### 12.1 Untouched Test Set Evaluation Benchmark ($N = 116$)

| Algorithm | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | False Negatives (FN) | False Positives (FP) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $79.31\%$ | $72.50\%$ | $82.89\%$ | $0.6905$ | $0.7073$ | $0.8444$ | 11 | 13 | $0.0412$ |
| **K-Nearest Neighbors** | $81.90\%$ | $77.50\%$ | $84.21\%$ | $0.7209$ | $0.7470$ | $0.8954$ | 9 | 12 | $0.0054$ |
| **Support Vector Machine (RBF)** | $83.62\%$ | $77.50\%$ | $86.84\%$ | $0.7561$ | $0.7654$ | $0.8808$ | 9 | 10 | $0.0852$ |
| **Decision Tree** | $87.93\%$ | $82.50\%$ | $90.79\%$ | $0.8250$ | $0.8250$ | $0.9021$ | 7 | 7 | $0.0085$ |
| **Random Forest** | $88.79\%$ | $87.50\%$ | $89.47\%$ | $0.8140$ | $0.8434$ | $0.9431$ | 5 | 8 | $0.5517$ |
| **Gradient Boosting** | $89.66\%$ | $85.00\%$ | $92.11\%$ | $0.8500$ | $0.8500$ | $0.9579$ | 6 | 6 | $0.4593$ |
| **MLP (Baseline, Un-tuned)** | $81.03\%$ | $65.00\%$ | $89.47\%$ | $0.7647$ | $0.7027$ | $0.8625$ | 14 | 8 | $0.1838$ |
| **MLP (Optimized & Retrained)** | $\mathbf{82.76\%}$ | $\mathbf{75.00\%}$ | $\mathbf{86.84\%}$ | $\mathbf{0.7500}$ | $\mathbf{0.7500}$ | $\mathbf{0.8681}$ | **10** | **10** | $\mathbf{3.2738}$ |

### 12.2 Optimized MLP Confusion Matrix Breakdown ($N = 116$)

$$\begin{pmatrix} \text{True Negatives (TN)} & \text{False Positives (FP)} \\ \text{False Negatives (FN)} & \text{True Positives (TP)} \end{pmatrix} = \begin{pmatrix} 66 & 10 \\ 10 & 30 \end{pmatrix}$$

- **True Negatives (TN)**: $66$ correctly identified non-diabetic patients.
- **True Positives (TP)**: $30$ correctly detected diabetic patients.
- **False Negatives (FN)**: $10$ missed diabetic patients ($25.0\%$ miss rate).
- **False Positives (FP)**: $10$ false alarms ($13.16\%$ false alarm rate).

---

## 13. Discussion of Results & Clinical Trade-offs

### 13.1 Why the Optimized MLP Performed Well
1. **Effective Non-Linear Representation**: The two-stage hierarchical topology ($24 \to 64 \to 32 \to 1$) allowed the network to compose composite interactions (e.g., combining `Insulin_Resistance_Proxy` with `BMI_Age_Interaction`) into high-level geometric risk surfaces.
2. **Regularization Balance**: L2 weight decay ($\alpha = 0.001$) and early stopping prevented co-adaptation of weights, improving generalization from the $81.03\%$ baseline to $82.76\%$.
3. **Imputation Synergy**: Imputing zeros with training medians prior to computing log transforms and interaction terms prevented extreme gradient anomalies.

### 13.2 Comparison with Tree-Based Ensembles
While ensemble models (Random Forest at $88.79\%$ and Gradient Boosting at $89.66\%$) achieved higher overall accuracy due to orthogonal axis-aligned decision trees on tabular data, the Multilayer Perceptron provides **well-calibrated, smooth continuous probability outputs $\hat{y} \in [0, 1]$**. In clinical decision support, continuous probabilities allow dynamic decision threshold adjustments ($\tau \in [0.20, 0.80]$), enabling clinicians to calibrate sensitivity vs. specificity depending on screening priorities.

### 13.3 Diagnostic Trade-off Analysis
In diabetes screening, **False Negatives (FN)** present severe clinical risks because untreated diabetes progresses to irreversible organ damage. Conversely, **False Positives (FP)** incur secondary confirmatory diagnostic costs (HbA1c tests). Calibrating the decision threshold to $\tau = 0.35$ in the deployed application reduces False Negatives from 10 down to 4, prioritizing early clinical intervention.

---

## 14. Conclusion & Future Scope

### 14.1 Summary of Findings
An end-to-end binary classification pipeline was successfully developed, evaluated, and deployed for diabetes onset prediction using the Pima Indians Diabetes Database. The investigation demonstrated:
1. Biological zero-value detection combined with training-fitted median imputation completely prevents data leakage.
2. Domain feature engineering (expanding 8 raw features to 24 dimensions) provides rich non-linear representations for neural modeling.
3. The optimized Scikit-Learn Multilayer Perceptron achieved **82.76% Accuracy**, **75.00% Sensitivity**, **86.84% Specificity**, and **0.8681 ROC-AUC** on untouched test data.
4. An enterprise Streamlit dashboard was constructed, tested, and prepared for public cloud deployment.

### 14.2 Future Scope & Research Extensions
1. **Multi-Center Cohort Validation**: Validate the pipeline on larger, ethnically diverse multi-center clinical datasets (e.g., NHANES, UK Biobank).
2. **Tabular Deep Architectures**: Benchmark against modern self-attention tabular architectures (TabNet, FT-Transformer, SAINT).
3. **Explainable AI Integration**: Incorporate SHAP (SHapley Additive exPlanations) and Integrated Gradients for local feature attribution in clinical workflows.
4. **Edge Device Deployment**: Quantize and export the trained MLP model into ONNX format for low-latency point-of-care mobile diagnostic devices.

---

## 15. References

1. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.
2. American Diabetes Association. (2024). Standards of Medical Care in Diabetes—2024. *Diabetes Care*, 47(Suppl. 1), S1–S343.
3. World Health Organization. (2020). *Classification of diabetes mellitus*. World Health Organization Technical Report Series.
4. Kavakiotis, I., Tsave, O., Salifoglou, A., Maglaveras, N., Vlahavas, I., & Chouvarda, I. (2017). Machine learning and data mining methods in diabetes research. *Computational and Structural Biotechnology Journal*, 15, 104–116.
5. Sisodia, D., & Sisodia, D. S. (2018). Prediction of diabetes using classification algorithms. *Procedia Computer Science*, 132, 1578–1585.
6. Zou, Q., Qu, K., Luo, Y., Yin, D., Ju, Y., & Tang, H. (2018). Predicting diabetes mellitus with machine learning techniques. *Frontiers in Genetics*, 9, 515.
7. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
8. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
