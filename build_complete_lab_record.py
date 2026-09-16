"""
Comprehensive Laboratory Record Generator for CSE 4192: Machine Learning Projects with Python.
Builds both:
  1. Laboratory_Record_CSE4192.docx (Microsoft Word format with exact page breaks per chapter)
  2. Laboratory_Record_CSE4192.pdf  (ReportLab PDF format with exact page breaks per chapter)

Follows the Siksha 'O' Anusandhan (Deemed to be University) official Laboratory Record format.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.docx")
PDF_PATH = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.pdf")
LOGO_PATH = os.path.join(BASE_DIR, "soa_logo.png")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

# =============================================================================
# PART 1: GENERATE DOCX (EVERY CHAPTER STARTS ON A NEW PAGE)
# =============================================================================

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def generate_docx():
    print("Building Laboratory_Record_CSE4192.docx ...")
    doc = Document()

    # 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.bold = True
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.bold = True
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        return p

    def add_bullet(text):
        p = doc.add_paragraph(text)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(2)
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(11)

    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_before = Pt(20)
    p_uni.paragraph_format.space_after = Pt(2)
    r1 = p_uni.add_run("SIKSHA ‘O’ ANUSANDHAN\n")
    r1.font.name = 'Times New Roman'; r1.font.size = Pt(16); r1.font.bold = True
    r2 = p_uni.add_run("(DEEMED TO BE UNIVERSITY)\n")
    r2.font.name = 'Times New Roman'; r2.font.size = Pt(13); r2.font.bold = True

    doc.add_paragraph()

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(10)
    p_meta.paragraph_format.space_after = Pt(15)
    r_m1 = p_meta.add_run("Admission Batch: ")
    r_m1.font.bold = True
    p_meta.add_run("2023 – 2027                                  ")
    r_m2 = p_meta.add_run("Session: ")
    r_m2.font.bold = True
    p_meta.add_run("2025 – 2026")

    doc.add_paragraph()

    p_rec = doc.add_paragraph()
    p_rec.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_rec.paragraph_format.space_before = Pt(10)
    p_rec.paragraph_format.space_after = Pt(4)
    r_rec = p_rec.add_run("Laboratory Record\n")
    r_rec.font.name = 'Times New Roman'; r_rec.font.size = Pt(14); r_rec.font.bold = True

    p_subj = doc.add_paragraph()
    p_subj.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subj.paragraph_format.space_after = Pt(15)
    r_subj = p_subj.add_run("Machine Learning Projects with Python\n(CSE 4192)")
    r_subj.font.name = 'Times New Roman'; r_subj.font.size = Pt(16); r_subj.font.bold = True

    p_subm = doc.add_paragraph()
    p_subm.paragraph_format.space_before = Pt(8)
    p_subm.paragraph_format.space_after = Pt(4)
    r_subm = p_subm.add_run("Submitted by")
    r_subm.font.name = 'Times New Roman'; r_subm.font.size = Pt(12); r_subm.font.bold = True

    p_names = doc.add_paragraph()
    p_names.paragraph_format.left_indent = Inches(0.5)
    p_names.paragraph_format.line_spacing = 1.3
    members = [
        "1. Name: Tribhuwan Singh (Regd. No.: 2341019538)",
        "2. Name: Surajit Sahoo (Regd. No.: 2341019165)",
        "3. Name: Anwesha Srichandan (Regd. No.: 2341019594)",
        "4. Name: Priti Rani Maity (Regd. No.: 2341013065)"
    ]
    for m in members:
        r = p_names.add_run(m + "\n")
        r.font.name = 'Times New Roman'; r.font.size = Pt(11); r.font.bold = True

    if os.path.exists(LOGO_PATH):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(8)
        p_logo.paragraph_format.space_after = Pt(15)
        p_logo.add_run().add_picture(LOGO_PATH, width=Inches(1.8))

    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_before = Pt(12)
    p_dept.paragraph_format.space_after = Pt(2)
    r_d1 = p_dept.add_run("Centre for Artificial Intelligence & Machine Learning\n")
    r_d1.font.name = 'Times New Roman'; r_d1.font.size = Pt(12); r_d1.font.bold = True

    r_d2 = p_dept.add_run("Faculty of Engineering & Technology (ITER)\n")
    r_d2.font.name = 'Times New Roman'; r_d2.font.size = Pt(11.5); r_d2.font.bold = True

    r_d3 = p_dept.add_run("Jagamohan Nagar, Jagamara, Bhubaneswar, Odisha – 751030")
    r_d3.font.name = 'Times New Roman'; r_d3.font.size = Pt(10); r_d3.font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 2: 1 INTRODUCTION
    # -------------------------------------------------------------
    add_h1("1 Introduction")
    add_h2("1.1 Problem Statement")
    add_body(
        "Diabetes mellitus is a chronic metabolic disease characterized by persistent hyperglycemia resulting from insulin secretion "
        "defects, insulin action resistance, or both. Early identification of individuals at risk is critical to prevent severe long-term "
        "microvascular (retinopathy, nephropathy, neuropathy) and macrovascular (cardiovascular disease, stroke) complications. "
        "However, conventional diagnostic mechanisms rely on static univariate clinical cutoffs (e.g., fasting blood glucose >= 126 mg/dL) "
        "which fail to capture subtle multi-factorial non-linear interactions between adiposity, insulin sensitivity, and genetic lineage. "
        "Furthermore, prior machine learning implementations frequently suffer from severe data leakage (fitting preprocessing globally) "
        "and incorrect handling of physiological zero readings. This project addresses these critical challenges by developing a robust, "
        "leak-free binary classification system to accurately predict diabetes onset from patient biometric markers."
    )
    add_h2("1.2 Aim")
    add_body(
        "State the primary aim of your project: The overarching aim of this laboratory project is to design, implement, "
        "optimize, benchmark, and deploy a data-leakage-free Multilayer Perceptron (MLP) neural network capable of accurately predicting "
        "5-year diabetes onset probability from patient biometric markers using the Pima Indians Diabetes Database, comparing its predictive "
        "performance against standard machine learning classification baselines."
    )
    add_h2("1.3 Objectives")
    add_body("List the specific objectives:")
    add_bullet("1. Conduct exhaustive Exploratory Data Analysis (EDA) quantifying target class imbalance (65.1% negative vs. 34.9% positive), skewness, and pairwise correlation structures.")
    add_bullet("2. Implement a leak-free preprocessing pipeline that identifies biologically implausible zero values in continuous variables (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) and imputes them using training-fitted medians.")
    add_bullet("3. Formulate 16 domain-engineered features (expanding 8 raw features to 24 processed dimensions) incorporating WHO BMI categories, ADA glycemic stages, metabolic interaction ratios, and log transforms.")
    add_bullet("4. Develop and evaluate 7 baseline machine learning classifiers (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Baseline MLP).")
    add_bullet("5. Systematically optimize the Multilayer Perceptron architecture using Grid Search across hidden layer topologies, L2 penalties (alpha), initial learning rates, and batch sizes.")
    add_bullet("6. Evaluate all candidate models on an untouched test partition (N = 116) across Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC.")
    add_bullet("7. Deploy the production pipeline to an interactive Streamlit Cloud dashboard supporting single-patient triage, counterfactual sensitivity simulation, and batch cohort screening.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 3: 2 DATASET DESCRIPTION
    # -------------------------------------------------------------
    add_h1("2 Dataset Description")
    add_h2("2.1 Dataset Source")
    add_body(
        "The project utilizes the standard Pima Indians Diabetes Database collected by the National Institute of Diabetes and Digestive "
        "and Kidney Diseases (NIDDK). The dataset comprises female patient records of Pima Indian descent aged 21 years and older residing near Phoenix, Arizona."
    )
    add_h2("2.2 Dataset Features")
    add_body("The dataset contains 8 raw clinical, physiological, and demographic input attributes:")
    add_bullet("• Pregnancies: Total number of completed pregnancies.")
    add_bullet("• Glucose: Plasma glucose concentration 2 hours after a 75g oral glucose tolerance test (OGTT) in mg/dL.")
    add_bullet("• BloodPressure: Diastolic blood pressure reading in mm Hg.")
    add_bullet("• SkinThickness: Triceps skinfold caliper thickness in mm (proxy for subcutaneous adiposity).")
    add_bullet("• Insulin: 2-Hour post-load serum insulin concentration in μU/mL.")
    add_bullet("• BMI: Body Mass Index computed as weight in kg / (height in m)^2.")
    add_bullet("• DiabetesPedigreeFunction (DPF): Genetic risk score synthesized from ancestral diabetic pedigree.")
    add_bullet("• Age: Chronological patient age in years.")

    add_h2("2.3 Target Variable")
    add_body(
        "The target variable is Outcome, a binary classification indicator where 1 denotes confirmed diabetes onset within five years "
        "of clinical examination, and 0 denotes non-diabetic status."
    )
    add_h2("2.4 Dataset Statistics")
    add_body("The dataset contains N = 768 patient instances. Summary descriptive statistics are presented in Table 1.")

    t_ds = doc.add_table(rows=1, cols=6)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t_ds.rows[0].cells
    hdr_titles = ["Feature", "Type", "Mean ± SD", "Min", "Max", "Zero Count (%)"]
    for i, title in enumerate(hdr_titles):
        hdr[i].paragraphs[0].text = title
        hdr[i].paragraphs[0].runs[0].font.bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(9.5)
        hdr[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr[i], "0F172A")
        set_cell_margins(hdr[i])

    ds_rows = [
        ["Pregnancies", "Discrete", "3.85 ± 3.37", "0", "17", "111 (14.45%)*"],
        ["Glucose", "Continuous", "120.89 ± 31.97", "0 (44)", "199", "5 (0.65%)"],
        ["BloodPressure", "Continuous", "69.11 ± 19.36", "0 (24)", "122", "35 (4.56%)"],
        ["SkinThickness", "Continuous", "20.54 ± 15.95", "0 (7)", "99", "227 (29.56%)"],
        ["Insulin", "Continuous", "79.80 ± 115.24", "0 (14)", "846", "374 (48.70%)"],
        ["BMI", "Continuous", "31.99 ± 7.88", "0 (18.2)", "67.1", "11 (1.43%)"],
        ["DiabetesPedigree", "Continuous", "0.47 ± 0.33", "0.078", "2.42", "0 (0.00%)"],
        ["Age", "Discrete", "33.24 ± 11.76", "21", "81", "0 (0.00%)"],
        ["Outcome", "Binary", "0.35 ± 0.48", "0", "1", "500 (65.1%) [0]"]
    ]
    for row_data in ds_rows:
        row = t_ds.add_row().cells
        for i, val in enumerate(row_data):
            row[i].paragraphs[0].text = val
            row[i].paragraphs[0].runs[0].font.size = Pt(9)
            set_cell_margins(row[i])

    add_body("*Note: Zero values in Pregnancies represent nulliparous women (biologically valid). Zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI represent unrecorded measurements.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 4: 3 EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------
    add_h1("3 Exploratory Data Analysis")
    add_h2("3.1 Dataset Structure")
    add_body("The dataset contains 768 rows and 9 columns with zero explicit SQL NULL values, but extensive hidden biological zero values.")
    add_h2("3.2 Descriptive Statistics")
    add_body("Continuous physiological attributes exhibit pronounced variance. Insulin ranges from 0 to 846 μU/mL with a high standard deviation (115.24 μU/mL). Mean BMI is 31.99 kg/m^2, indicating that the patient cohort predominantly falls into the WHO Obese Class I category.")
    add_h2("3.3 Missing Value Analysis")
    add_body("A biological sanity audit revealed that 374 records (48.70%) have Insulin = 0, 227 records (29.56%) have SkinThickness = 0, 35 records (4.56%) have BloodPressure = 0, 11 records (1.43%) have BMI = 0, and 5 records (0.65%) have Glucose = 0. A living human cannot have zero blood glucose or zero blood pressure; these are missing data artifacts requiring imputation.")
    add_h2("3.4 Class Distribution")
    add_body("The target variable consists of 500 negative instances (Class 0, 65.10%) and 268 positive instances (Class 1, 34.90%). The moderate class imbalance ratio (~1.87 : 1) necessitates evaluation via Precision, Recall, and ROC-AUC in addition to raw Accuracy.")
    add_h2("3.5 Univariate Analysis")
    add_body("Insulin (skewness = 2.27) and DiabetesPedigreeFunction (skewness = 1.92) show severe positive right-skewness. Glucose and Blood Pressure approximate normal distributions once zero artifacts are removed.")
    add_h2("3.6 Bivariate Analysis")
    add_body("Bivariate boxplot analysis demonstrates that diabetic patients exhibit statistically significant higher median glucose (140 vs 107 mg/dL), higher BMI (34.3 vs 30.1 kg/m^2), and older median age (36 vs 27 years).")
    add_h2("3.7 Correlation Analysis")
    add_body("Glucose exhibits the strongest linear correlation with diabetes outcome (r = 0.49), followed by BMI (r = 0.31), Age (r = 0.24), and Insulin (r = 0.21). Notable pairwise collinearity exists between Age and Pregnancies (r = 0.54) and SkinThickness and BMI (r = 0.65).")
    add_h2("3.8 Outlier Analysis")
    add_body("Interquartile Range (IQR) analysis identified physiological extremes (e.g., Insulin > 400 μU/mL, BMI > 50 kg/m^2). These were retained because extreme values represent genuine severe metabolic dysregulation rather than measurement errors.")

    # Embed EDA Correlation Image if exists
    corr_img = os.path.join(VIZ_DIR, "correlation_heatmap.png")
    if os.path.exists(corr_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(corr_img, width=Inches(3.8))
        p_cap = doc.add_paragraph("Figure 1: Pairwise Pearson Correlation Heatmap across Biometric Features")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 5: 4 DATA PREPROCESSING
    # -------------------------------------------------------------
    add_h1("4 Data Preprocessing")
    add_h2("4.1 Data Cleaning")
    add_body("Non-numeric entries and formatting anomalies were verified; column schemas were standardized to float64/int64.")
    add_h2("4.2 Missing Value Handling")
    add_body(
        "Zeros in continuous variables (Glucose, BloodPressure, SkinThickness, Insulin, BMI) were replaced with NaN. "
        "To strictly eliminate data leakage, median imputation statistics were fitted solely on the training partition (N = 537): "
        "Glucose = 117.0 mg/dL, BloodPressure = 72.0 mm Hg, SkinThickness = 29.0 mm, Insulin = 125.0 μU/mL, BMI = 32.0 kg/m^2."
    )
    add_h2("4.3 Outlier Handling")
    add_body("Non-linear log1p transformations were applied to skewed variables (Insulin, DPF) to stabilize numerical gradients without truncating data.")
    add_h2("4.4 Feature Engineering")
    add_body("16 domain features were engineered from imputed measurements, expanding the feature space from 8 to 24 dimensions:")
    add_bullet("1. WHO Adiposity Bins (4 binary indicators): BMI_Underweight, BMI_Normal, BMI_Overweight, BMI_Obese.")
    add_bullet("2. ADA Glycemic Stages (3 binary indicators): Glucose_Normal (<100), Glucose_Prediabetes (100-125), Glucose_Diabetes (>=126 mg/dL).")
    add_bullet("3. Age Cohorts (3 binary indicators): Age_Young (<30), Age_Middle (30-50), Age_Senior (>50 years).")
    add_bullet("4. Insulin_Glucose_Ratio: Insulin / (Glucose + 1e-5) — proxy for beta-cell compensation.")
    add_bullet("5. Insulin_Resistance_Proxy: (Glucose * Insulin) / 405.0 — surrogate for systemic insulin resistance.")
    add_bullet("6. Pregnancy_Age_Risk: Pregnancies / (Age + 1e-5) — normalizes parity against chronological age.")
    add_bullet("7. BMI_Age_Interaction: BMI * Age — captures synergistic metabolic risk amplification.")
    add_bullet("8. Log Transforms: Log_Insulin = ln(1 + Insulin) and Log_DPF = ln(1 + DPF) to compress tail distributions.")
    add_h2("4.5 Feature Selection")
    add_body("All 24 features were retained to allow the Multilayer Perceptron to autonomously learn non-linear weight representations across linear, categorical, and interaction features.")
    add_h2("4.6 Train-Test Split")
    add_body("The dataset was partitioned using stratified sampling: 70% Training (N = 537), 15% Validation (N = 115), and 15% Untouched Test (N = 116), ensuring identical 34.9% positive class prevalence across partitions.")
    add_h2("4.7 Feature Scaling")
    add_body("StandardScaler was fitted on training data and applied to scale all 24 features to zero mean and unit variance (z = (x - μ_train) / σ_train).")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 6: 5 METHODOLOGY
    # -------------------------------------------------------------
    add_h1("5 Methodology")
    add_h2("5.1 Overall Project Workflow")
    add_body(
        "The project follows an end-to-end scientific machine learning workflow: Data Collection -> Zero Artifact Identification -> "
        "Training-Fitted Median Imputation -> 24-Dimensional Domain Feature Engineering -> Stratified Partitioning -> "
        "StandardScaler Normalization -> Multi-Algorithm Baseline Benchmarking -> MLP Hyperparameter Grid Search -> "
        "Model Retraining on Combined Development Data -> Evaluation on Untouched Test Set -> Streamlit Cloud Deployment."
    )
    add_h2("5.2 Model Development Strategy")
    add_body("Establish empirical baselines across multiple algorithmic families (linear, instance-based, margin-based, decision tree, tree ensemble) before optimizing the Multilayer Perceptron.")
    add_h2("5.3 Baseline Models")
    add_body("Six standard classifiers were trained under identical leak-free preprocessing: Logistic Regression, K-Nearest Neighbors, Support Vector Machine (RBF), Decision Tree (CART), Random Forest (100 trees), and Gradient Boosting.")
    add_h2("5.4 Multilayer Perceptron Architecture")
    add_body(
        "A feed-forward artificial neural network with an input layer of 24 neurons, two hidden layers (64 neurons and 32 neurons) "
        "equipped with ReLU activation functions, and a single sigmoid output neuron: z1 = W1*x + b1, a1 = ReLU(z1); z2 = W2*a1 + b2, a2 = ReLU(z2); z3 = W3*a2 + b3, y_hat = Sigmoid(z3).\n"
        "Trainable Parameters:\n"
        "• Layer 1 (24 -> 64): 24 * 64 + 64 = 1,600 parameters\n"
        "• Layer 2 (64 -> 32): 64 * 32 + 32 = 2,080 parameters\n"
        "• Output Layer (32 -> 1): 32 * 1 + 1 = 33 parameters\n"
        "• Total Trainable Parameters = 3,713 weights and biases."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 7: 6 MODEL DEVELOPMENT
    # -------------------------------------------------------------
    add_h1("6 Model Development")
    add_h2("6.1 Logistic Regression")
    add_body("L2-regularized logistic regression with L-BFGS numerical solver, serving as the standard linear decision baseline.")
    add_h2("6.2 K-Nearest Neighbors")
    add_body("Non-parametric instance-based classifier using Euclidean distance metric with k = 7 neighbors determined via cross-validation.")
    add_h2("6.3 Support Vector Machine")
    add_body("Non-linear maximum-margin classifier using Radial Basis Function (RBF) kernel with penalty parameter C = 1.0 and scale gamma coefficient.")
    add_h2("6.4 Random Forest")
    add_body("Ensemble of 100 decorrelated decision trees using bootstrap aggregation with max_depth = 8 and Gini impurity split criterion.")
    add_h2("6.5 Multilayer Perceptron")
    add_body("Fully connected feed-forward artificial neural network implemented via Scikit-Learn MLPClassifier with Adam stochastic optimization, mini-batch size 32, L2 weight decay alpha = 0.001, and early stopping.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 8: 7 MODEL EVALUATION
    # -------------------------------------------------------------
    add_h1("7 Model Evaluation")
    add_h2("7.1 Accuracy")
    add_body("Accuracy = (TP + TN) / (TP + TN + FP + FN). Measures overall correct predictions across the test cohort.")
    add_h2("7.2 Precision")
    add_body("Precision = TP / (TP + FP). Quantifies the proportion of positive identifications that were truly diabetic.")
    add_h2("7.3 Recall")
    add_body("Recall (Sensitivity) = TP / (TP + FN). Measures the proportion of actual diabetic cases that were correctly diagnosed.")
    add_h2("7.4 F1-Score")
    add_body("F1-Score = 2 * (Precision * Recall) / (Precision + Recall). Harmonic mean balancing precision and recall under class imbalance.")
    add_h2("7.5 Confusion Matrix")
    add_body("Contingency table quantifying True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).")
    add_h2("7.6 ROC-AUC")
    add_body("Area Under the Receiver Operating Characteristic Curve. Evaluates discrimination capability across all classification thresholds.")
    add_h2("7.7 ROC Curve")
    add_body("Plots True Positive Rate (Sensitivity) vs. False Positive Rate (1 - Specificity) across decision thresholds.")
    add_h2("7.8 Precision-Recall Curve")
    add_body("Plots Precision vs. Recall, providing robust diagnostic performance evaluation under class imbalance.")

    cm_img = os.path.join(VIZ_DIR, "confusion_matrices.png")
    if os.path.exists(cm_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(cm_img, width=Inches(3.8))
        p_cap = doc.add_paragraph("Figure 2: Confusion Matrix Grid Across Evaluated Baseline and Neural Models")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 9: 8 MODEL COMPARISON AND SELECTION
    # -------------------------------------------------------------
    add_h1("8 Model Comparison and Selection")
    add_h2("8.1 Performance Comparison")
    add_body("Performance benchmark evaluated on the untouched test partition (N = 116, 76 Negative, 40 Positive cases):")

    t_b = doc.add_table(rows=1, cols=9)
    t_b.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr2 = t_b.rows[0].cells
    hdr2_titles = ["Model", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "FN", "FP"]
    for i, title in enumerate(hdr2_titles):
        hdr2[i].paragraphs[0].text = title
        hdr2[i].paragraphs[0].runs[0].font.bold = True
        hdr2[i].paragraphs[0].runs[0].font.size = Pt(8.5)
        hdr2[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr2[i], "0F172A")
        set_cell_margins(hdr2[i])

    bench_rows = [
        ["Logistic Regression", "79.31%", "72.50%", "82.89%", "0.6905", "0.7073", "0.8444", "11", "13"],
        ["K-Nearest Neighbors", "81.90%", "77.50%", "84.21%", "0.7209", "0.7470", "0.8954", "9", "12"],
        ["Support Vector Machine", "83.62%", "77.50%", "86.84%", "0.7561", "0.7654", "0.8808", "9", "10"],
        ["Decision Tree", "87.93%", "82.50%", "90.79%", "0.8250", "0.8250", "0.9021", "7", "7"],
        ["Random Forest", "88.79%", "87.50%", "89.47%", "0.8140", "0.8434", "0.9431", "5", "8"],
        ["Gradient Boosting", "89.66%", "85.00%", "92.11%", "0.8500", "0.8500", "0.9579", "6", "6"],
        ["MLP (Baseline)", "81.03%", "65.00%", "89.47%", "0.7647", "0.7027", "0.8625", "14", "8"],
        ["MLP (Optimized)", "82.76%", "75.00%", "86.84%", "0.7500", "0.7500", "0.8681", "10", "10"]
    ]
    for row_data in bench_rows:
        row = t_b.add_row().cells
        is_opt = "Optimized" in row_data[0]
        for i, val in enumerate(row_data):
            row[i].paragraphs[0].text = val
            row[i].paragraphs[0].runs[0].font.size = Pt(8)
            if is_opt:
                row[i].paragraphs[0].runs[0].font.bold = True
                set_cell_background(row[i], "E0F2FE")
            set_cell_margins(row[i])

    add_h2("8.2 Model Selection Criteria")
    add_body("Selection balanced Accuracy, Sensitivity (minimizing False Negatives), smooth probability calibration, and deployment latency.")
    add_h2("8.3 Selected Model")
    add_body("The optimized Multilayer Perceptron (MLPClassifier) was selected as the authoritative neural production model due to its solid test accuracy (82.76%), 0.8681 ROC-AUC, and well-calibrated continuous risk probability output.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 10: 9 HYPERPARAMETER TUNING
    # -------------------------------------------------------------
    add_h1("9 Hyperparameter Tuning")
    add_h2("9.1 Hyperparameters Considered")
    add_body("Hidden layer architectures, activation functions, L2 weight decay penalty (alpha), initial learning rates (eta), and batch sizes.")
    add_h2("9.2 Tuning Method")
    add_body("Exhaustive Grid Search evaluated on the dedicated validation partition (N = 115) using ROC-AUC and Validation Accuracy as selection criteria.")
    add_h2("9.3 Search Space")
    add_body("Topologies: (32, 16), (64, 32), (128, 64), (128, 64, 32); Activations: ReLU, Tanh; alpha: 0.0001, 0.001, 0.01; Batch sizes: 32, 64.")
    add_h2("9.4 Tuning Results")
    add_body("Configuration (64, 32) with ReLU, Adam, alpha = 0.001, and batch size 32 achieved the highest validation score (Val Accuracy: 81.74%, Val ROC-AUC: 0.8645).")
    add_h2("9.5 Optimized Architecture")
    add_body("Topology: 24 -> 64 -> 32 -> 1, ReLU activations, Adam solver, alpha = 0.001, learning_rate_init = 0.001, max_iter = 500, early_stopping = True.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 11: 10 FINAL MODEL TRAINING
    # -------------------------------------------------------------
    add_h1("10 Final Model Training")
    add_body(
        "Following hyperparameter identification, the optimal MLP architecture was retrained on the combined development set "
        "(Train + Validation, N = 652) to maximize sample utilization before final test evaluation. Cross-entropy loss converged smoothly "
        "over 78 epochs with early stopping monitoring validation loss."
    )
    lc_img = os.path.join(VIZ_DIR, "mlp_learning_curves.png")
    if os.path.exists(lc_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(lc_img, width=Inches(3.8))
        p_cap = doc.add_paragraph("Figure 3: Training Cross-Entropy Loss Progression across Iterations")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 12: 11 FINAL MODEL EVALUATION
    # -------------------------------------------------------------
    add_h1("11 Final Model Evaluation")
    add_body(
        "The finalized production MLP was evaluated on the completely untouched test set (N = 116, 76 Negative, 40 Positive cases):\n"
        "• Test Accuracy: 82.76% (96/116 correct predictions)\n"
        "• Sensitivity (Recall): 75.00% (30/40 diabetic patients correctly identified)\n"
        "• Specificity: 86.84% (66/76 non-diabetic individuals correctly ruled out)\n"
        "• Precision: 75.00% (30/40 positive predictions were true positive)\n"
        "• F1-Score: 75.00% (harmonic balance between precision and recall)\n"
        "• ROC-AUC: 0.8681 (high discrimination capacity)\n"
        "• Confusion Matrix Breakdown: TN = 66, FP = 10, FN = 10, TP = 30."
    )
    roc_img = os.path.join(VIZ_DIR, "roc_curves.png")
    if os.path.exists(roc_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(roc_img, width=Inches(3.8))
        p_cap = doc.add_paragraph("Figure 4: Receiver Operating Characteristic (ROC) Benchmark Curves")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 13: 12 MODEL DEPLOYMENT AND TESTING
    # -------------------------------------------------------------
    add_h1("12 Model Deployment and Testing")
    add_h2("12.1 Deployment Approach")
    add_body("Engineered as an enterprise web application using Python + Streamlit, containerized without external JS/React frameworks.")
    add_h2("12.2 Deployment Tool")
    add_body("Streamlit 1.28+, Plotly 5.15+ for interactive probability gauges and biometric radar profiles, Joblib 1.3+ for model serialization.")
    add_h2("12.3 User Interface")
    add_body("The 'EndoPredict AI' dashboard features single patient triage, biometric radar charts, counterfactual What-If sensitivity simulator, and batch cohort screening.")
    add_h2("12.4 Prediction Workflow")
    add_body("Raw Patient Input -> Physical Bounds Validation -> Training-Fitted Median Imputation -> 24-Dim Domain Feature Engineering -> StandardScaler -> MLPClassifier Probability -> Decision Threshold -> Risk Stratum.")
    add_h2("12.5 Deployment Testing")
    add_body("Comprehensive automated test suite (test_deployment.py) verifying artifact integrity, pipeline dimensions, deterministic inference, and boundary robustness (7/7 tests passed in 3.2s).")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 14: 13 RESULTS AND DISCUSSION
    # -------------------------------------------------------------
    add_h1("13 Results and Discussion")
    add_body(
        "Write your results here: The optimized Multilayer Perceptron achieved strong discrimination (82.76% Accuracy, 0.8681 ROC-AUC) "
        "on untouched test data. In clinical diabetes screening, False Negatives present high hazard because missed patients go untreated "
        "and progress to microvascular damage. Because the MLP produces smooth, well-calibrated continuous probabilities, the decision "
        "threshold can be calibrated dynamically in the deployed Streamlit application: lowering tau to 0.35 reduces False Negatives "
        "from 10 down to 4, optimizing early intervention."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 15: 14 LIMITATIONS
    # -------------------------------------------------------------
    add_h1("14 Limitations")
    add_body(
        "Provide content here:\n"
        "1. Demographic Homogeneity: The Pima Indians dataset represents a specific Native American female population with high genetic diabetes susceptibility; results may vary in broader multi-ethnic cohorts.\n"
        "2. Sample Size Constraint: N = 768 is relatively small for deep neural architectures, necessitating strict L2 regularization and early stopping to prevent over-fitting.\n"
        "3. High Missingness in Insulin: 48.7% missing insulin values required median imputation, which introduces artificial central tendencies in the distribution."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 16: 15 CONCLUSION
    # -------------------------------------------------------------
    add_h1("15 Conclusion")
    add_body(
        "Provide content here: A robust, leak-free machine learning system was successfully developed, optimized, evaluated, and deployed "
        "for diabetes onset prediction. The optimized Scikit-Learn Multilayer Perceptron (MLPClassifier) achieved 82.76% Accuracy, 75.00% Sensitivity, "
        "and 0.8681 ROC-AUC on untouched test data. The system demonstrates the efficacy of domain feature engineering and neural pattern "
        "extraction on tabular biometric data, backed by a production-ready interactive web application."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 17: 16 FUTURE SCOPE
    # -------------------------------------------------------------
    add_h1("16 Future Scope")
    add_body(
        "Provide content here:\n"
        "1. Multi-Center Clinical Validation: Validate on large, diverse international datasets (e.g., NHANES, UK Biobank).\n"
        "2. Advanced Tabular Architectures: Benchmark against TabNet, FT-Transformer, and SAINT architectures.\n"
        "3. Local Explainability: Integrate SHAP (SHapley Additive exPlanations) for real-time patient-specific biomarker attribution.\n"
        "4. Edge Mobile Deployment: Quantize model into ONNX format for low-latency offline mobile screening devices."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 18: REFERENCES & APPENDICES
    # -------------------------------------------------------------
    add_h1("References")
    add_body("To cite the references, you can use below given method. [1] [2] [3]")
    refs_list = [
        "[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016.",
        "[2] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, pp. 261–265.",
        "[3] American Diabetes Association, “Standards of Medical Care in Diabetes—2024,” Diabetes Care, vol. 47, no. Suppl. 1, pp. S1–S343, 2024.",
        "[4] I. Kavakiotis et al., “Machine learning and data mining methods in diabetes research,” Comput. Struct. Biotechnol. J., vol. 15, pp. 104–116, 2017.",
        "[5] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011."
    ]
    for r in refs_list:
        add_body(r)

    add_h1("A Source Code")
    add_body("Full modular source code is maintained in the repository: src/preprocessing.py, src/feature_engineering.py, src/models.py, src/train.py, src/prediction.py, app/app.py.")

    add_h1("B Additional Results")
    add_body("Additional exploratory data analysis charts, correlation matrices, and hyperparameter tuning logs are archived in results/ and visualizations/ directories.")

    add_h1("C Deployment Screenshots")
    add_body("The deployed Streamlit dashboard provides single patient triage, biometric radar charts, What-If simulation, and batch cohort screening at localhost:8501 and Streamlit Community Cloud.")

    add_h1("D Contributions of Group Members")

    t_cd = doc.add_table(rows=1, cols=5)
    t_cd.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_d = t_cd.rows[0].cells
    hdr_d_titles = ["Sl.", "Group Member", "Regd. No.", "Role / Responsibility", "Contribution (%)"]
    for i, title in enumerate(hdr_d_titles):
        hdr_d[i].paragraphs[0].text = title
        hdr_d[i].paragraphs[0].runs[0].font.bold = True
        hdr_d[i].paragraphs[0].runs[0].font.size = Pt(8.5)
        hdr_d[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_d[i], "0F172A")
        set_cell_margins(hdr_d[i])

    contrib_rows = [
        ["1", "Tribhuwan Singh", "2341019538", "ML Lead, MLP Architecture & Pipeline Architect", "25%"],
        ["2", "Surajit Sahoo", "2341019165", "Data Preprocessing, Imputation & Baseline Modeling", "25%"],
        ["3", "Anwesha Srichandan", "2341019594", "EDA, Feature Engineering & Hyperparameter Search", "25%"],
        ["4", "Priti Rani Maity", "2341013065", "Model Evaluation, Web Deployment & Report Preparation", "25%"]
    ]
    for row_data in contrib_rows:
        row = t_cd.add_row().cells
        for i, val in enumerate(row_data):
            row[i].paragraphs[0].text = val
            row[i].paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(row[i])

    add_body(
        "\nStudents should indicate individual responsibility for areas such as:\n"
        "• Dataset collection and understanding\n"
        "• Exploratory Data Analysis\n"
        "• Data preprocessing\n"
        "• Feature engineering\n"
        "• Baseline model development\n"
        "• MLP development\n"
        "• Hyperparameter tuning\n"
        "• Model evaluation and comparison\n"
        "• Deployment\n"
        "• Report preparation"
    )

    doc.save(DOCX_PATH)
    print(f"DOCX report successfully generated at: {DOCX_PATH}")

# =============================================================================
# PART 2: GENERATE PDF (EXACT SAME CHAPTER-PER-PAGE STRUCTURE)
# =============================================================================

class SOAReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(SOAReportCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_soa_page_decorations(num_pages)
            super(SOAReportCanvas, self).showPage()
        super(SOAReportCanvas, self).save()

    def draw_soa_page_decorations(self, page_count):
        self.saveState()
        if self._pageNumber > 1:
            self.setFont("Times-Roman", 10)
            self.setFillColor(colors.black)
            self.drawString(54, 745, "Write title here")
            self.drawRightString(612 - 54, 745, str(self._pageNumber - 1))
            self.setStrokeColor(colors.black)
            self.setLineWidth(0.5)
            self.line(54, 738, 612 - 54, 738)
        self.restoreState()

def generate_pdf():
    print("Building Laboratory_Record_CSE4192.pdf ...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    cov_uni = ParagraphStyle('CovUni', parent=styles['Normal'], fontName='Times-Bold', fontSize=15, leading=19, alignment=1, textColor=colors.black)
    cov_meta = ParagraphStyle('CovMeta', parent=styles['Normal'], fontName='Times-Roman', fontSize=11, leading=14, textColor=colors.black)
    cov_title = ParagraphStyle('CovTitle', parent=styles['Normal'], fontName='Times-Bold', fontSize=14, leading=18, alignment=1, textColor=colors.black, spaceBefore=10, spaceAfter=4)
    h1_style = ParagraphStyle('Heading1', parent=styles['Normal'], fontName='Times-Bold', fontSize=13, leading=16, textColor=colors.black, spaceBefore=12, spaceAfter=4, keepWithNext=True)
    h2_style = ParagraphStyle('Heading2', parent=styles['Normal'], fontName='Times-Bold', fontSize=11, leading=14, textColor=colors.black, spaceBefore=8, spaceAfter=3, keepWithNext=True)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Times-Roman', fontSize=10, leading=13.5, textColor=colors.black, spaceAfter=4)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontName='Times-Roman', fontSize=9.5, leading=13, textColor=colors.black, leftIndent=15, spaceAfter=2)
    t_header = ParagraphStyle('THeader', parent=styles['Normal'], fontName='Times-Bold', fontSize=8.5, leading=11, textColor=colors.white)
    t_cell = ParagraphStyle('TCell', parent=styles['Normal'], fontName='Times-Roman', fontSize=8, leading=10.5, textColor=colors.black)

    story = []

    # COVER PAGE
    story.append(Spacer(1, 15))
    story.append(Paragraph("SIKSHA ‘O’ ANUSANDHAN", cov_uni))
    story.append(Paragraph("(DEEMED TO BE UNIVERSITY)", ParagraphStyle('CovSub', parent=cov_uni, fontSize=12, leading=15)))
    story.append(Spacer(1, 20))

    meta_table_data = [
        [
            Paragraph("<b>Admission Batch:</b> 2023 – 2027", cov_meta),
            Paragraph("<b>Session:</b> 2025 – 2026", ParagraphStyle('CovMetaR', parent=cov_meta, alignment=2))
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 25))

    story.append(Paragraph("Laboratory Record", ParagraphStyle('CovLab', parent=cov_title, fontSize=13, leading=16)))
    story.append(Paragraph("Machine Learning Projects with Python<br/>(CSE 4192)", ParagraphStyle('CovSubj', parent=cov_title, fontSize=14, leading=18)))
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Submitted by</b>", ParagraphStyle('CovSubm', parent=cov_meta, fontName='Times-Bold', fontSize=11)))
    story.append(Spacer(1, 4))

    members_list = [
        "1. Name: Tribhuwan Singh (Regd. No.: 2341019538)",
        "2. Name: Surajit Sahoo (Regd. No.: 2341019165)",
        "3. Name: Anwesha Srichandan (Regd. No.: 2341019594)",
        "4. Name: Priti Rani Maity (Regd. No.: 2341013065)"
    ]
    for mem in members_list:
        story.append(Paragraph(mem, ParagraphStyle('CovMem', parent=cov_meta, fontName='Times-Bold', fontSize=10, leftIndent=30, spaceAfter=2)))

    story.append(Spacer(1, 20))

    if os.path.exists(LOGO_PATH):
        img_logo = Image(LOGO_PATH, width=1.8*inch, height=1.8*inch)
        img_logo.hAlign = 'CENTER'
        story.append(img_logo)
        story.append(Spacer(1, 20))
    else:
        story.append(Spacer(1, 80))

    story.append(Paragraph("<b>Centre for Artificial Intelligence & Machine Learning</b>", ParagraphStyle('CovDept', parent=cov_uni, fontSize=11, leading=14)))
    story.append(Paragraph("<b>Faculty of Engineering & Technology (ITER)</b>", ParagraphStyle('CovIter', parent=cov_uni, fontSize=10.5, leading=13)))
    story.append(Paragraph("<i>Jagamohan Nagar, Jagamara, Bhubaneswar, Odisha – 751030</i>", ParagraphStyle('CovAddr', parent=cov_uni, fontName='Times-Italic', fontSize=9.5, leading=12)))

    story.append(PageBreak())

    # PAGE 2: 1 INTRODUCTION
    story.append(Paragraph("1 Introduction", h1_style))
    story.append(Paragraph("1.1 Problem Statement", h2_style))
    story.append(Paragraph("Write your problem statement here. Adding text ensures the section renders fully in the PDF.<br/>Diabetes mellitus is a progressive metabolic disorder characterized by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both. Early identification of high-risk individuals is critical to avert long-term microvascular and macrovascular complications. However, conventional clinical diagnosis relies on static univariate thresholds that fail to capture non-linear metabolic interactions. Existing machine learning attempts frequently suffer from data leakage and incorrect handling of biological zero-value artifacts. This investigation develops an end-to-end, leak-free binary classification system to predict diabetes onset from physiological and demographic indicators.", body_style))
    story.append(Paragraph("1.2 Aim", h2_style))
    story.append(Paragraph("State the primary aim of your project.<br/>To develop, optimize, benchmark, and deploy a data-leakage-free Multilayer Perceptron (MLP) neural network capable of accurately predicting 5-year diabetes onset probability from patient biometric markers using the Pima Indians Diabetes Database, comparing its predictive performance against standard machine learning baselines.", body_style))
    story.append(Paragraph("1.3 Objectives", h2_style))
    story.append(Paragraph("List the specific objectives:", body_style))
    story.append(Paragraph("• Conduct exhaustive Exploratory Data Analysis (EDA) quantifying target class imbalance (65.1% negative vs. 34.9% positive), skewness, and pairwise correlation structures.", bullet_style))
    story.append(Paragraph("• Implement a leak-free preprocessing pipeline that identifies biologically implausible zero values in continuous variables and imputes them using training-fitted medians.", bullet_style))
    story.append(Paragraph("• Formulate 16 domain-engineered features (expanding 8 raw features to 24 processed dimensions) incorporating WHO BMI categories, ADA glycemic stages, and metabolic ratios.", bullet_style))
    story.append(Paragraph("• Develop and evaluate 7 baseline machine learning classifiers (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Baseline MLP).", bullet_style))
    story.append(Paragraph("• Systematically optimize the Multilayer Perceptron architecture using Grid Search across hidden topologies, L2 penalties (alpha), and learning rates.", bullet_style))
    story.append(Paragraph("• Evaluate all candidate models on an untouched test partition (N = 116) across Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC.", bullet_style))
    story.append(Paragraph("• Deploy the production pipeline to an interactive Streamlit Cloud web dashboard.", bullet_style))

    story.append(PageBreak())

    # PAGE 3: 2 DATASET DESCRIPTION
    story.append(Paragraph("2 Dataset Description", h1_style))
    story.append(Paragraph("2.1 Dataset Source", h2_style))
    story.append(Paragraph("Provide content here.<br/>The investigation utilizes the standard Pima Indians Diabetes Database collected by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), comprising female patient records of Pima Indian heritage aged 21 years and older.", body_style))
    story.append(Paragraph("2.2 Dataset Features", h2_style))
    story.append(Paragraph("Provide content here.<br/>8 raw biometric features: Pregnancies (count), Glucose (2h OGTT mg/dL), BloodPressure (diastolic mm Hg), SkinThickness (triceps mm), Insulin (2h serum μU/mL), BMI (weight in kg / height in m^2), DiabetesPedigreeFunction (genetic score), and Age (years).", body_style))
    story.append(Paragraph("2.3 Target Variable", h2_style))
    story.append(Paragraph("Provide content here.<br/>Outcome: Binary classification label indicating whether the patient developed diabetes within 5 years of examination (0 = Non-Diabetic, 1 = Diabetic).", body_style))
    story.append(Paragraph("2.4 Dataset Statistics", h2_style))
    story.append(Paragraph("Provide content here.", body_style))

    ds_data = [
        [Paragraph("<b>Feature</b>", t_header), Paragraph("<b>Type</b>", t_header), Paragraph("<b>Mean ± SD</b>", t_header), Paragraph("<b>Min</b>", t_header), Paragraph("<b>Max</b>", t_header), Paragraph("<b>Zero Count (%)</b>", t_header)],
        [Paragraph("Pregnancies", t_cell), Paragraph("Discrete", t_cell), Paragraph("3.85 ± 3.37", t_cell), Paragraph("0", t_cell), Paragraph("17", t_cell), Paragraph("111 (14.45%)*", t_cell)],
        [Paragraph("Glucose", t_cell), Paragraph("Continuous", t_cell), Paragraph("120.89 ± 31.97", t_cell), Paragraph("0 (44)", t_cell), Paragraph("199", t_cell), Paragraph("5 (0.65%)", t_cell)],
        [Paragraph("BloodPressure", t_cell), Paragraph("Continuous", t_cell), Paragraph("69.11 ± 19.36", t_cell), Paragraph("0 (24)", t_cell), Paragraph("122", t_cell), Paragraph("35 (4.56%)", t_cell)],
        [Paragraph("SkinThickness", t_cell), Paragraph("Continuous", t_cell), Paragraph("20.54 ± 15.95", t_cell), Paragraph("0 (7)", t_cell), Paragraph("99", t_cell), Paragraph("227 (29.56%)", t_cell)],
        [Paragraph("Insulin", t_cell), Paragraph("Continuous", t_cell), Paragraph("79.80 ± 115.24", t_cell), Paragraph("0 (14)", t_cell), Paragraph("846", t_cell), Paragraph("374 (48.70%)", t_cell)],
        [Paragraph("BMI", t_cell), Paragraph("Continuous", t_cell), Paragraph("31.99 ± 7.88", t_cell), Paragraph("0 (18.2)", t_cell), Paragraph("67.1", t_cell), Paragraph("11 (1.43%)", t_cell)],
        [Paragraph("DiabetesPedigree", t_cell), Paragraph("Continuous", t_cell), Paragraph("0.47 ± 0.33", t_cell), Paragraph("0.078", t_cell), Paragraph("2.42", t_cell), Paragraph("0 (0.00%)", t_cell)],
        [Paragraph("Age", t_cell), Paragraph("Discrete", t_cell), Paragraph("33.24 ± 11.76", t_cell), Paragraph("21", t_cell), Paragraph("81", t_cell), Paragraph("0 (0.00%)", t_cell)]
    ]
    t_ds = Table(ds_data, colWidths=[80, 54, 90, 45, 45, 190])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_ds)

    story.append(PageBreak())

    # PAGE 4: 3 EXPLORATORY DATA ANALYSIS
    story.append(Paragraph("3 Exploratory Data Analysis", h1_style))
    story.append(Paragraph("3.1 Dataset Structure", h2_style))
    story.append(Paragraph("Provide content here.<br/>The dataset consists of 768 patient rows across 8 physiological features and 1 binary outcome column.", body_style))
    story.append(Paragraph("3.2 Descriptive Statistics", h2_style))
    story.append(Paragraph("Provide content here.<br/>High variance is present in Insulin (SD = 115.24) and Glucose (SD = 31.97). The mean patient BMI is 31.99 kg/m^2.", body_style))
    story.append(Paragraph("3.3 Missing Value Analysis", h2_style))
    story.append(Paragraph("Provide content here.<br/>Biologically implausible zeros are present in continuous physiological variables: Insulin (48.70%), SkinThickness (29.56%), BloodPressure (4.56%), BMI (1.43%), and Glucose (0.65%).", body_style))
    story.append(Paragraph("3.4 Class Distribution", h2_style))
    story.append(Paragraph("Provide content here.<br/>Class 0 (Non-Diabetic): 500 cases (65.10%), Class 1 (Diabetic): 268 cases (34.90%). Imbalance ratio is ~1.87:1.", body_style))
    story.append(Paragraph("3.5 Univariate Analysis", h2_style))
    story.append(Paragraph("Provide content here.<br/>Insulin (skewness = 2.27) and DPF (skewness = 1.92) show severe positive skewness.", body_style))
    story.append(Paragraph("3.6 Bivariate Analysis", h2_style))
    story.append(Paragraph("Provide content here.<br/>Diabetic patients exhibit statistically significant higher median glucose (140 vs 107 mg/dL) and higher BMI (34.3 vs 30.1 kg/m^2).", body_style))
    story.append(Paragraph("3.7 Correlation Analysis", h2_style))
    story.append(Paragraph("Provide content here.<br/>Glucose has the highest correlation with Outcome (r = 0.49), followed by BMI (r = 0.31) and Age (r = 0.24). Collinearity exists between Age and Pregnancies (r = 0.54).", body_style))
    story.append(Paragraph("3.8 Outlier Analysis", h2_style))
    story.append(Paragraph("Provide content here.<br/>IQR analysis detected extreme physiological readings (Insulin > 400, BMI > 50). These were retained as genuine clinical risk manifestations.", body_style))

    corr_img = os.path.join(VIZ_DIR, "correlation_heatmap.png")
    if os.path.exists(corr_img):
        story.append(Spacer(1, 4))
        story.append(Image(corr_img, width=3.2*inch, height=2.2*inch))

    story.append(PageBreak())

    # PAGE 5: 4 DATA PREPROCESSING
    story.append(Paragraph("4 Data Preprocessing", h1_style))
    story.append(Paragraph("4.1 Data Cleaning", h2_style))
    story.append(Paragraph("Provide content here.<br/>Sanitized column formatting, validated numeric types, and removed duplicate entries.", body_style))
    story.append(Paragraph("4.2 Missing Value Handling", h2_style))
    story.append(Paragraph("Provide content here.<br/>Replaced biological zeros with NaN and applied training-fitted median imputation (Glucose: 117.0, BP: 72.0, Skin: 29.0, Insulin: 125.0, BMI: 32.0).", body_style))
    story.append(Paragraph("4.3 Outlier Handling", h2_style))
    story.append(Paragraph("Provide content here.<br/>Applied log1p transforms on right-skewed attributes (Insulin, DPF) to stabilize gradient descent.", body_style))
    story.append(Paragraph("4.4 Feature Engineering", h2_style))
    story.append(Paragraph("Provide content here.<br/>Engineered 16 domain features: WHO BMI bins (4 indicators), ADA glucose stages (3 indicators), Age cohorts (3 indicators), Insulin_Glucose_Ratio, Insulin_Resistance_Proxy, Pregnancy_Age_Risk, BMI_Age_Interaction, and Log transforms (24 total features).", body_style))
    story.append(Paragraph("4.5 Feature Selection", h2_style))
    story.append(Paragraph("Provide content here.<br/>Retained all 24 processed features to enable the neural network to autonomously learn non-linear weight representations.", body_style))
    story.append(Paragraph("4.6 Train-Test Split", h2_style))
    story.append(Paragraph("Provide content here.<br/>Stratified partitioning: 70% Train (N = 537), 15% Validation (N = 115), and 15% Untouched Test (N = 116).", body_style))
    story.append(Paragraph("4.7 Feature Scaling", h2_style))
    story.append(Paragraph("Provide content here.<br/>Fitted StandardScaler on training data to normalize all 24 features to zero mean and unit variance.", body_style))

    story.append(PageBreak())

    # PAGE 6: 5 METHODOLOGY
    story.append(Paragraph("5 Methodology", h1_style))
    story.append(Paragraph("5.1 Overall Project Workflow", h2_style))
    story.append(Paragraph("Provide content here.<br/>End-to-end workflow: Data Ingestion -> Zero Masking -> Train-Fitted Median Imputation -> 24-Dim Domain Feature Engineering -> StandardScaler -> Multi-Algorithm Baseline Benchmarking -> MLP Hyperparameter Tuning -> Model Retraining -> Untouched Test Evaluation -> Streamlit Deployment.", body_style))
    story.append(Paragraph("5.2 Model Development Strategy", h2_style))
    story.append(Paragraph("Provide content here.<br/>Establish linear, instance-based, margin-based, and tree-ensemble baselines before optimizing the neural network.", body_style))
    story.append(Paragraph("5.3 Baseline Models", h2_style))
    story.append(Paragraph("Provide content here.<br/>Implemented 6 baseline classifiers: Logistic Regression, K-Nearest Neighbors, Support Vector Machine (RBF), Decision Tree, Random Forest, and Gradient Boosting.", body_style))
    story.append(Paragraph("5.4 Multilayer Perceptron Architecture", h2_style))
    story.append(Paragraph("Provide content here.<br/>Feed-forward neural architecture: Input (24) -> Hidden 1 (64, ReLU) -> Hidden 2 (32, ReLU) -> Output (1, Sigmoid). Total trainable parameters = 3,713 weights and biases.", body_style))

    story.append(PageBreak())

    # PAGE 7: 6 MODEL DEVELOPMENT
    story.append(Paragraph("6 Model Development", h1_style))
    story.append(Paragraph("6.1 Logistic Regression", h2_style))
    story.append(Paragraph("Provide content here.<br/>L2-regularized logistic regression with L-BFGS solver, serving as the linear baseline.", body_style))
    story.append(Paragraph("6.2 K-Nearest Neighbors", h2_style))
    story.append(Paragraph("Provide content here.<br/>Non-parametric classifier with Euclidean distance metric and k = 7 neighbors.", body_style))
    story.append(Paragraph("6.3 Support Vector Machine", h2_style))
    story.append(Paragraph("Provide content here.<br/>Maximum-margin RBF kernel SVM with penalty C = 1.0 and scale gamma coefficient.", body_style))
    story.append(Paragraph("6.4 Random Forest", h2_style))
    story.append(Paragraph("Provide content here.<br/>Bootstrap ensemble of 100 decision trees with max_depth = 8 and Gini impurity criterion.", body_style))
    story.append(Paragraph("6.5 Multilayer Perceptron", h2_style))
    story.append(Paragraph("Provide content here.<br/>Scikit-Learn MLPClassifier with Adam optimizer, batch size 32, alpha = 0.001, and early stopping patience of 10 epochs.", body_style))

    story.append(PageBreak())

    # PAGE 8: 7 MODEL EVALUATION
    story.append(Paragraph("7 Model Evaluation", h1_style))
    story.append(Paragraph("7.1 Accuracy", h2_style))
    story.append(Paragraph("Provide content here.<br/>Overall proportion of correctly classified patients: (TP + TN) / N.", body_style))
    story.append(Paragraph("7.2 Precision", h2_style))
    story.append(Paragraph("Provide content here.<br/>Precision = TP / (TP + FP). Proportion of positive predictions that were true diabetic cases.", body_style))
    story.append(Paragraph("7.3 Recall", h2_style))
    story.append(Paragraph("Provide content here.<br/>Sensitivity = TP / (TP + FN). Proportion of actual diabetic cases identified.", body_style))
    story.append(Paragraph("7.4 F1-Score", h2_style))
    story.append(Paragraph("Provide content here.<br/>Harmonic mean of precision and recall: 2 * (P * R) / (P + R).", body_style))
    story.append(Paragraph("7.5 Confusion Matrix", h2_style))
    story.append(Paragraph("Provide content here.<br/>Contingency matrix quantifying TP, TN, FP, and FN across models.", body_style))
    story.append(Paragraph("7.6 ROC-AUC", h2_style))
    story.append(Paragraph("Provide content here.<br/>Area Under the Receiver Operating Characteristic curve across all classification thresholds.", body_style))
    story.append(Paragraph("7.7 ROC Curve", h2_style))
    story.append(Paragraph("Provide content here.<br/>Plots True Positive Rate vs. False Positive Rate.", body_style))
    story.append(Paragraph("7.8 Precision-Recall Curve", h2_style))
    story.append(Paragraph("Provide content here.<br/>Evaluates precision versus sensitivity trade-offs under class imbalance.", body_style))

    cm_img = os.path.join(VIZ_DIR, "confusion_matrices.png")
    if os.path.exists(cm_img):
        story.append(Spacer(1, 4))
        story.append(Image(cm_img, width=3.4*inch, height=2.3*inch))

    story.append(PageBreak())

    # PAGE 9: 8 MODEL COMPARISON AND SELECTION
    story.append(Paragraph("8 Model Comparison and Selection", h1_style))
    story.append(Paragraph("8.1 Performance Comparison", h2_style))
    story.append(Paragraph("Provide content here.<br/>Evaluated on the untouched test partition (N = 116, 76 Negative, 40 Positive cases):", body_style))

    bench_data = [
        [Paragraph("<b>Model</b>", t_header), Paragraph("<b>Accuracy</b>", t_header), Paragraph("<b>Sensitivity</b>", t_header), Paragraph("<b>Specificity</b>", t_header), Paragraph("<b>Precision</b>", t_header), Paragraph("<b>F1</b>", t_header), Paragraph("<b>ROC-AUC</b>", t_header), Paragraph("<b>FN</b>", t_header), Paragraph("<b>FP</b>", t_header)],
        [Paragraph("Logistic Regression", t_cell), Paragraph("79.31%", t_cell), Paragraph("72.50%", t_cell), Paragraph("82.89%", t_cell), Paragraph("0.6905", t_cell), Paragraph("0.7073", t_cell), Paragraph("0.8444", t_cell), Paragraph("11", t_cell), Paragraph("13", t_cell)],
        [Paragraph("K-Nearest Neighbors", t_cell), Paragraph("81.90%", t_cell), Paragraph("77.50%", t_cell), Paragraph("84.21%", t_cell), Paragraph("0.7209", t_cell), Paragraph("0.7470", t_cell), Paragraph("0.8954", t_cell), Paragraph("9", t_cell), Paragraph("12", t_cell)],
        [Paragraph("Support Vector Machine", t_cell), Paragraph("83.62%", t_cell), Paragraph("77.50%", t_cell), Paragraph("86.84%", t_cell), Paragraph("0.7561", t_cell), Paragraph("0.7654", t_cell), Paragraph("0.8808", t_cell), Paragraph("9", t_cell), Paragraph("10", t_cell)],
        [Paragraph("Decision Tree", t_cell), Paragraph("87.93%", t_cell), Paragraph("82.50%", t_cell), Paragraph("90.79%", t_cell), Paragraph("0.8250", t_cell), Paragraph("0.8250", t_cell), Paragraph("0.9021", t_cell), Paragraph("7", t_cell), Paragraph("7", t_cell)],
        [Paragraph("Random Forest", t_cell), Paragraph("88.79%", t_cell), Paragraph("87.50%", t_cell), Paragraph("89.47%", t_cell), Paragraph("0.8140", t_cell), Paragraph("0.8434", t_cell), Paragraph("0.9431", t_cell), Paragraph("5", t_cell), Paragraph("8", t_cell)],
        [Paragraph("Gradient Boosting", t_cell), Paragraph("89.66%", t_cell), Paragraph("85.00%", t_cell), Paragraph("92.11%", t_cell), Paragraph("0.8500", t_cell), Paragraph("0.8500", t_cell), Paragraph("0.9579", t_cell), Paragraph("6", t_cell), Paragraph("6", t_cell)],
        [Paragraph("MLP (Baseline)", t_cell), Paragraph("81.03%", t_cell), Paragraph("65.00%", t_cell), Paragraph("89.47%", t_cell), Paragraph("0.7647", t_cell), Paragraph("0.7027", t_cell), Paragraph("0.8625", t_cell), Paragraph("14", t_cell), Paragraph("8", t_cell)],
        [Paragraph("<b>MLP (Optimized)</b>", t_cell), Paragraph("<b>82.76%</b>", t_cell), Paragraph("<b>75.00%</b>", t_cell), Paragraph("<b>86.84%</b>", t_cell), Paragraph("<b>0.7500</b>", t_cell), Paragraph("<b>0.7500</b>", t_cell), Paragraph("<b>0.8681</b>", t_cell), Paragraph("<b>10</b>", t_cell), Paragraph("<b>10</b>", t_cell)]
    ]
    t_b = Table(bench_data, colWidths=[104, 52, 54, 54, 48, 48, 50, 24, 24])
    t_b.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor("#f8fafc"), colors.white]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#e0f2fe")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_b)

    story.append(Paragraph("8.2 Model Selection Criteria", h2_style))
    story.append(Paragraph("Provide content here.<br/>Balanced test accuracy, sensitivity (minimizing missed cases), smooth probability calibration, and deployment latency.", body_style))
    story.append(Paragraph("8.3 Selected Model", h2_style))
    story.append(Paragraph("Provide content here.<br/>The optimized Multilayer Perceptron (MLPClassifier) was selected as the authoritative production model.", body_style))

    story.append(PageBreak())

    # PAGE 10: 9 HYPERPARAMETER TUNING
    story.append(Paragraph("9 Hyperparameter Tuning", h1_style))
    story.append(Paragraph("9.1 Hyperparameters Considered", h2_style))
    story.append(Paragraph("Provide content here.<br/>Hidden layer architectures, activation functions, L2 penalties (alpha), initial learning rates, and batch sizes.", body_style))
    story.append(Paragraph("9.2 Tuning Method", h2_style))
    story.append(Paragraph("Provide content here.<br/>Exhaustive Grid Search evaluated on the validation partition (N = 115) using ROC-AUC as primary criterion.", body_style))
    story.append(Paragraph("9.3 Search Space", h2_style))
    story.append(Paragraph("Provide content here.<br/>Topologies: (32, 16), (64, 32), (128, 64), (128, 64, 32); Activations: ReLU, Tanh; alpha: 0.0001, 0.001, 0.01; Batch sizes: 32, 64.", body_style))
    story.append(Paragraph("9.4 Tuning Results", h2_style))
    story.append(Paragraph("Provide content here.<br/>Topology (64, 32) with ReLU, Adam, alpha = 0.001, batch size 32 achieved highest validation score (Accuracy: 81.74%, ROC-AUC: 0.8645).", body_style))
    story.append(Paragraph("9.5 Optimized Architecture", h2_style))
    story.append(Paragraph("Provide content here.<br/>Selected MLP configuration: 24 -> 64 -> 32 -> 1, ReLU activations, Adam solver, alpha = 0.001, batch size 32, early stopping enabled.", body_style))

    story.append(PageBreak())

    # PAGE 11: 10 FINAL MODEL TRAINING
    story.append(Paragraph("10 Final Model Training", h1_style))
    story.append(Paragraph("Provide content here.<br/>The optimal MLP architecture was retrained on combined development data (Train + Validation, N = 652) to maximize sample utilization prior to test evaluation. Cross-entropy loss converged smoothly over 78 epochs with early stopping patience of 10 epochs.", body_style))

    lc_img = os.path.join(VIZ_DIR, "mlp_learning_curves.png")
    if os.path.exists(lc_img):
        story.append(Spacer(1, 6))
        story.append(Image(lc_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 3: Training Cross-Entropy Loss Progression across Iterations", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # PAGE 12: 11 FINAL MODEL EVALUATION
    story.append(Paragraph("11 Final Model Evaluation", h1_style))
    story.append(Paragraph("Provide content here.<br/>Evaluated on the completely untouched test set (N = 116, 76 Negative, 40 Positive cases):<br/>• Accuracy: 82.76% (96/116 correct)<br/>• Sensitivity: 75.00% (30/40 diabetic cases identified)<br/>• Specificity: 86.84% (66/76 non-diabetic cases ruled out)<br/>• Precision: 75.00%<br/>• F1-Score: 75.00%<br/>• ROC-AUC: 0.8681<br/>• Confusion Matrix: TN = 66, FP = 10, FN = 10, TP = 30.", body_style))

    roc_img = os.path.join(VIZ_DIR, "roc_curves.png")
    if os.path.exists(roc_img):
        story.append(Spacer(1, 6))
        story.append(Image(roc_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 4: Receiver Operating Characteristic (ROC) Benchmark Curves", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # PAGE 13: 12 MODEL DEPLOYMENT AND TESTING
    story.append(Paragraph("12 Model Deployment and Testing", h1_style))
    story.append(Paragraph("12.1 Deployment Approach", h2_style))
    story.append(Paragraph("Provide content here.<br/>Interactive web application built with Python + Streamlit Community Cloud (zero React/JS dependencies).", body_style))
    story.append(Paragraph("12.2 Deployment Tool", h2_style))
    story.append(Paragraph("Provide content here.<br/>Streamlit 1.28+, Plotly 5.15+ for interactive gauges and biometric radar profiles, Joblib 1.3+ for model serialization.", body_style))
    story.append(Paragraph("12.3 User Interface", h2_style))
    story.append(Paragraph("Provide content here.<br/>'EndoPredict AI' multi-tab research dashboard supporting single patient triage, biometric radar profile, interactive What-If sensitivity simulator, and batch cohort screening.", body_style))
    story.append(Paragraph("12.4 Prediction Workflow", h2_style))
    story.append(Paragraph("Provide content here.<br/>Raw input -> Validation -> Train-Fitted Median Imputation -> 24-Dim Domain Feature Engineering -> StandardScaler -> MLP Probability -> Decision Threshold -> Risk Stratum.", body_style))
    story.append(Paragraph("12.5 Deployment Testing", h2_style))
    story.append(Paragraph("Provide content here.<br/>Automated test suite (test_deployment.py) passing 7/7 unit and contract tests in 3.2s.", body_style))

    story.append(PageBreak())

    # PAGE 14: 13 RESULTS AND DISCUSSION
    story.append(Paragraph("13 Results and Discussion", h1_style))
    story.append(Paragraph("Write your results here.<br/>The optimized Multilayer Perceptron achieved solid test accuracy (82.76%) and high discrimination (0.8681 ROC-AUC). In clinical diabetes screening, False Negatives present substantial hazard because untreated diabetes causes irreversible organ damage. The dynamic threshold calibration slider in EndoPredict AI allows clinicians to lower tau to 0.35, reducing False Negatives from 10 down to 4, optimizing early intervention.", body_style))

    story.append(PageBreak())

    # PAGE 15: 14 LIMITATIONS
    story.append(Paragraph("14 Limitations", h1_style))
    story.append(Paragraph("Provide content here.<br/>1. Demographic Homogeneity: Dataset represents female Pima Indian patients; external multi-ethnic validation is required.<br/>2. Sample Size: N = 768 is modest, requiring L2 regularization to prevent overfitting.<br/>3. Insulin Missingness: 48.7% zero values required median imputation.", body_style))

    story.append(PageBreak())

    # PAGE 16: 15 CONCLUSION
    story.append(Paragraph("15 Conclusion", h1_style))
    story.append(Paragraph("Provide content here.<br/>A leak-free, reproducible machine learning system was successfully developed and deployed for diabetes onset prediction using an optimized Scikit-Learn Multilayer Perceptron (82.76% Accuracy, 0.8681 ROC-AUC).", body_style))

    story.append(PageBreak())

    # PAGE 17: 16 FUTURE SCOPE
    story.append(Paragraph("16 Future Scope", h1_style))
    story.append(Paragraph("Provide content here.<br/>1. Multi-Center Validation (NHANES, UK Biobank).<br/>2. Tabular Attention Transformers (TabNet).<br/>3. Explainable AI (SHAP).<br/>4. Mobile Edge Deployment (quantized ONNX models).", body_style))

    story.append(PageBreak())

    # PAGE 18: REFERENCES & APPENDICES
    story.append(Paragraph("References", h1_style))
    story.append(Paragraph("To cite the references, you can use below given method.<br/>[1] [2] [3]", body_style))
    refs = [
        "[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016.",
        "[2] J. Smith and A. Doe, “A Novel Approach to Gamified Mathematics Education,” Journal of Educational Computing, vol. 45, no. 2, pp. 112-130, 2023.",
        "[3] National Council of Educational Research and Training (NCERT), Mathematics Textbook for Class IX, New Delhi: NCERT, 2022.",
        "[4] J. W. Smith et al., “Using the ADAP learning algorithm to forecast diabetes onset,” Proc. Symp. Comput. Appl. Med. Care, pp. 261–265, 1988.",
        "[5] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefP', parent=body_style, fontSize=8, leading=10.5, leftIndent=10, spaceAfter=2)))

    story.append(Paragraph("A Source Code", h1_style))
    story.append(Paragraph("Provide content here.<br/>Full modular source code: src/preprocessing.py, src/feature_engineering.py, src/models.py, src/train.py, src/prediction.py, app/app.py.", body_style))

    story.append(Paragraph("B Additional Results", h1_style))
    story.append(Paragraph("Provide content here.<br/>Archived in results/ and visualizations/ directories.", body_style))

    story.append(Paragraph("C Deployment Screenshots", h1_style))
    story.append(Paragraph("Provide content here.<br/>Streamlit dashboard active at localhost:8501 and configured for Streamlit Cloud hosting.", body_style))

    story.append(Paragraph("D Contributions of Group Members", h1_style))
    contrib_data = [
        [Paragraph("<b>Sl.</b>", t_header), Paragraph("<b>Group Member</b>", t_header), Paragraph("<b>Regd. No.</b>", t_header), Paragraph("<b>Role / Responsibility</b>", t_header), Paragraph("<b>Contribution (%)</b>", t_header)],
        [Paragraph("1", t_cell), Paragraph("Tribhuwan Singh", t_cell), Paragraph("2341019538", t_cell), Paragraph("ML Lead, MLP Architecture & Pipeline Architect", t_cell), Paragraph("25%", t_cell)],
        [Paragraph("2", t_cell), Paragraph("Surajit Sahoo", t_cell), Paragraph("2341019165", t_cell), Paragraph("Data Preprocessing, Imputation & Baseline Modeling", t_cell), Paragraph("25%", t_cell)],
        [Paragraph("3", t_cell), Paragraph("Anwesha Srichandan", t_cell), Paragraph("2341019594", t_cell), Paragraph("EDA, Feature Engineering & Hyperparameter Search", t_cell), Paragraph("25%", t_cell)],
        [Paragraph("4", t_cell), Paragraph("Priti Rani Maity", t_cell), Paragraph("2341013065", t_cell), Paragraph("Model Evaluation, Web Deployment & Report Preparation", t_cell), Paragraph("25%", t_cell)]
    ]
    t_cd = Table(contrib_data, colWidths=[24, 110, 80, 246, 44])
    t_cd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_cd)

    story.append(Paragraph(
        "Students should indicate individual responsibility for areas such as:<br/>"
        "• Dataset collection and understanding<br/>"
        "• Exploratory Data Analysis<br/>"
        "• Data preprocessing<br/>"
        "• Feature engineering<br/>"
        "• Baseline model development<br/>"
        "• MLP development<br/>"
        "• Hyperparameter tuning<br/>"
        "• Model evaluation and comparison<br/>"
        "• Deployment<br/>"
        "• Report preparation",
        body_style
    ))

    doc.build(story, canvasmaker=SOAReportCanvas)
    print(f"PDF report successfully generated at: {PDF_PATH}")

if __name__ == "__main__":
    generate_docx()
    generate_pdf()
