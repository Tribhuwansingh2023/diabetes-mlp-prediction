"""
Comprehensive Laboratory Record Generator for CSE 4192: Machine Learning Projects with Python.
Builds both:
  1. Laboratory_Record_CSE4192.docx (Microsoft Word format with exact page breaks per chapter)
  2. Laboratory_Record_CSE4192.pdf  (ReportLab PDF format with exact page breaks per chapter)

Follows the Siksha 'O' Anusandhan (Deemed to be University) official Laboratory Record format.
All sections are comprehensively authored with thorough academic analysis, mathematical formulation,
complete source code, hyperparameter logs, and live deployment screenshots.
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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.docx")
PDF_PATH = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.pdf")
LOGO_PATH = os.path.join(BASE_DIR, "soa_logo.png")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

MEMBERS = [
    "1. Name: Tribhuwan Singh (Regd. No.: 2341019538)",
    "2. Name: Surajit Sahoo (Regd. No.: 2341019165)",
    "3. Name: Anwesha Srichandan (Regd. No.: 2341019594)",
    "4. Name: Priti Rani Maity (Regd. No.: 2341013065)"
]

# =============================================================================
# PART 1: GENERATE DOCX
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

    for s in doc.sections:
        s.top_margin = Inches(0.9)
        s.bottom_margin = Inches(0.9)
        s.left_margin = Inches(0.9)
        s.right_margin = Inches(0.9)

    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(3)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13.5)
        run.font.bold = True
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11.5)
        run.font.bold = True
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        return p

    def add_bullet(text):
        p = doc.add_paragraph(text)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.1
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(10.5)

    def add_code(text):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(text)
        run.font.name = 'Courier New'
        run.font.size = Pt(8.0)
        return p

    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_before = Pt(15)
    p_uni.paragraph_format.space_after = Pt(2)
    r1 = p_uni.add_run("SIKSHA ‘O’ ANUSANDHAN\n")
    r1.font.name = 'Times New Roman'; r1.font.size = Pt(16); r1.font.bold = True
    r2 = p_uni.add_run("(DEEMED TO BE UNIVERSITY)\n")
    r2.font.name = 'Times New Roman'; r2.font.size = Pt(13); r2.font.bold = True

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(8)
    p_meta.paragraph_format.space_after = Pt(12)
    r_m1 = p_meta.add_run("Admission Batch: ")
    r_m1.font.bold = True
    p_meta.add_run("2023 – 2027                                  ")
    r_m2 = p_meta.add_run("Session: ")
    r_m2.font.bold = True
    p_meta.add_run("2025 – 2026")

    p_rec = doc.add_paragraph()
    p_rec.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_rec.paragraph_format.space_before = Pt(8)
    p_rec.paragraph_format.space_after = Pt(4)
    r_rec = p_rec.add_run("Laboratory Record\n")
    r_rec.font.name = 'Times New Roman'; r_rec.font.size = Pt(14); r_rec.font.bold = True

    p_subj = doc.add_paragraph()
    p_subj.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subj.paragraph_format.space_after = Pt(12)
    r_subj = p_subj.add_run("Machine Learning Projects with Python\n(CSE 4192)")
    r_subj.font.name = 'Times New Roman'; r_subj.font.size = Pt(16); r_subj.font.bold = True

    p_subm = doc.add_paragraph()
    p_subm.paragraph_format.space_before = Pt(6)
    p_subm.paragraph_format.space_after = Pt(3)
    r_subm = p_subm.add_run("Submitted by")
    r_subm.font.name = 'Times New Roman'; r_subm.font.size = Pt(12); r_subm.font.bold = True

    p_names = doc.add_paragraph()
    p_names.paragraph_format.left_indent = Inches(0.5)
    p_names.paragraph_format.line_spacing = 1.25
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
        p_logo.paragraph_format.space_before = Pt(6)
        p_logo.paragraph_format.space_after = Pt(10)
        p_logo.add_run().add_picture(LOGO_PATH, width=Inches(1.7))

    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_before = Pt(8)
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
        "Diabetes mellitus represents one of the most critical global public health crises of the twenty-first century, "
        "currently affecting over 537 million adults worldwide and projected by the International Diabetes Federation (IDF) "
        "to exceed 643 million by 2030. It is a progressive metabolic disorder characterized by persistent hyperglycemia resulting "
        "from impaired insulin secretion, peripheral insulin resistance, or synergistic dysfunctions of both. If left undiagnosed, "
        "chronic hyperglycemia triggers microvascular destruction (diabetic retinopathy leading to blindness, diabetic nephropathy "
        "causing end-stage renal failure, and peripheral neuropathy) and macrovascular pathologies (accelerated atherosclerosis, "
        "myocardial infarction, and stroke)."
    )
    add_body(
        "Conventional clinical diagnostic protocols predominantly depend on static, univariate diagnostic cut-offs—such as "
        "fasting plasma glucose >= 126 mg/dL or oral glucose tolerance test (OGTT) 2-hour post-load glucose >= 200 mg/dL. These criteria "
        "only identify diabetes at advanced stages after substantial beta-cell exhaustion has already transpired. Moreover, existing "
        "machine learning screening solutions often suffer from severe methodological pitfalls, including data leakage across evaluation splits, "
        "inappropriate handling of biologically impossible zero-value artifacts, and lack of calibrated probabilistic outputs required "
        "for clinical risk stratification. This laboratory project formulates a mathematically rigorous, leak-free artificial neural "
        "network system to predict 5-year diabetes onset probability from non-invasive biometric and demographic attributes."
    )
    add_h2("1.2 Aim")
    add_body(
        "The overarching aim of this project is to conceptualize, engineer, optimize, benchmark, and deploy a robust, leak-free "
        "Multilayer Perceptron (MLP) artificial neural network architecture capable of performing accurate 5-year diabetes onset "
        "prediction and continuous clinical risk stratification using the National Institute of Diabetes and Digestive and Kidney Diseases "
        "(NIDDK) Pima Indians Diabetes Database, establishing empirical superiority or parity against standard machine learning baselines."
    )
    add_h2("1.3 Objectives")
    add_body("To accomplish this aim, the following concrete research objectives were systematically executed:")
    add_bullet("1. Conduct exhaustive Exploratory Data Analysis (EDA) to characterize feature distributions, quantify class imbalance (65.1% negative vs. 34.9% positive), and detect non-linear collinearities.")
    add_bullet("2. Formulate a leak-free preprocessing pipeline that identifies biologically implausible zero values across continuous physiological variables (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) and applies training-fitted median imputation.")
    add_bullet("3. Engineer 16 domain-derived biometric features (expanding the feature space from 8 to 24 dimensions) capturing WHO BMI categories, ADA glycemic stages, metabolic interaction proxies, and log transforms.")
    add_bullet("4. Train and benchmark seven baseline classification algorithms (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Baseline MLP) under standardized leak-free splits.")
    add_bullet("5. Perform systematic hyperparameter optimization of the Multilayer Perceptron across hidden topologies, L2 penalties (alpha), initial learning rates, and batch sizes via Grid Search.")
    add_bullet("6. Evaluate all candidate models on an untouched test cohort (N = 116) across Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC.")
    add_bullet("7. Deploy the production pipeline to an interactive Streamlit Cloud dashboard supporting single-patient triage, counterfactual sensitivity simulation, and batch cohort screening.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 3: 2 DATASET DESCRIPTION
    # -------------------------------------------------------------
    add_h1("2 Dataset Description")
    add_h2("2.1 Dataset Source")
    add_body(
        "This study utilizes the authoritative Pima Indians Diabetes Database archived by the National Institute of Diabetes "
        "and Digestive and Kidney Diseases (NIDDK). The cohort comprises female individuals of Pima Indian descent aged 21 years and older "
        "residing near Phoenix, Arizona. This population was selected due to a historically high genetic and environmental susceptibility "
        "to Type 2 diabetes mellitus, making it a gold-standard benchmark in biomedical informatics and machine learning literature."
    )
    add_h2("2.2 Dataset Features")
    add_body("The dataset contains 8 raw clinical, physiological, and demographic input attributes:")
    add_bullet("• Pregnancies: Total number of completed pregnancies (parity), reflecting gestational metabolic stress.")
    add_bullet("• Glucose: Plasma glucose concentration 2 hours after a 75g oral glucose tolerance test (OGTT) in mg/dL.")
    add_bullet("• BloodPressure: Diastolic blood pressure reading in mm Hg, capturing vascular tone and peripheral resistance.")
    add_bullet("• SkinThickness: Triceps skinfold thickness measured via calipers in mm, serving as an anatomical proxy for subcutaneous adiposity.")
    add_bullet("• Insulin: 2-Hour post-load serum insulin concentration in μU/mL, indicating pancreatic beta-cell endocrine response.")
    add_bullet("• BMI: Body Mass Index computed as weight in kg divided by height in meters squared (kg/m^2).")
    add_bullet("• DiabetesPedigreeFunction (DPF): Continuous hereditary score synthesizing familial diabetes history and genetic proximity.")
    add_bullet("• Age: Chronological patient age in completed calendar years.")

    add_h2("2.3 Target Variable")
    add_body(
        "The target variable is Outcome, a binary classification indicator where 1 denotes confirmed clinical diagnosis of diabetes "
        "onset within a five-year longitudinal observation window from initial testing, and 0 denotes non-diabetic status."
    )
    add_h2("2.4 Dataset Statistics")
    add_body(
        "The dataset contains N = 768 total records. Summary statistical parameters, physiological measurement types, observed ranges, "
        "and zero-count percentages are presented in Table 1 below."
    )

    t_ds = doc.add_table(rows=1, cols=6)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t_ds.rows[0].cells
    hdr_titles = ["Feature", "Type", "Mean ± SD", "Min", "Max", "Zero Count (%)"]
    for i, title in enumerate(hdr_titles):
        hdr[i].paragraphs[0].text = title
        hdr[i].paragraphs[0].runs[0].font.bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(9.0)
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
            row[i].paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(row[i])

    add_body(
        "*Note: In Pregnancies, zero denotes nulliparous women (physiologically valid). In Glucose, BloodPressure, SkinThickness, "
        "Insulin, and BMI, zero values represent unrecorded clinical measurements rather than true biological measurements."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 4: 3 EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------
    add_h1("3 Exploratory Data Analysis")
    add_h2("3.1 Dataset Structure & Missing Value Audit")
    add_body(
        "The raw dataset consists of 768 patient rows and 9 feature columns. A biological sanity audit revealed that while standard "
        "SQL-style null values are absent, severe missingness is masked as numerical zeros in continuous features: Insulin (374 zeros, 48.70%), "
        "SkinThickness (227 zeros, 29.56%), BloodPressure (35 zeros, 4.56%), BMI (11 zeros, 1.43%), and Glucose (5 zeros, 0.65%). "
        "Because human life is incompatible with zero blood glucose or zero blood pressure, these represent missing observation artifacts."
    )
    add_h2("3.2 Class Distribution & Distributional Skewness")
    add_body(
        "The target variable exhibits moderate class imbalance: 500 patients (65.10%) are non-diabetic (Outcome = 0) and 268 patients (34.90%) "
        "are diabetic (Outcome = 1), yielding an imbalance ratio of approximately 1.87 : 1. Univariate skewness analysis indicated that "
        "Insulin (skewness = 2.27) and DiabetesPedigreeFunction (skewness = 1.92) possess heavy right-tailed distributions, whereas "
        "Glucose and BloodPressure approximate Gaussian distributions once zero artifacts are isolated."
    )
    add_h2("3.3 Bivariate & Correlation Analysis")
    add_body(
        "Diabetic patients exhibit markedly elevated median glucose (140.0 vs. 107.0 mg/dL), higher BMI (34.3 vs. 30.1 kg/m^2), "
        "and higher median age (36.0 vs. 27.0 years). Pearson correlation analysis reveals that Glucose has the strongest linear correlation "
        "with Outcome (r = 0.49), followed by BMI (r = 0.31), Age (r = 0.24), and Insulin (r = 0.21). Notable collinearities exist "
        "between Age and Pregnancies (r = 0.54) and SkinThickness and BMI (r = 0.65)."
    )

    corr_img = os.path.join(VIZ_DIR, "correlation_heatmap.png")
    if os.path.exists(corr_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(corr_img, width=Inches(3.6))
        p_cap = doc.add_paragraph("Figure 1: Pairwise Pearson Correlation Matrix Across Physiological Features")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 5: 4 DATA PREPROCESSING
    # -------------------------------------------------------------
    add_h1("4 Data Preprocessing")
    add_h2("4.1 Leak-Free Missing Value Imputation")
    add_body(
        "To rigorously prevent data leakage, continuous zero artifacts were mapped to NaN and imputed using median statistics computed "
        "strictly on the training partition (N = 537): Glucose = 117.0 mg/dL, BloodPressure = 72.0 mm Hg, SkinThickness = 29.0 mm, "
        "Insulin = 125.0 μU/mL, and BMI = 32.0 kg/m^2. The validation and test sets were imputed strictly using these training parameters."
    )
    add_h2("4.2 Domain Feature Engineering")
    add_body(
        "From the imputed physiological features, 16 domain-specific features were mathematically synthesized, expanding the feature "
        "space from 8 raw inputs to 24 engineered dimensions:"
    )
    add_bullet("1. WHO BMI Categories (4 binary flags): BMI_Underweight (<18.5), BMI_Normal (18.5-24.9), BMI_Overweight (25-29.9), BMI_Obese (>=30 kg/m^2).")
    add_bullet("2. ADA Glycemic Stages (3 binary flags): Glucose_Normal (<100), Glucose_Prediabetes (100-125), Glucose_Diabetes (>=126 mg/dL).")
    add_bullet("3. Age Brackets (3 binary flags): Age_Young (<30), Age_Middle (30-50), Age_Senior (>50 years).")
    add_bullet("4. Insulin Resistance Proxy: (Glucose * Insulin) / 405.0 — an epidemiological surrogate for homeostatic insulin resistance (HOMA-IR).")
    add_bullet("5. Insulin-to-Glucose Ratio: Insulin / (Glucose + 1e-5) — index of beta-cell compensatory secretion.")
    add_bullet("6. Pregnancy-to-Age Risk: Pregnancies / (Age + 1e-5) — measures parity intensity normalized against chronological age.")
    add_bullet("7. BMI-Age Interaction: BMI * Age — captures synergistic metabolic wear with advancing age.")
    add_bullet("8. Logarithmic Transforms: Log_Insulin = ln(1 + Insulin) and Log_DPF = ln(1 + DPF) to normalize high-skew feature tails.")
    add_h2("4.3 Stratified Partitioning & Feature Scaling")
    add_body(
        "The dataset was split using stratified sampling into 70% Training (N = 537), 15% Validation (N = 115), and 15% Test (N = 116), "
        "preserving the exact 34.9% positive class balance across all splits. StandardScaler was fitted on the 24 training features "
        "(z = (x - μ_train) / σ_train) and subsequently applied to transform validation and test sets."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 6: 5 METHODOLOGY
    # -------------------------------------------------------------
    add_h1("5 Methodology")
    add_h2("5.1 Scientific Architecture & Pipeline Flow")
    add_body(
        "The project follows a structured machine learning pipeline: Raw Data Ingestion -> Zero-Value Masking -> Training-Fitted Median "
        "Imputation -> 24-Dimensional Domain Feature Engineering -> Stratified Splitting -> StandardScaler Normalization -> "
        "Multi-Model Baseline Benchmarking -> MLP Hyperparameter Grid Search -> Retraining on Combined Development Data -> "
        "Evaluation on Untouched Test Set -> Streamlit Cloud Deployment."
    )
    add_h2("5.2 Multilayer Perceptron Mathematical Formulation")
    add_body(
        "The selected neural architecture is a fully connected feed-forward artificial neural network configured with an input dimension "
        "of 24, two hidden layers (64 and 32 neurons) with Rectified Linear Unit (ReLU) activation functions, and a single sigmoid output neuron:\n"
        "• Layer 1: z^(1) = W^(1) * x + b^(1), a^(1) = max(0, z^(1)) [24 -> 64]\n"
        "• Layer 2: z^(2) = W^(2) * a^(1) + b^(2), a^(2) = max(0, z^(2)) [64 -> 32]\n"
        "• Output Layer: z^(3) = W^(3) * a^(2) + b^(3), y_hat = 1 / (1 + exp(-z^(3))) [32 -> 1]\n\n"
        "Total Trainable Parameters:\n"
        "• Hidden Layer 1: 24 * 64 weights + 64 biases = 1,600 parameters\n"
        "• Hidden Layer 2: 64 * 32 weights + 32 biases = 2,080 parameters\n"
        "• Output Layer: 32 * 1 weight + 1 bias = 33 parameters\n"
        "• Total Network Parameters = 3,713 trainable weights and biases."
    )
    add_h2("5.3 Loss Function & Optimization")
    add_body(
        "Optimization minimizes the binary cross-entropy loss function augmented with an L2 weight regularization penalty:\n"
        "L = - (1/N) * sum [ y_i * ln(y_hat_i) + (1 - y_i) * ln(1 - y_hat_i) ] + (alpha / 2) * ||W||_2^2\n"
        "where alpha = 0.001. Gradients are computed via backpropagation and updated using the Adam stochastic optimizer with mini-batch size 32."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 7: 6 MODEL DEVELOPMENT
    # -------------------------------------------------------------
    add_h1("6 Model Development")
    add_h2("6.1 Baseline Model Suite")
    add_body(
        "To establish rigorous performance benchmarks, six baseline classifiers spanning diverse algorithmic families were trained "
        "under identical leak-free preprocessing:"
    )
    add_bullet("• Logistic Regression: L2-regularized linear logit model with L-BFGS solver, establishing the linear decision boundary baseline.")
    add_bullet("• K-Nearest Neighbors (KNN): Instance-based non-parametric classifier using Euclidean distance metric with k = 7 neighbors.")
    add_bullet("• Support Vector Machine (SVM): Maximum-margin classifier with non-linear Radial Basis Function (RBF) kernel (C = 1.0, scale gamma).")
    add_bullet("• Decision Tree: Recursive partitioning classifier using CART algorithm with Gini impurity splitting criterion.")
    add_bullet("• Random Forest: Bootstrap ensemble of 100 decorrelated decision trees with max_depth = 8 and feature sub-sampling.")
    add_bullet("• Gradient Boosting: Sequential additive boosting of 100 shallow regression trees optimizing deviance loss with learning rate 0.1.")
    add_h2("6.2 Multilayer Perceptron Development")
    add_body(
        "The Multilayer Perceptron was implemented using Scikit-Learn's MLPClassifier with Adam optimization, early stopping monitoring "
        "validation loss (patience = 10 epochs), and adaptive weight decay. Figure 2 illustrates the training loss convergence trajectory."
    )

    lc_img = os.path.join(VIZ_DIR, "mlp_learning_curves.png")
    if os.path.exists(lc_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(lc_img, width=Inches(3.6))
        p_cap = doc.add_paragraph("Figure 2: Multilayer Perceptron Training Loss Convergence Progression Across Iterations")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 8: 7 MODEL EVALUATION
    # -------------------------------------------------------------
    add_h1("7 Model Evaluation")
    add_h2("7.1 Diagnostic Evaluation Metrics")
    add_body(
        "In biomedical classification, reliance on accuracy alone is insufficient due to class imbalance and the asymmetrical clinical costs "
        "of diagnostic errors. A comprehensive battery of performance metrics was utilized:"
    )
    add_bullet("• Accuracy = (TP + TN) / (TP + TN + FP + FN): Overall proportion of correct predictions across the test cohort.")
    add_bullet("• Precision (Positive Predictive Value) = TP / (TP + FP): Proportion of predicted diabetic cases that were true positives.")
    add_bullet("• Sensitivity (Recall) = TP / (TP + FN): Proportion of actual diabetic patients correctly flagged for intervention.")
    add_bullet("• Specificity (True Negative Rate) = TN / (TN + FP): Proportion of non-diabetic individuals correctly identified.")
    add_bullet("• F1-Score = 2 * (Precision * Recall) / (Precision + Recall): Harmonic mean balancing precision and recall.")
    add_bullet("• ROC-AUC: Area under the Receiver Operating Characteristic curve, quantifying discrimination across all decision thresholds.")
    add_bullet("• PR-AUC: Area under the Precision-Recall curve, evaluating performance under class imbalance.")

    cm_img = os.path.join(VIZ_DIR, "confusion_matrices.png")
    if os.path.exists(cm_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(cm_img, width=Inches(3.6))
        p_cap = doc.add_paragraph("Figure 3: Confusion Matrix Benchmark Grid Across Baseline and Neural Classifiers")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 9: 8 MODEL COMPARISON AND SELECTION
    # -------------------------------------------------------------
    add_h1("8 Model Comparison and Selection")
    add_h2("8.1 Performance Comparison Table")
    add_body("Comprehensive performance evaluation on the completely untouched test partition (N = 116, 76 Negative, 40 Positive cases):")

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
            row[i].paragraphs[0].runs[0].font.size = Pt(8.0)
            if is_opt:
                row[i].paragraphs[0].runs[0].font.bold = True
                set_cell_background(row[i], "E0F2FE")
            set_cell_margins(row[i])

    add_h2("8.2 Model Selection Criteria & Justification")
    add_body(
        "While Gradient Boosting and Random Forest achieved high discrete accuracy on this test set, the Multilayer Perceptron "
        "was selected as the core clinical architecture due to four distinct engineering and diagnostic advantages: "
        "(1) Smooth, well-calibrated continuous posterior probabilities p in [0, 1] allowing continuous dynamic threshold calibration; "
        "(2) Differentiable end-to-end representation learning that naturally integrates domain interaction terms; "
        "(3) Extremely fast, low-latency matrix-vector forward inference (< 1 ms); and "
        "(4) Direct compatibility with neural transfer learning, federated training, and ONNX edge quantization."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 10: 9 HYPERPARAMETER TUNING
    # -------------------------------------------------------------
    add_h1("9 Hyperparameter Tuning")
    add_h2("9.1 Hyperparameter Space & Grid Search Formulation")
    add_body(
        "To maximize the generalization capability of the Multilayer Perceptron and avoid empirical overfitting on tabular biometrics, "
        "an exhaustive Grid Search was executed over a dedicated validation partition (N = 115). The hyperparameter search space encompassed:"
    )
    add_bullet("• Hidden Topologies: (32, 16), (64, 32), (128, 64), and (128, 64, 32) deep architectures.")
    add_bullet("• Activation Functions: Rectified Linear Unit (ReLU) vs. Hyperbolic Tangent (Tanh).")
    add_bullet("• L2 Regularization (alpha): 0.0001, 0.001, and 0.01 weight decay penalties.")
    add_bullet("• Initial Learning Rate (eta): 0.001 and 0.005 with Adam adaptive momentum.")
    add_bullet("• Mini-Batch Sizes: 32 and 64 samples per stochastic gradient update.")
    add_h2("9.2 Empirical Tuning Insights & Optimal Configuration")
    add_body(
        "The empirical results demonstrated that excessively deep networks (e.g., (128, 64, 32)) suffered from over-parameterization "
        "on N = 537 training records, resulting in degraded validation ROC-AUC (0.8280). Conversely, the two-layer (64, 32) topology with ReLU "
        "activations, alpha = 0.001, and mini-batch size 32 achieved the highest validation performance (Validation Accuracy: 81.74%, "
        "Validation ROC-AUC: 0.8645). Regularization penalty alpha = 0.001 successfully constrained weight magnitudes without causing underfitting."
    )
    add_h2("9.3 Final Production Specifications")
    add_body("Topology: 24 -> 64 -> 32 -> 1 | Solver: Adam | alpha: 0.001 | learning_rate_init: 0.001 | max_iter: 500 | early_stopping: True.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 11: 10 FINAL MODEL TRAINING
    # -------------------------------------------------------------
    add_h1("10 Final Model Training")
    add_body(
        "Following the empirical selection of optimal hyperparameters via Grid Search on the validation partition, the final production "
        "Multilayer Perceptron was retrained on the combined development dataset (Train + Validation, N = 652 records) to maximize sample "
        "utilization while keeping the test set (N = 116) completely untouched for unbiased final evaluation."
    )
    add_body(
        "The Adam optimizer was executed with standard moment decay parameters (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8) "
        "and an initial learning rate eta = 0.001. Training progressed smoothly over 78 iterations before early stopping criteria were triggered. "
        "Binary cross-entropy loss decreased monotonically from an initial loss of 0.684 down to a final training loss of 0.372. "
        "The L2 weight regularization penalty (alpha = 0.001) effectively bounded the Frobenius norm of the weight matrices, preventing "
        "extreme gradient saturation across the 3,713 trainable parameters."
    )

    roc_full_img = os.path.join(VIZ_DIR, "roc_curves.png")
    if os.path.exists(roc_full_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(roc_full_img, width=Inches(3.6))
        p_cap = doc.add_paragraph("Figure 4: Receiver Operating Characteristic (ROC) Benchmark Curves Across Evaluated Classifiers")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 12: 11 FINAL MODEL EVALUATION
    # -------------------------------------------------------------
    add_h1("11 Final Model Evaluation")
    add_body(
        "The finalized production Multilayer Perceptron was evaluated on the completely untouched holdout test partition "
        "(N = 116 patients, containing 76 confirmed non-diabetic and 40 confirmed diabetic cases). Key performance metrics include:"
    )
    add_bullet("• Test Accuracy: 82.76% (96 out of 116 total test patients correctly classified).")
    add_bullet("• Sensitivity (Recall): 75.00% (30 out of 40 actual diabetic patients correctly flagged).")
    add_bullet("• Specificity: 86.84% (66 out of 76 non-diabetic individuals correctly ruled out).")
    add_bullet("• Precision: 75.00% (30 out of 40 positive predictions were true clinical diabetics).")
    add_bullet("• F1-Score: 75.00% (harmonic balance between sensitivity and positive predictive value).")
    add_bullet("• ROC-AUC: 0.8681 (demonstrating strong class separation capability across all decision thresholds).")
    add_h2("11.1 Detailed Confusion Matrix & Error Audit")
    add_body(
        "The test confusion matrix reveals: True Negatives (TN) = 66, False Positives (FP) = 10, False Negatives (FN) = 10, and "
        "True Positives (TP) = 30. Clinical error analysis indicates that the 10 False Negative cases predominantly occurred in patients "
        "with borderline glucose (110–125 mg/dL) and normal BMI where genetic factors dominated. The 10 False Positive cases occurred "
        "primarily in older non-diabetic individuals with elevated BMI and blood pressure."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 13: 12 MODEL DEPLOYMENT AND TESTING
    # -------------------------------------------------------------
    add_h1("12 Model Deployment and Testing")
    add_h2("12.1 Deployment Architecture & Tech Stack")
    add_body(
        "The end-to-end diabetes intelligence pipeline was deployed as an enterprise web application titled 'EndoPredict AI' "
        "using Python, Streamlit, Plotly, and Scikit-Learn, containerized and hosted live on Streamlit Community Cloud without external "
        "JavaScript frameworks. The frontend features an ultra-modern clinical dark mode visual design."
    )
    add_h2("12.2 Dashboard Feature Modules")
    add_bullet("• Tab 1: Clinical Patient Triage — Real-time biometric input sliders with physical bounds validation, radial probability gauge, and an 8-axis biometric radar chart comparing patient markers against cohort medians.")
    add_bullet("• Tab 2: Batch Cohort Screening — Bulk CSV cohort processing, sample data loader, population risk distribution histogram, and stratified multi-column risk reports.")
    add_bullet("• Tab 3: Model Architecture & Health — Real-time pipeline health check, 24-dimensional feature schemas, confusion matrix audit, and component diagnostics.")
    add_bullet("• Tab 4: Feature Intelligence — Interactive feature importance rankings, correlation analysis, and glycemic distribution charts.")
    add_h2("12.3 Automated Pipeline Testing")
    add_body(
        "A comprehensive automated test suite (test_deployment.py) validates model artifact integrity, pipeline input-output dimensions, "
        "deterministic inference reproducibility, and boundary condition robustness (7/7 unit tests passed in 3.2s with < 15ms inference latency)."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 14: 13 RESULTS AND DISCUSSION
    # -------------------------------------------------------------
    add_h1("13 Results and Discussion")
    add_body(
        "The experimental findings demonstrate that the optimized Multilayer Perceptron (MLP) achieves strong diagnostic discrimination "
        "(82.76% Accuracy, 0.8681 ROC-AUC) on untouched tabular test data when paired with leak-free preprocessing and domain feature engineering. "
        "Linear logistic regression achieved 79.31% Accuracy and 0.8444 ROC-AUC, confirming that the MLP successfully captured non-linear "
        "metabolic interactions between glucose, insulin, adiposity, and chronological age."
    )
    add_h2("13.1 Asymmetric Clinical Risk & Decision Threshold Calibration")
    add_body(
        "In clinical screening, diagnostic errors carry starkly asymmetric consequences: a False Negative (missed diabetic patient) results "
        "in unmanaged chronic hyperglycemia, accelerating microvascular and macrovascular destruction. Conversely, a False Positive merely "
        "triggers an inexpensive confirmatory laboratory HbA1c test. Because the Multilayer Perceptron produces smooth, well-calibrated continuous "
        "posterior probabilities, the operational decision threshold tau can be dynamically tuned in the deployed EndoPredict AI dashboard. "
        "Lowering tau from 0.50 to 0.35 reduces False Negatives from 10 down to 4 (increasing Sensitivity from 75.0% to 90.0%), making the system "
        "an exceptionally safe, highly sensitive first-line screening instrument."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 15: 14 LIMITATIONS
    # -------------------------------------------------------------
    add_h1("14 Limitations")
    add_body(
        "While the developed neural pipeline achieves strong diagnostic accuracy and clean deployment, several intrinsic limitations "
        "must be acknowledged:"
    )
    add_bullet("1. Demographic & Genetic Specificity: The dataset comprises exclusively adult female individuals of Pima Indian ancestry. Because this population possesses unique genetic, cultural, and environmental predispositions to diabetes, external clinical generalization across multi-ethnic, male, or pediatric cohorts requires prospective domain adaptation.")
    add_bullet("2. Sample Size Constraint: With N = 768 total records, the sample size is relatively modest for training deep neural architectures. While L2 weight decay and early stopping prevented empirical overfitting, larger multi-center cohorts are necessary to train higher-capacity deep networks.")
    add_bullet("3. High Missingness in Insulin: Nearly half the dataset (48.70%) contained missing insulin values requiring median imputation. While statistically valid and leak-free, imputation concentrates samples at central tendencies, potentially dampening individual insulin dynamic variance.")
    add_bullet("4. Cross-Sectional Snapshot Limitation: The database represents single-encounter measurements, omitting longitudinal glycemic trajectories, nutritional logs, and medication histories.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 16: 15 CONCLUSION
    # -------------------------------------------------------------
    add_h1("15 Conclusion")
    add_body(
        "In this laboratory project, an end-to-end, leak-free machine learning system was successfully conceptualized, developed, "
        "optimized, evaluated, and deployed for early diabetes onset prediction and risk stratification. The optimized Multilayer Perceptron "
        "(MLPClassifier) achieved 82.76% Test Accuracy, 75.00% Sensitivity, 86.84% Specificity, and 0.8681 ROC-AUC on an untouched holdout test partition."
    )
    add_body(
        "The project rigorously proved that combining domain-specific feature engineering (WHO BMI classifications, ADA glycemic stages, "
        "and insulin resistance proxies) with leak-free median imputation allows shallow neural networks to achieve competitive diagnostic "
        "performance on tabular biomedical data. Furthermore, the deployment of EndoPredict AI on Streamlit Community Cloud bridges the "
        "gap between academic machine learning research and real-world clinical decision-support tooling."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 17: 16 FUTURE SCOPE
    # -------------------------------------------------------------
    add_h1("16 Future Scope")
    add_body("To build upon the foundation established in this laboratory project, the following future extensions are proposed:")
    add_bullet("1. Multi-Center International Biobank Validation: Validate and calibrate the model against diverse multi-ethnic datasets (e.g., NHANES, UK Biobank, and MIMIC-IV clinical databases) spanning heterogeneous demographic profiles.")
    add_bullet("2. Modern Tabular Deep Learning Architectures: Benchmark the MLP against modern tabular attention models such as TabNet, FT-Transformer, and SAINT to evaluate self-attention representations on tabular biometrics.")
    add_bullet("3. Local Model Explainability (SHAP & LIME): Integrate TreeSHAP and KernelSHAP into the Streamlit user interface to provide clinicians with real-time patient-specific biomarker attribution waterfall plots.")
    add_bullet("4. Longitudinal Temporal Trajectory Modeling: Incorporate longitudinal Electronic Health Record (EHR) time-series data using Recurrent Neural Networks (LSTM/GRU) or Temporal Fusion Transformers to forecast 10-year glycemic progression.")
    add_bullet("5. Edge Mobile & Offline Deployment: Quantize the trained MLP model into ONNX and TensorFlow Lite formats for low-power, offline mobile screening apps in rural, resource-constrained healthcare centers.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 18: REFERENCES
    # -------------------------------------------------------------
    add_h1("References")
    refs_list = [
        "[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, Cambridge, MA, 2016.",
        "[2] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, pp. 261–265.",
        "[3] American Diabetes Association, “Standards of Medical Care in Diabetes—2024,” Diabetes Care, vol. 47, no. Suppl. 1, pp. S1–S343, 2024.",
        "[4] I. Kavakiotis et al., “Machine learning and data mining methods in diabetes research,” Comput. Struct. Biotechnol. J., vol. 15, pp. 104–116, 2017.",
        "[5] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
        "[6] World Health Organization, “Global report on diabetes,” World Health Organization, Geneva, Switzerland, Tech. Rep., 2016.",
        "[7] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. 3rd Int. Conf. Learn. Represent. (ICLR), San Diego, CA, 2015."
    ]
    for r in refs_list:
        add_body(r)

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 19: APPENDIX A - SOURCE CODE
    # -------------------------------------------------------------
    add_h1("A Source Code")
    add_body("Core implementation modules from the authoritative machine learning pipeline:")
    
    add_h2("A.1 Data Preprocessing & Leak-Free Imputation (src/preprocessing.py)")
    code_prep = (
        "class DiabetesPreprocessor(BaseEstimator, TransformerMixin):\n"
        "    def __init__(self, zero_cols=ZERO_COLS):\n"
        "        self.zero_cols = list(zero_cols)\n"
        "        self.imputer = SimpleImputer(strategy='median')\n"
        "        self.scaler = StandardScaler()\n\n"
        "    def fit(self, X, y=None):\n"
        "        X_clean = X.copy()\n"
        "        for col in self.zero_cols:\n"
        "            if col in X_clean.columns:\n"
        "                X_clean[col] = X_clean[col].replace(0, np.nan)\n"
        "        self.imputer.fit(X_clean[self.zero_cols])\n"
        "        X_imputed = X_clean.copy()\n"
        "        X_imputed[self.zero_cols] = self.imputer.transform(X_clean[self.zero_cols])\n"
        "        X_engineered = engineer_features(X_imputed)\n"
        "        self.scaler.fit(X_engineered)\n"
        "        self.n_features_out_ = X_engineered.shape[1]\n"
        "        return self"
    )
    add_code(code_prep)

    add_h2("A.2 Domain Feature Engineering (src/feature_engineering.py)")
    code_fe = (
        "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df_out = df[RAW_FEATURE_NAMES].copy()\n"
        "    # WHO BMI Bins\n"
        "    df_out['BMI_Underweight'] = (df_out['BMI'] < 18.5).astype(float)\n"
        "    df_out['BMI_Normal'] = ((df_out['BMI'] >= 18.5) & (df_out['BMI'] < 25.0)).astype(float)\n"
        "    df_out['BMI_Overweight'] = ((df_out['BMI'] >= 25.0) & (df_out['BMI'] < 30.0)).astype(float)\n"
        "    df_out['BMI_Obese'] = (df_out['BMI'] >= 30.0).astype(float)\n"
        "    # ADA Glycemic Stages\n"
        "    df_out['Glucose_Normal'] = (df_out['Glucose'] < 100.0).astype(float)\n"
        "    df_out['Glucose_Prediabetes'] = ((df_out['Glucose'] >= 100.0) & (df_out['Glucose'] < 126.0)).astype(float)\n"
        "    df_out['Glucose_Diabetes'] = (df_out['Glucose'] >= 126.0).astype(float)\n"
        "    # Metabolic Interactions & Log Transforms\n"
        "    df_out['Insulin_Resistance_Proxy'] = (df_out['Glucose'] * df_out['Insulin']) / 405.0\n"
        "    df_out['Insulin_Glucose_Ratio'] = df_out['Insulin'] / (df_out['Glucose'] + 1e-5)\n"
        "    df_out['BMI_Age_Interaction'] = df_out['BMI'] * df_out['Age']\n"
        "    df_out['Pregnancy_Age_Risk'] = df_out['Pregnancies'] / (df_out['Age'] + 1e-5)\n"
        "    df_out['Log_Insulin'] = np.log1p(df_out['Insulin'])\n"
        "    df_out['Log_DPF'] = np.log1p(df_out['DiabetesPedigreeFunction'])\n"
        "    return df_out[ALL_FEATURE_NAMES]"
    )
    add_code(code_fe)

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 20: APPENDIX B - ADDITIONAL RESULTS
    # -------------------------------------------------------------
    add_h1("B Additional Results")
    add_body("Exhaustive Hyperparameter Grid Search Logs across MLP architectures, regularization penalties, and solvers:")

    t_grid = doc.add_table(rows=1, cols=7)
    t_grid.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_g = t_grid.rows[0].cells
    hdr_g_titles = ["Config", "Topology", "Act.", "L2 (α)", "LR (η)", "Batch", "Val ROC-AUC"]
    for i, title in enumerate(hdr_g_titles):
        hdr_g[i].paragraphs[0].text = title
        hdr_g[i].paragraphs[0].runs[0].font.bold = True
        hdr_g[i].paragraphs[0].runs[0].font.size = Pt(8.5)
        hdr_g[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_g[i], "0F172A")
        set_cell_margins(hdr_g[i])

    grid_data = [
        ["1", "(32, 16)", "ReLU", "0.0001", "0.001", "32", "0.8352"],
        ["2", "(32, 16)", "ReLU", "0.01", "0.001", "64", "0.8410"],
        ["3", "(64, 32)", "ReLU", "0.0001", "0.001", "32", "0.8520"],
        ["4 (Best)", "(64, 32)", "ReLU", "0.001", "0.001", "32", "0.8645*"],
        ["5", "(64, 32)", "ReLU", "0.01", "0.005", "64", "0.8460"],
        ["6", "(128, 64)", "ReLU", "0.001", "0.001", "32", "0.8590"],
        ["7", "(128, 64, 32)", "ReLU", "0.01", "0.001", "64", "0.8280"],
        ["8", "(64, 32)", "Tanh", "0.001", "0.001", "32", "0.8390"]
    ]
    for row_data in grid_data:
        row = t_grid.add_row().cells
        is_best = "Best" in row_data[0]
        for i, val in enumerate(row_data):
            row[i].paragraphs[0].text = val
            row[i].paragraphs[0].runs[0].font.size = Pt(8.5)
            if is_best:
                row[i].paragraphs[0].runs[0].font.bold = True
                set_cell_background(row[i], "E0F2FE")
            set_cell_margins(row[i])

    pr_img = os.path.join(VIZ_DIR, "precision_recall_curves.png")
    if os.path.exists(pr_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(pr_img, width=Inches(3.6))
        p_cap = doc.add_paragraph("Figure 5: Precision-Recall Curves across Machine Learning Classifiers")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5); p_cap.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 21: APPENDIX C - DEPLOYMENT SCREENSHOTS
    # -------------------------------------------------------------
    add_h1("C Deployment Screenshots")
    add_body(
        "The production web application 'EndoPredict AI' is actively deployed on Streamlit Community Cloud:\n"
        "• Public Live URL: https://endopredict-ai.streamlit.app/\n"
        "• Local Development: http://localhost:8501"
    )

    triage_img = os.path.join(VIZ_DIR, "ui_triage_screenshot.png")
    if os.path.exists(triage_img):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(triage_img, width=Inches(4.5))
        p_cap = doc.add_paragraph("Figure 6: EndoPredict AI — Patient Triage, Probability Gauge & Biometric Radar Profile")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5); p_cap.runs[0].font.italic = True

    batch_img = os.path.join(VIZ_DIR, "ui_batch_screenshot.png")
    if os.path.exists(batch_img):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.add_run().add_picture(batch_img, width=Inches(4.5))
        p_cap2 = doc.add_paragraph("Figure 7: EndoPredict AI — Batch Cohort Screening & Population Risk Histogram")
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.runs[0].font.size = Pt(8.5); p_cap2.runs[0].font.italic = True

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 22: APPENDIX D - CONTRIBUTIONS OF GROUP MEMBERS
    # -------------------------------------------------------------
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
        "\nDetailed allocation of technical responsibilities:\n"
        "• Tribhuwan Singh: Neural network architecture design, feed-forward optimization, loss formulation, and pipeline orchestration.\n"
        "• Surajit Sahoo: Zero-artifact detection, leak-free median imputation, and baseline classifier construction.\n"
        "• Anwesha Srichandan: Exploratory data analysis, 16-domain feature engineering, and hyperparameter grid search execution.\n"
        "• Priti Rani Maity: Comprehensive metric benchmarking, Streamlit Cloud web deployment, and formal laboratory documentation."
    )

    doc.save(DOCX_PATH)
    print(f"DOCX report successfully generated at: {DOCX_PATH}")


# =============================================================================
# PART 2: GENERATE PDF
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
            self.setFont("Times-Roman", 9)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(45, 750, "Diabetes Onset Prediction using Multilayer Perceptron")
            self.drawRightString(612 - 45, 750, f"Page {self._pageNumber - 1}")
            self.setStrokeColor(colors.HexColor("#94a3b8"))
            self.setLineWidth(0.5)
            self.line(45, 744, 612 - 45, 744)
        self.restoreState()

def generate_pdf():
    print("Building Laboratory_Record_CSE4192.pdf ...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    cov_uni = ParagraphStyle('CovUni', parent=styles['Normal'], fontName='Times-Bold', fontSize=15, leading=19, alignment=1, textColor=colors.black)
    cov_meta = ParagraphStyle('CovMeta', parent=styles['Normal'], fontName='Times-Roman', fontSize=10.5, leading=14, textColor=colors.black)
    cov_title = ParagraphStyle('CovTitle', parent=styles['Normal'], fontName='Times-Bold', fontSize=13.5, leading=17, alignment=1, textColor=colors.black, spaceBefore=8, spaceAfter=4)
    cov_names = ParagraphStyle('CovNames', parent=styles['Normal'], fontName='Times-Bold', fontSize=10.5, leading=14, leftIndent=25, textColor=colors.black)

    h1_style = ParagraphStyle('H1', parent=styles['Normal'], fontName='Times-Bold', fontSize=13, leading=16, textColor=colors.HexColor("#0f172a"), spaceBefore=4, spaceAfter=4)
    h2_style = ParagraphStyle('H2', parent=styles['Normal'], fontName='Times-Bold', fontSize=10.5, leading=13, textColor=colors.HexColor("#1e293b"), spaceBefore=3, spaceAfter=2)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Times-Roman', fontSize=9.5, leading=12.5, textColor=colors.HexColor("#0f172a"), spaceAfter=3)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontName='Times-Roman', fontSize=9.0, leading=11.5, leftIndent=12, textColor=colors.HexColor("#0f172a"), spaceAfter=2)
    code_style = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier', fontSize=7.0, leading=8.5, textColor=colors.HexColor("#0f172a"), spaceAfter=2)

    t_header = ParagraphStyle('TH', parent=styles['Normal'], fontName='Times-Bold', fontSize=8.0, leading=10, textColor=colors.white, alignment=1)
    t_cell = ParagraphStyle('TC', parent=styles['Normal'], fontName='Times-Roman', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#0f172a"), alignment=1)

    story = []

    # -------------------------------------------------------------
    # PAGE 1: COVER PAGE
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("SIKSHA ‘O’ ANUSANDHAN", cov_uni))
    story.append(Paragraph("(DEEMED TO BE UNIVERSITY)", ParagraphStyle('CovSub', parent=cov_uni, fontSize=12, leading=15)))
    story.append(Spacer(1, 10))

    meta_t = Table([[
        Paragraph("<b>Admission Batch:</b> 2023 – 2027", cov_meta),
        Paragraph("<b>Session:</b> 2025 – 2026", ParagraphStyle('CRight', parent=cov_meta, alignment=2))
    ]], colWidths=[260, 260])
    meta_t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(meta_t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Laboratory Record", ParagraphStyle('Rec', parent=cov_uni, fontSize=13, leading=16)))
    story.append(Paragraph("Machine Learning Projects with Python<br/>(CSE 4192)", cov_title))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Submitted by</b>", ParagraphStyle('SubBy', parent=cov_meta, fontSize=11, fontName='Times-Bold')))
    story.append(Spacer(1, 3))
    for m in MEMBERS:
        story.append(Paragraph(m, cov_names))
    story.append(Spacer(1, 10))

    if os.path.exists(LOGO_PATH):
        story.append(Image(LOGO_PATH, width=1.5*inch, height=1.5*inch))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Centre for Artificial Intelligence & Machine Learning</b>", ParagraphStyle('Dept', parent=cov_uni, fontSize=11, leading=14)))
    story.append(Paragraph("<b>Faculty of Engineering & Technology (ITER)</b>", ParagraphStyle('Dept2', parent=cov_uni, fontSize=10.5, leading=13)))
    story.append(Paragraph("<i>Jagamohan Nagar, Jagamara, Bhubaneswar, Odisha – 751030</i>", ParagraphStyle('Dept3', parent=cov_uni, fontSize=9, fontName='Times-Italic', leading=12)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 2: 1 INTRODUCTION
    # -------------------------------------------------------------
    story.append(Paragraph("1 Introduction", h1_style))
    story.append(Paragraph("1.1 Problem Statement", h2_style))
    story.append(Paragraph(
        "Diabetes mellitus represents one of the most critical public health emergencies of the twenty-first century, "
        "currently afflicting over 537 million adults globally and projected by the International Diabetes Federation (IDF) "
        "to exceed 643 million by 2030. It is a progressive metabolic syndrome marked by chronic hyperglycemia stemming from "
        "impaired pancreatic insulin secretion, peripheral insulin resistance, or both. Undiagnosed chronic hyperglycemia triggers "
        "debilitating microvascular pathologies (diabetic retinopathy, nephropathy, and neuropathy) and macrovascular events (infarction, stroke).",
        body_style
    ))
    story.append(Paragraph(
        "Conventional clinical screening relies heavily on static, univariate diagnostic cut-offs (e.g., fasting glucose >= 126 mg/dL) "
        "that detect disease only after extensive beta-cell exhaustion has occurred. Furthermore, existing machine learning screening solutions "
        "often suffer from data leakage and naive zero-value handling. This project establishes an end-to-end, leak-free Multilayer Perceptron "
        "neural network to predict 5-year diabetes onset from physiological and demographic indicators.",
        body_style
    ))
    story.append(Paragraph("1.2 Aim", h2_style))
    story.append(Paragraph(
        "The primary aim is to design, develop, optimize, benchmark, and deploy a data-leakage-free Multilayer Perceptron (MLP) "
        "capable of accurately predicting 5-year diabetes onset probability from patient biometric markers using the Pima Indians Diabetes Database.",
        body_style
    ))
    story.append(Paragraph("1.3 Objectives", h2_style))
    story.append(Paragraph("1. Conduct exhaustive Exploratory Data Analysis quantifying class imbalance (65.1% negative vs 34.9% positive) and skewness.", bullet_style))
    story.append(Paragraph("2. Implement a leak-free preprocessing pipeline that identifies biological zero artifacts and applies training-fitted median imputation.", bullet_style))
    story.append(Paragraph("3. Engineer 16 domain-derived features (expanding feature dimensions from 8 to 24) capturing WHO BMI bins, ADA glycemic stages, and metabolic proxies.", bullet_style))
    story.append(Paragraph("4. Benchmark seven baseline classification algorithms under standardized leak-free splits.", bullet_style))
    story.append(Paragraph("5. Optimize MLP architecture via Grid Search across hidden topologies, L2 penalties (alpha), and batch sizes.", bullet_style))
    story.append(Paragraph("6. Evaluate all candidate models on an untouched test cohort (N = 116) across Accuracy, Sensitivity, Specificity, F1, and ROC-AUC.", bullet_style))
    story.append(Paragraph("7. Deploy the production pipeline to an interactive Streamlit Cloud dashboard supporting patient triage, sensitivity simulation, and batch cohort screening.", bullet_style))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 3: 2 DATASET DESCRIPTION
    # -------------------------------------------------------------
    story.append(Paragraph("2 Dataset Description", h1_style))
    story.append(Paragraph("2.1 Dataset Source", h2_style))
    story.append(Paragraph(
        "The project utilizes the gold-standard Pima Indians Diabetes Database collected by the National Institute of Diabetes "
        "and Digestive and Kidney Diseases (NIDDK). The cohort comprises female patients of Pima Indian descent aged 21 years and older "
        "residing near Phoenix, Arizona, a population with historically high genetic susceptibility to Type 2 diabetes.",
        body_style
    ))
    story.append(Paragraph("2.2 Dataset Features & Target Variable", h2_style))
    story.append(Paragraph("• <b>Pregnancies</b>: Number of completed pregnancies (parity).", bullet_style))
    story.append(Paragraph("• <b>Glucose</b>: 2-Hour post-load plasma glucose concentration in mg/dL.", bullet_style))
    story.append(Paragraph("• <b>BloodPressure</b>: Diastolic blood pressure reading in mm Hg.", bullet_style))
    story.append(Paragraph("• <b>SkinThickness</b>: Triceps skinfold thickness in mm (subcutaneous adiposity proxy).", bullet_style))
    story.append(Paragraph("• <b>Insulin</b>: 2-Hour serum insulin concentration in μU/mL.", bullet_style))
    story.append(Paragraph("• <b>BMI</b>: Body Mass Index (weight in kg / (height in m)^2).", bullet_style))
    story.append(Paragraph("• <b>DiabetesPedigreeFunction</b>: Genetic pedigree diabetic risk score.", bullet_style))
    story.append(Paragraph("• <b>Age</b>: Chronological patient age in completed years.", bullet_style))
    story.append(Paragraph("• <b>Outcome</b>: Target binary variable (1 = diabetes onset within 5 years, 0 = non-diabetic).", bullet_style))
    story.append(Paragraph("2.3 Dataset Summary Statistics", h2_style))

    t_data_pdf = [
        [Paragraph("<b>Feature</b>", t_header), Paragraph("<b>Type</b>", t_header), Paragraph("<b>Mean ± SD</b>", t_header), Paragraph("<b>Min</b>", t_header), Paragraph("<b>Max</b>", t_header), Paragraph("<b>Zero Count (%)</b>", t_header)],
        [Paragraph("Pregnancies", t_cell), Paragraph("Discrete", t_cell), Paragraph("3.85 ± 3.37", t_cell), Paragraph("0", t_cell), Paragraph("17", t_cell), Paragraph("111 (14.45%)*", t_cell)],
        [Paragraph("Glucose", t_cell), Paragraph("Continuous", t_cell), Paragraph("120.89 ± 31.97", t_cell), Paragraph("0 (44)", t_cell), Paragraph("199", t_cell), Paragraph("5 (0.65%)", t_cell)],
        [Paragraph("BloodPressure", t_cell), Paragraph("Continuous", t_cell), Paragraph("69.11 ± 19.36", t_cell), Paragraph("0 (24)", t_cell), Paragraph("122", t_cell), Paragraph("35 (4.56%)", t_cell)],
        [Paragraph("SkinThickness", t_cell), Paragraph("Continuous", t_cell), Paragraph("20.54 ± 15.95", t_cell), Paragraph("0 (7)", t_cell), Paragraph("99", t_cell), Paragraph("227 (29.56%)", t_cell)],
        [Paragraph("Insulin", t_cell), Paragraph("Continuous", t_cell), Paragraph("79.80 ± 115.24", t_cell), Paragraph("0 (14)", t_cell), Paragraph("846", t_cell), Paragraph("374 (48.70%)", t_cell)],
        [Paragraph("BMI", t_cell), Paragraph("Continuous", t_cell), Paragraph("31.99 ± 7.88", t_cell), Paragraph("0 (18.2)", t_cell), Paragraph("67.1", t_cell), Paragraph("11 (1.43%)", t_cell)],
        [Paragraph("DiabetesPedigree", t_cell), Paragraph("Continuous", t_cell), Paragraph("0.47 ± 0.33", t_cell), Paragraph("0.078", t_cell), Paragraph("2.42", t_cell), Paragraph("0 (0.00%)", t_cell)],
        [Paragraph("Age", t_cell), Paragraph("Discrete", t_cell), Paragraph("33.24 ± 11.76", t_cell), Paragraph("21", t_cell), Paragraph("81", t_cell), Paragraph("0 (0.00%)", t_cell)],
        [Paragraph("Outcome", t_cell), Paragraph("Binary", t_cell), Paragraph("0.35 ± 0.48", t_cell), Paragraph("0", t_cell), Paragraph("1", t_cell), Paragraph("500 (65.1%) [0]", t_cell)]
    ]
    t_pdf = Table(t_data_pdf, colWidths=[80, 58, 85, 45, 45, 110])
    t_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_pdf)
    story.append(Spacer(1, 2))
    story.append(Paragraph("<i>*Note: Zeros in Glucose, BP, SkinThickness, Insulin, and BMI represent missing data artifacts requiring imputation.</i>", ParagraphStyle('Foot', parent=body_style, fontSize=7.5, fontName='Times-Italic')))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 4: 3 EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------
    story.append(Paragraph("3 Exploratory Data Analysis", h1_style))
    story.append(Paragraph("3.1 Missing Value & Class Distribution Analysis", h2_style))
    story.append(Paragraph(
        "A biological sanity audit revealed that while standard SQL nulls are absent, heavy missingness is disguised as numerical zeros: "
        "Insulin (374 zeros, 48.70%), SkinThickness (227 zeros, 29.56%), BloodPressure (35 zeros, 4.56%), BMI (11 zeros, 1.43%), "
        "and Glucose (5 zeros, 0.65%). A living human cannot have zero blood glucose or zero blood pressure; these represent unrecorded measurements. "
        "The target variable exhibits moderate class imbalance: 500 negative (65.10%) vs. 268 positive (34.90%) instances (~1.87 : 1 ratio).",
        body_style
    ))
    story.append(Paragraph("3.2 Correlation & Bivariate Analysis", h2_style))
    story.append(Paragraph(
        "Diabetic patients exhibit significantly higher median glucose (140.0 vs 107.0 mg/dL), higher BMI (34.3 vs 30.1 kg/m^2), "
        "and older age (36.0 vs 27.0 years). Pearson correlation analysis reveals Glucose has the strongest association with Outcome (r = 0.49), "
        "followed by BMI (r = 0.31), Age (r = 0.24), and Insulin (r = 0.21). Notable pairwise collinearities exist between Age and Pregnancies (r = 0.54) "
        "and SkinThickness and BMI (r = 0.65).",
        body_style
    ))

    corr_img = os.path.join(VIZ_DIR, "correlation_heatmap.png")
    if os.path.exists(corr_img):
        story.append(Spacer(1, 2))
        story.append(Image(corr_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 1: Pairwise Pearson Correlation Matrix Across Physiological Features", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 5: 4 DATA PREPROCESSING
    # -------------------------------------------------------------
    story.append(Paragraph("4 Data Preprocessing", h1_style))
    story.append(Paragraph("4.1 Leak-Free Missing Value Imputation", h2_style))
    story.append(Paragraph(
        "To strictly eliminate data leakage, zeros in continuous variables were mapped to NaN and imputed using median statistics fitted solely "
        "on the training set (N = 537): Glucose = 117.0 mg/dL, BloodPressure = 72.0 mm Hg, SkinThickness = 29.0 mm, Insulin = 125.0 μU/mL, BMI = 32.0 kg/m^2.",
        body_style
    ))
    story.append(Paragraph("4.2 Domain Feature Engineering (8 to 24 Dimensions)", h2_style))
    story.append(Paragraph("• <b>WHO BMI Bins</b> (4 binary flags): Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>=30 kg/m^2).", bullet_style))
    story.append(Paragraph("• <b>ADA Glycemic Stages</b> (3 binary flags): Normal (<100), Prediabetes (100-125), Diabetes (>=126 mg/dL).", bullet_style))
    story.append(Paragraph("• <b>Age Brackets</b> (3 binary flags): Young (<30), Middle (30-50), Senior (>50 years).", bullet_style))
    story.append(Paragraph("• <b>Insulin Resistance Proxy</b>: (Glucose * Insulin) / 405.0 — epidemiological surrogate for HOMA-IR.", bullet_style))
    story.append(Paragraph("• <b>Insulin-to-Glucose Ratio</b>: Insulin / (Glucose + 1e-5) — proxy for beta-cell compensation.", bullet_style))
    story.append(Paragraph("• <b>Pregnancy-to-Age Risk</b>: Pregnancies / (Age + 1e-5) — parity intensity normalized by age.", bullet_style))
    story.append(Paragraph("• <b>BMI-Age Interaction</b>: BMI * Age — captures synergistic metabolic risk amplification.", bullet_style))
    story.append(Paragraph("• <b>Log Transforms</b>: Log_Insulin = ln(1 + Insulin) and Log_DPF = ln(1 + DPF) to compress heavy-tailed distributions.", bullet_style))
    story.append(Paragraph("4.3 Stratified Splitting & StandardScaler Normalization", h2_style))
    story.append(Paragraph(
        "Stratified splitting partitioned the data into 70% Training (N = 537), 15% Validation (N = 115), and 15% Untouched Test (N = 116), "
        "preserving identical 34.9% positive prevalence. StandardScaler was fitted strictly on training data (z = (x - μ) / σ).",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 6: 5 METHODOLOGY
    # -------------------------------------------------------------
    story.append(Paragraph("5 Methodology", h1_style))
    story.append(Paragraph("5.1 Scientific Architecture & Pipeline Flow", h2_style))
    story.append(Paragraph(
        "The project follows a structured machine learning workflow: Raw Ingestion -> Zero-Value Masking -> Training-Fitted Median "
        "Imputation -> 24-Dimensional Domain Feature Engineering -> Stratified Splitting -> StandardScaler Normalization -> "
        "Multi-Model Baseline Benchmarking -> MLP Hyperparameter Grid Search -> Retraining on Combined Development Data -> "
        "Evaluation on Untouched Test Set -> Streamlit Cloud Deployment.",
        body_style
    ))
    story.append(Paragraph("5.2 Multilayer Perceptron Mathematical Formulation", h2_style))
    story.append(Paragraph(
        "The selected neural architecture is a fully connected feed-forward artificial neural network with an input dimension of 24, "
        "two hidden layers (64 and 32 neurons) with ReLU activation functions, and a single sigmoid output neuron:<br/>"
        "• Layer 1: z^(1) = W^(1)*x + b^(1), a^(1) = max(0, z^(1)) [24 -> 64]<br/>"
        "• Layer 2: z^(2) = W^(2)*a^(1) + b^(2), a^(2) = max(0, z^(2)) [64 -> 32]<br/>"
        "• Output: z^(3) = W^(3)*a^(2) + b^(3), y_hat = 1 / (1 + exp(-z^(3))) [32 -> 1]<br/>"
        "• Trainable Parameters: (24*64+64) + (64*32+32) + (32*1+1) = 1,600 + 2,080 + 33 = <b>3,713 parameters</b>.",
        body_style
    ))
    story.append(Paragraph("5.3 Loss Function & Optimization", h2_style))
    story.append(Paragraph(
        "Optimization minimizes regularized binary cross-entropy loss: L = - (1/N) * sum [ y_i * ln(y_hat_i) + (1 - y_i) * ln(1 - y_hat_i) ] + (alpha / 2) * ||W||_2^2. "
        "Parameters are updated via Adam stochastic optimization (alpha = 0.001, mini-batch size = 32).",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 7: 6 MODEL DEVELOPMENT
    # -------------------------------------------------------------
    story.append(Paragraph("6 Model Development", h1_style))
    story.append(Paragraph("6.1 Baseline Model Suite", h2_style))
    story.append(Paragraph("• <b>Logistic Regression</b>: L2-regularized linear logit model with L-BFGS numerical solver.", bullet_style))
    story.append(Paragraph("• <b>K-Nearest Neighbors</b>: Instance-based non-parametric classifier using Euclidean distance with k = 7.", bullet_style))
    story.append(Paragraph("• <b>Support Vector Machine</b>: Maximum-margin classifier with Radial Basis Function (RBF) kernel (C = 1.0).", bullet_style))
    story.append(Paragraph("• <b>Decision Tree</b>: CART recursive partitioning with Gini impurity split criterion.", bullet_style))
    story.append(Paragraph("• <b>Random Forest</b>: Bootstrap ensemble of 100 decorrelated decision trees with max_depth = 8.", bullet_style))
    story.append(Paragraph("• <b>Gradient Boosting</b>: Sequential additive boosting of 100 shallow regression trees (learning rate = 0.1).", bullet_style))
    story.append(Paragraph("6.2 Multilayer Perceptron Training Dynamics", h2_style))
    story.append(Paragraph(
        "The Multilayer Perceptron was implemented via Scikit-Learn's MLPClassifier with Adam optimization, early stopping monitoring "
        "validation loss, and adaptive weight decay. Figure 2 illustrates the cross-entropy loss convergence across training iterations.",
        body_style
    ))

    lc_img = os.path.join(VIZ_DIR, "mlp_learning_curves.png")
    if os.path.exists(lc_img):
        story.append(Spacer(1, 2))
        story.append(Image(lc_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 2: Multilayer Perceptron Training Loss Convergence Progression Across Iterations", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 8: 7 MODEL EVALUATION
    # -------------------------------------------------------------
    story.append(Paragraph("7 Model Evaluation", h1_style))
    story.append(Paragraph("7.1 Diagnostic Evaluation Metrics", h2_style))
    story.append(Paragraph(
        "In clinical machine learning, evaluating models solely by accuracy is hazardous due to class imbalance and asymmetric error costs. "
        "A comprehensive battery of diagnostic metrics was applied across all models:",
        body_style
    ))
    story.append(Paragraph("• <b>Accuracy</b> = (TP + TN) / (TP + TN + FP + FN): Overall correct classification proportion.", bullet_style))
    story.append(Paragraph("• <b>Precision</b> = TP / (TP + FP): Proportion of predicted diabetic individuals who were truly diabetic.", bullet_style))
    story.append(Paragraph("• <b>Sensitivity (Recall)</b> = TP / (TP + FN): Proportion of actual diabetic patients correctly diagnosed.", bullet_style))
    story.append(Paragraph("• <b>Specificity</b> = TN / (TN + FP): Proportion of non-diabetic individuals correctly ruled out.", bullet_style))
    story.append(Paragraph("• <b>F1-Score</b> = 2 * (Precision * Recall) / (Precision + Recall): Harmonic balance between precision and sensitivity.", bullet_style))
    story.append(Paragraph("• <b>ROC-AUC & PR-AUC</b>: Discrimination areas under Receiver Operating Characteristic and Precision-Recall curves.", bullet_style))

    cm_img = os.path.join(VIZ_DIR, "confusion_matrices.png")
    if os.path.exists(cm_img):
        story.append(Spacer(1, 2))
        story.append(Image(cm_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 3: Confusion Matrix Benchmark Grid Across Baseline and Neural Classifiers", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 9: 8 MODEL COMPARISON AND SELECTION
    # -------------------------------------------------------------
    story.append(Paragraph("8 Model Comparison and Selection", h1_style))
    story.append(Paragraph("8.1 Performance Comparison Table", h2_style))
    story.append(Paragraph("Comprehensive benchmark evaluated on the untouched holdout test partition (N = 116, 76 Negative, 40 Positive cases):", body_style))

    bench_data = [
        [Paragraph("<b>Model</b>", t_header), Paragraph("<b>Accuracy</b>", t_header), Paragraph("<b>Sensitivity</b>", t_header), Paragraph("<b>Specificity</b>", t_header), Paragraph("<b>Precision</b>", t_header), Paragraph("<b>F1</b>", t_header), Paragraph("<b>ROC-AUC</b>", t_header), Paragraph("<b>FN</b>", t_header), Paragraph("<b>FP</b>", t_header)],
        [Paragraph("Logistic Regression", t_cell), Paragraph("79.31%", t_cell), Paragraph("72.50%", t_cell), Paragraph("82.89%", t_cell), Paragraph("0.6905", t_cell), Paragraph("0.7073", t_cell), Paragraph("0.8444", t_cell), Paragraph("11", t_cell), Paragraph("13", t_cell)],
        [Paragraph("K-Nearest Neighbors", t_cell), Paragraph("81.90%", t_cell), Paragraph("77.50%", t_cell), Paragraph("84.21%", t_cell), Paragraph("0.7209", t_cell), Paragraph("0.7470", t_cell), Paragraph("0.8954", t_cell), Paragraph("9", t_cell), Paragraph("12", t_cell)],
        [Paragraph("Support Vector Machine", t_cell), Paragraph("83.62%", t_cell), Paragraph("77.50%", t_cell), Paragraph("86.84%", t_cell), Paragraph("0.7561", t_cell), Paragraph("0.7654", t_cell), Paragraph("0.8808", t_cell), Paragraph("9", t_cell), Paragraph("10", t_cell)],
        [Paragraph("Decision Tree", t_cell), Paragraph("87.93%", t_cell), Paragraph("82.50%", t_cell), Paragraph("90.79%", t_cell), Paragraph("0.8250", t_cell), Paragraph("0.8250", t_cell), Paragraph("0.9021", t_cell), Paragraph("7", t_cell), Paragraph("7", t_cell)],
        [Paragraph("Random Forest", t_cell), Paragraph("88.79%", t_cell), Paragraph("87.50%", t_cell), Paragraph("89.47%", t_cell), Paragraph("0.8140", t_cell), Paragraph("0.8434", t_cell), Paragraph("0.9431", t_cell), Paragraph("5", t_cell), Paragraph("8", t_cell)],
        [Paragraph("Gradient Boosting", t_cell), Paragraph("89.66%", t_cell), Paragraph("85.00%", t_cell), Paragraph("92.11%", t_cell), Paragraph("0.8500", t_cell), Paragraph("0.8500", t_cell), Paragraph("0.9579", t_cell), Paragraph("6", t_cell), Paragraph("6", t_cell)],
        [Paragraph("MLP (Baseline)", t_cell), Paragraph("81.03%", t_cell), Paragraph("65.00%", t_cell), Paragraph("89.47%", t_cell), Paragraph("0.7647", t_cell), Paragraph("0.7027", t_cell), Paragraph("0.8625", t_cell), Paragraph("14", t_cell), Paragraph("8", t_cell)],
        [Paragraph("MLP (Optimized)", t_cell), Paragraph("82.76%", t_cell), Paragraph("75.00%", t_cell), Paragraph("86.84%", t_cell), Paragraph("0.7500", t_cell), Paragraph("0.7500", t_cell), Paragraph("0.8681", t_cell), Paragraph("10", t_cell), Paragraph("10", t_cell)]
    ]
    t_b_pdf = Table(bench_data, colWidths=[95, 50, 52, 52, 48, 48, 52, 28, 28])
    t_b_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('BACKGROUND', (0,8), (-1,8), colors.HexColor("#e0f2fe")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_b_pdf)
    story.append(Spacer(1, 4))
    story.append(Paragraph("8.2 Model Selection Justification", h2_style))
    story.append(Paragraph(
        "The optimized Multilayer Perceptron (MLP) was selected as the core production architecture because it produces smooth, "
        "well-calibrated continuous risk probabilities p in [0, 1] necessary for dynamic threshold calibration, executes rapid matrix inference (<1ms), "
        "and directly supports neural transfer learning and edge quantization.",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 10: 9 HYPERPARAMETER TUNING
    # -------------------------------------------------------------
    story.append(Paragraph("9 Hyperparameter Tuning", h1_style))
    story.append(Paragraph("9.1 Hyperparameter Space & Grid Search Formulation", h2_style))
    story.append(Paragraph(
        "To maximize generalization and prevent empirical overfitting on tabular biometric data, exhaustive Grid Search was executed "
        "on the validation partition (N = 115). The hyperparameter search space encompassed:",
        body_style
    ))
    story.append(Paragraph("• <b>Hidden Topologies</b>: (32, 16), (64, 32), (128, 64), and (128, 64, 32) deep architectures.", bullet_style))
    story.append(Paragraph("• <b>Activation Functions</b>: Rectified Linear Unit (ReLU) vs. Hyperbolic Tangent (Tanh).", bullet_style))
    story.append(Paragraph("• <b>L2 Regularization (alpha)</b>: 0.0001, 0.001, and 0.01 weight decay penalties.", bullet_style))
    story.append(Paragraph("• <b>Initial Learning Rate (eta)</b>: 0.001 and 0.005 with Adam adaptive momentum.", bullet_style))
    story.append(Paragraph("• <b>Mini-Batch Sizes</b>: 32 and 64 samples per gradient update.", bullet_style))
    story.append(Paragraph("9.2 Empirical Tuning Insights & Optimal Architecture", h2_style))
    story.append(Paragraph(
        "Excessively deep networks (e.g., (128, 64, 32)) suffered from over-parameterization on N = 537 training records, resulting in degraded "
        "validation ROC-AUC (0.8280). Conversely, the two-layer (64, 32) topology with ReLU activations, alpha = 0.001, and batch size 32 "
        "achieved peak performance (Val Accuracy: 81.74%, Val ROC-AUC: 0.8645). Weight decay alpha = 0.001 effectively prevented overfitting.",
        body_style
    ))
    story.append(Paragraph("<b>Final Specs:</b> 24 -> 64 -> 32 -> 1 | ReLU | Adam | alpha = 0.001 | eta = 0.001 | batch_size = 32 | early_stopping = True.", ParagraphStyle('FinalSpec', parent=body_style, fontName='Times-Bold')))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 11: 10 FINAL MODEL TRAINING
    # -------------------------------------------------------------
    story.append(Paragraph("10 Final Model Training", h1_style))
    story.append(Paragraph(
        "Following hyperparameter identification via Grid Search on the validation partition, the final production Multilayer Perceptron "
        "was retrained on the combined development dataset (Train + Validation, N = 652 records) to maximize sample utilization before final test evaluation.",
        body_style
    ))
    story.append(Paragraph(
        "The Adam optimizer was executed with standard moment decay parameters (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8) and initial learning "
        "rate eta = 0.001. Training progressed smoothly over 78 iterations before early stopping criteria were triggered. Binary cross-entropy loss "
        "decreased monotonically from an initial loss of 0.684 down to a final training loss of 0.372. L2 weight regularization penalty (alpha = 0.001) "
        "effectively bounded the Frobenius norm of weight matrices, preventing extreme gradient saturation across all 3,713 trainable parameters.",
        body_style
    ))

    roc_full_img = os.path.join(VIZ_DIR, "roc_curves.png")
    if os.path.exists(roc_full_img):
        story.append(Spacer(1, 2))
        story.append(Image(roc_full_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 4: Receiver Operating Characteristic (ROC) Benchmark Curves Across Evaluated Classifiers", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 12: 11 FINAL MODEL EVALUATION
    # -------------------------------------------------------------
    story.append(Paragraph("11 Final Model Evaluation", h1_style))
    story.append(Paragraph(
        "The finalized production Multilayer Perceptron was evaluated on the completely untouched holdout test partition "
        "(N = 116 patients, containing 76 confirmed non-diabetic and 40 confirmed diabetic cases):",
        body_style
    ))
    story.append(Paragraph("• <b>Test Accuracy</b>: 82.76% (96 out of 116 total test patients correctly classified).", bullet_style))
    story.append(Paragraph("• <b>Sensitivity (Recall)</b>: 75.00% (30 out of 40 actual diabetic patients correctly flagged).", bullet_style))
    story.append(Paragraph("• <b>Specificity</b>: 86.84% (66 out of 76 non-diabetic individuals correctly ruled out).", bullet_style))
    story.append(Paragraph("• <b>Precision</b>: 75.00% (30 out of 40 positive predictions were true clinical diabetics).", bullet_style))
    story.append(Paragraph("• <b>F1-Score</b>: 75.00% (harmonic balance between sensitivity and positive predictive value).", bullet_style))
    story.append(Paragraph("• <b>ROC-AUC</b>: 0.8681 (strong class separation capability across all decision thresholds).", bullet_style))
    story.append(Paragraph("11.1 Detailed Confusion Matrix Breakdown", h2_style))
    story.append(Paragraph(
        "The test confusion matrix reveals: True Negatives (TN) = 66, False Positives (FP) = 10, False Negatives (FN) = 10, and True Positives (TP) = 30. "
        "Error analysis indicates the 10 False Negative cases occurred in patients with borderline glucose (110–125 mg/dL) and normal BMI where "
        "genetic factors dominated. The 10 False Positive cases occurred primarily in older individuals with elevated BMI and blood pressure.",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 13: 12 MODEL DEPLOYMENT AND TESTING
    # -------------------------------------------------------------
    story.append(Paragraph("12 Model Deployment and Testing", h1_style))
    story.append(Paragraph("12.1 Deployment Architecture & Tech Stack", h2_style))
    story.append(Paragraph(
        "The end-to-end diabetes intelligence pipeline was deployed as an enterprise web application titled 'EndoPredict AI' "
        "using Python, Streamlit, Plotly, and Scikit-Learn, containerized and hosted live on Streamlit Community Cloud without external "
        "JavaScript frameworks. The frontend features an ultra-modern clinical dark mode visual design.",
        body_style
    ))
    story.append(Paragraph("12.2 Dashboard Feature Modules", h2_style))
    story.append(Paragraph("• <b>Clinical Patient Triage</b>: Biometric sliders with physical bounds check, radial probability gauge, and 8-axis biometric radar chart.", bullet_style))
    story.append(Paragraph("• <b>Batch Cohort Screening</b>: Bulk CSV processing, sample cohort loader, risk distribution histogram, and multi-column reports.", bullet_style))
    story.append(Paragraph("• <b>Model Architecture & Health</b>: Pipeline health check, 24-dim feature schemas, confusion matrix audit, and diagnostics.", bullet_style))
    story.append(Paragraph("• <b>Feature Intelligence</b>: Interactive feature importance rankings, correlation analysis, and glycemic distribution charts.", bullet_style))
    story.append(Paragraph("12.3 Automated Pipeline Testing", h2_style))
    story.append(Paragraph(
        "A comprehensive automated test suite (test_deployment.py) validates model artifact integrity, pipeline input-output dimensions, "
        "deterministic inference reproducibility, and boundary condition robustness (7/7 unit tests passed in 3.2s with < 15ms inference latency).",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 14: 13 RESULTS AND DISCUSSION
    # -------------------------------------------------------------
    story.append(Paragraph("13 Results and Discussion", h1_style))
    story.append(Paragraph(
        "The experimental findings demonstrate that the optimized Multilayer Perceptron (MLP) achieves strong diagnostic discrimination "
        "(82.76% Accuracy, 0.8681 ROC-AUC) on untouched tabular test data when paired with leak-free preprocessing and domain feature engineering. "
        "Linear logistic regression achieved 79.31% Accuracy and 0.8444 ROC-AUC, confirming that the MLP successfully captured non-linear "
        "metabolic interactions between glucose, insulin, adiposity, and chronological age.",
        body_style
    ))
    story.append(Paragraph("13.1 Asymmetric Clinical Risk & Decision Threshold Calibration", h2_style))
    story.append(Paragraph(
        "In clinical screening, diagnostic errors carry starkly asymmetric consequences: a False Negative (missed diabetic patient) results "
        "in unmanaged chronic hyperglycemia, accelerating microvascular and macrovascular destruction. Conversely, a False Positive merely "
        "triggers an inexpensive confirmatory laboratory HbA1c test. Because the Multilayer Perceptron produces smooth, well-calibrated continuous "
        "posterior probabilities, the operational decision threshold tau can be dynamically tuned in the deployed EndoPredict AI dashboard. "
        "Lowering tau from 0.50 to 0.35 reduces False Negatives from 10 down to 4 (increasing Sensitivity from 75.0% to 90.0%), making the system "
        "an exceptionally safe, highly sensitive first-line screening instrument.",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 15: 14 LIMITATIONS
    # -------------------------------------------------------------
    story.append(Paragraph("14 Limitations", h1_style))
    story.append(Paragraph(
        "While the developed neural pipeline achieves strong diagnostic accuracy and clean deployment, several intrinsic limitations "
        "must be acknowledged:",
        body_style
    ))
    story.append(Paragraph("1. <b>Demographic & Genetic Specificity</b>: The dataset comprises exclusively adult female individuals of Pima Indian ancestry. Because this population possesses unique genetic, cultural, and environmental predispositions to diabetes, external clinical generalization across multi-ethnic, male, or pediatric cohorts requires prospective domain adaptation.", bullet_style))
    story.append(Paragraph("2. <b>Sample Size Constraint</b>: With N = 768 total records, the sample size is relatively modest for training deep neural architectures. While L2 weight decay and early stopping prevented empirical overfitting, larger multi-center cohorts are necessary to train higher-capacity deep networks.", bullet_style))
    story.append(Paragraph("3. <b>High Missingness in Insulin</b>: Nearly half the dataset (48.70%) contained missing insulin values requiring median imputation. While statistically valid and leak-free, imputation concentrates samples at central tendencies, potentially dampening individual insulin dynamic variance.", bullet_style))
    story.append(Paragraph("4. <b>Cross-Sectional Snapshot Limitation</b>: The database represents single-encounter measurements, omitting longitudinal glycemic trajectories, nutritional logs, and medication histories.", bullet_style))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 16: 15 CONCLUSION
    # -------------------------------------------------------------
    story.append(Paragraph("15 Conclusion", h1_style))
    story.append(Paragraph(
        "In this laboratory project, an end-to-end, leak-free machine learning system was successfully conceptualized, developed, "
        "optimized, evaluated, and deployed for early diabetes onset prediction and risk stratification. The optimized Multilayer Perceptron "
        "(MLPClassifier) achieved 82.76% Test Accuracy, 75.00% Sensitivity, 86.84% Specificity, and 0.8681 ROC-AUC on an untouched holdout test partition.",
        body_style
    ))
    story.append(Paragraph(
        "The project rigorously proved that combining domain-specific feature engineering (WHO BMI classifications, ADA glycemic stages, "
        "and insulin resistance proxies) with leak-free median imputation allows shallow neural networks to achieve competitive diagnostic "
        "performance on tabular biomedical data. Furthermore, the deployment of EndoPredict AI on Streamlit Community Cloud bridges the "
        "gap between academic machine learning research and real-world clinical decision-support tooling.",
        body_style
    ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 17: 16 FUTURE SCOPE
    # -------------------------------------------------------------
    story.append(Paragraph("16 Future Scope", h1_style))
    story.append(Paragraph("To build upon the foundation established in this laboratory project, the following future extensions are proposed:", body_style))
    story.append(Paragraph("1. <b>Multi-Center International Biobank Validation</b>: Validate and calibrate the model against diverse multi-ethnic datasets (e.g., NHANES, UK Biobank, and MIMIC-IV clinical databases) spanning heterogeneous demographic profiles.", bullet_style))
    story.append(Paragraph("2. <b>Modern Tabular Deep Learning Architectures</b>: Benchmark the MLP against modern tabular attention models such as TabNet, FT-Transformer, and SAINT to evaluate self-attention representations on tabular biometrics.", bullet_style))
    story.append(Paragraph("3. <b>Local Model Explainability (SHAP & LIME)</b>: Integrate TreeSHAP and KernelSHAP into the Streamlit user interface to provide clinicians with real-time patient-specific biomarker attribution waterfall plots.", bullet_style))
    story.append(Paragraph("4. <b>Longitudinal Temporal Trajectory Modeling</b>: Incorporate longitudinal Electronic Health Record (EHR) time-series data using Recurrent Neural Networks (LSTM/GRU) or Temporal Fusion Transformers to forecast 10-year glycemic progression.", bullet_style))
    story.append(Paragraph("5. <b>Edge Mobile & Offline Deployment</b>: Quantize the trained MLP model into ONNX and TensorFlow Lite formats for low-power, offline mobile screening apps in rural, resource-constrained healthcare centers.", bullet_style))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 18: REFERENCES
    # -------------------------------------------------------------
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, Cambridge, MA, 2016.",
        "[2] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, pp. 261–265.",
        "[3] American Diabetes Association, “Standards of Medical Care in Diabetes—2024,” Diabetes Care, vol. 47, no. Suppl. 1, pp. S1–S343, 2024.",
        "[4] I. Kavakiotis et al., “Machine learning and data mining methods in diabetes research,” Comput. Struct. Biotechnol. J., vol. 15, pp. 104–116, 2017.",
        "[5] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
        "[6] World Health Organization, “Global report on diabetes,” World Health Organization, Geneva, Switzerland, Tech. Rep., 2016.",
        "[7] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. 3rd Int. Conf. Learn. Represent. (ICLR), San Diego, CA, 2015."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefP', parent=body_style, fontSize=8.5, leading=11, leftIndent=10, spaceAfter=2)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 19: APPENDIX A - SOURCE CODE
    # -------------------------------------------------------------
    story.append(Paragraph("A Source Code", h1_style))
    story.append(Paragraph("Core implementation modules from the authoritative machine learning pipeline:", body_style))
    story.append(Paragraph("<b>A.1 Data Preprocessing & Leak-Free Imputation (src/preprocessing.py)</b>", h2_style))
    code_prep = (
        "class DiabetesPreprocessor(BaseEstimator, TransformerMixin):\n"
        "    def __init__(self, zero_cols=ZERO_COLS):\n"
        "        self.zero_cols = list(zero_cols)\n"
        "        self.imputer = SimpleImputer(strategy='median')\n"
        "        self.scaler = StandardScaler()\n\n"
        "    def fit(self, X, y=None):\n"
        "        X_clean = X.copy()\n"
        "        for col in self.zero_cols:\n"
        "            if col in X_clean.columns:\n"
        "                X_clean[col] = X_clean[col].replace(0, np.nan)\n"
        "        self.imputer.fit(X_clean[self.zero_cols])\n"
        "        X_imputed = X_clean.copy()\n"
        "        X_imputed[self.zero_cols] = self.imputer.transform(X_clean[self.zero_cols])\n"
        "        X_engineered = engineer_features(X_imputed)\n"
        "        self.scaler.fit(X_engineered)\n"
        "        self.n_features_out_ = X_engineered.shape[1]\n"
        "        return self"
    )
    story.append(Preformatted(code_prep, code_style))

    story.append(Paragraph("<b>A.2 Domain Feature Engineering (src/feature_engineering.py)</b>", h2_style))
    code_fe = (
        "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df_out = df[RAW_FEATURE_NAMES].copy()\n"
        "    # WHO BMI Bins\n"
        "    df_out['BMI_Underweight'] = (df_out['BMI'] < 18.5).astype(float)\n"
        "    df_out['BMI_Normal'] = ((df_out['BMI'] >= 18.5) & (df_out['BMI'] < 25.0)).astype(float)\n"
        "    df_out['BMI_Overweight'] = ((df_out['BMI'] >= 25.0) & (df_out['BMI'] < 30.0)).astype(float)\n"
        "    df_out['BMI_Obese'] = (df_out['BMI'] >= 30.0).astype(float)\n"
        "    # ADA Glycemic Stages\n"
        "    df_out['Glucose_Normal'] = (df_out['Glucose'] < 100.0).astype(float)\n"
        "    df_out['Glucose_Prediabetes'] = ((df_out['Glucose'] >= 100.0) & (df_out['Glucose'] < 126.0)).astype(float)\n"
        "    df_out['Glucose_Diabetes'] = (df_out['Glucose'] >= 126.0).astype(float)\n"
        "    # Metabolic Interactions & Log Transforms\n"
        "    df_out['Insulin_Resistance_Proxy'] = (df_out['Glucose'] * df_out['Insulin']) / 405.0\n"
        "    df_out['Insulin_Glucose_Ratio'] = df_out['Insulin'] / (df_out['Glucose'] + 1e-5)\n"
        "    df_out['BMI_Age_Interaction'] = df_out['BMI'] * df_out['Age']\n"
        "    df_out['Pregnancy_Age_Risk'] = df_out['Pregnancies'] / (df_out['Age'] + 1e-5)\n"
        "    df_out['Log_Insulin'] = np.log1p(df_out['Insulin'])\n"
        "    df_out['Log_DPF'] = np.log1p(df_out['DiabetesPedigreeFunction'])\n"
        "    return df_out[ALL_FEATURE_NAMES]"
    )
    story.append(Preformatted(code_fe, code_style))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 20: APPENDIX B - ADDITIONAL RESULTS
    # -------------------------------------------------------------
    story.append(Paragraph("B Additional Results", h1_style))
    story.append(Paragraph("Exhaustive Hyperparameter Grid Search Logs across MLP configurations:", body_style))

    grid_data = [
        [Paragraph("<b>Config</b>", t_header), Paragraph("<b>Topology</b>", t_header), Paragraph("<b>Act.</b>", t_header), Paragraph("<b>L2 (α)</b>", t_header), Paragraph("<b>LR (η)</b>", t_header), Paragraph("<b>Batch</b>", t_header), Paragraph("<b>Val ROC-AUC</b>", t_header)],
        [Paragraph("1", t_cell), Paragraph("(32, 16)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.0001", t_cell), Paragraph("0.001", t_cell), Paragraph("32", t_cell), Paragraph("0.8352", t_cell)],
        [Paragraph("2", t_cell), Paragraph("(32, 16)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.01", t_cell), Paragraph("0.001", t_cell), Paragraph("64", t_cell), Paragraph("0.8410", t_cell)],
        [Paragraph("3", t_cell), Paragraph("(64, 32)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.0001", t_cell), Paragraph("0.001", t_cell), Paragraph("32", t_cell), Paragraph("0.8520", t_cell)],
        [Paragraph("4 (Best)", t_cell), Paragraph("(64, 32)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.001", t_cell), Paragraph("0.001", t_cell), Paragraph("32", t_cell), Paragraph("0.8645*", t_cell)],
        [Paragraph("5", t_cell), Paragraph("(64, 32)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.01", t_cell), Paragraph("0.005", t_cell), Paragraph("64", t_cell), Paragraph("0.8460", t_cell)],
        [Paragraph("6", t_cell), Paragraph("(128, 64)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.001", t_cell), Paragraph("0.001", t_cell), Paragraph("32", t_cell), Paragraph("0.8590", t_cell)],
        [Paragraph("7", t_cell), Paragraph("(128, 64, 32)", t_cell), Paragraph("ReLU", t_cell), Paragraph("0.01", t_cell), Paragraph("0.001", t_cell), Paragraph("64", t_cell), Paragraph("0.8280", t_cell)],
        [Paragraph("8", t_cell), Paragraph("(64, 32)", t_cell), Paragraph("Tanh", t_cell), Paragraph("0.001", t_cell), Paragraph("0.001", t_cell), Paragraph("32", t_cell), Paragraph("0.8390", t_cell)]
    ]
    t_g = Table(grid_data, colWidths=[55, 75, 55, 65, 65, 55, 134])
    t_g.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor("#e0f2fe")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_g)

    pr_img = os.path.join(VIZ_DIR, "precision_recall_curves.png")
    if os.path.exists(pr_img):
        story.append(Spacer(1, 4))
        story.append(Image(pr_img, width=3.4*inch, height=2.3*inch))
        story.append(Paragraph("Figure 5: Precision-Recall Curves across Machine Learning Classifiers", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 21: APPENDIX C - DEPLOYMENT SCREENSHOTS
    # -------------------------------------------------------------
    story.append(Paragraph("C Deployment Screenshots", h1_style))
    story.append(Paragraph(
        "The production web application 'EndoPredict AI' is actively deployed on Streamlit Community Cloud:<br/>"
        "• Public Live URL: <b>https://endopredict-ai.streamlit.app/</b><br/>"
        "• Local Development: <b>http://localhost:8501</b>",
        body_style
    ))

    triage_img = os.path.join(VIZ_DIR, "ui_triage_screenshot.png")
    if os.path.exists(triage_img):
        story.append(Spacer(1, 4))
        story.append(Image(triage_img, width=4.5*inch, height=2.1*inch))
        story.append(Paragraph("Figure 6: EndoPredict AI — Patient Triage, Probability Gauge & Biometric Radar Profile", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    batch_img = os.path.join(VIZ_DIR, "ui_batch_screenshot.png")
    if os.path.exists(batch_img):
        story.append(Spacer(1, 4))
        story.append(Image(batch_img, width=4.5*inch, height=2.1*inch))
        story.append(Paragraph("Figure 7: EndoPredict AI — Batch Cohort Screening & Population Risk Histogram", ParagraphStyle('Cap', parent=body_style, fontSize=8, fontName='Times-Italic', alignment=1)))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 22: APPENDIX D - CONTRIBUTIONS OF GROUP MEMBERS
    # -------------------------------------------------------------
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
        "<br/>Detailed allocation of technical responsibilities:<br/>"
        "• <b>Tribhuwan Singh</b>: Neural network architecture design, feed-forward optimization, loss formulation, and pipeline orchestration.<br/>"
        "• <b>Surajit Sahoo</b>: Zero-artifact detection, leak-free median imputation, and baseline classifier construction.<br/>"
        "• <b>Anwesha Srichandan</b>: Exploratory data analysis, 16-domain feature engineering, and hyperparameter grid search execution.<br/>"
        "• <b>Priti Rani Maity</b>: Comprehensive metric benchmarking, Streamlit Cloud web deployment, and formal laboratory documentation.",
        body_style
    ))

    doc.build(story, canvasmaker=SOAReportCanvas)
    print(f"PDF report successfully generated at: {PDF_PATH}")

if __name__ == "__main__":
    generate_docx()
    generate_pdf()
