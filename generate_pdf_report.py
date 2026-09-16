"""
PDF Report Generator for CSE 4192: Machine Learning Projects with Python.
Generates a publication-quality research report PDF using ReportLab with embedded charts, tables, and typography.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "report.pdf")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count and headers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "CSE 4192: Machine Learning Projects with Python — Laboratory Assignment 01")
            self.drawRightString(612 - 54, 755, "Multilayer Perceptron Diabetes Prediction")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 612 - 54, 748)
            
        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Department of Computer Science & Engineering • Centre for AI & ML")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.restoreState()

def build_pdf_report():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")    # Deep Slate
    c_accent = colors.HexColor("#0284c7")     # Ocean Blue
    c_sub = colors.HexColor("#475569")        # Muted Slate
    c_body = colors.HexColor("#1e293b")       # Dark Charcoal
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=0,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceAfter=14
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_body,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_body,
        leftIndent=12,
        spaceAfter=3
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=c_body
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.white
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=c_sub
    )

    story = []

    # -------------------------------------------------------------
    # Cover / Header Section
    # -------------------------------------------------------------
    story.append(Paragraph("CSE 4192: Machine Learning Projects with Python", subtitle_style))
    story.append(Paragraph("Diabetes Onset Prediction and Risk Stratification Using an Optimized Multilayer Perceptron (MLP) and Conventional Machine Learning Baselines", title_style))
    story.append(Paragraph("<b>Course Faculty:</b> Dr. Gyana Ranjan Patra &nbsp;|&nbsp; <b>Institution:</b> Centre for AI & ML / Dept. of CSE &nbsp;|&nbsp; <b>Term:</b> 2026", callout_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceAfter=10))

    # -------------------------------------------------------------
    # Team Members Table
    # -------------------------------------------------------------
    story.append(Paragraph("<b>Project Investigators & Contributors:</b>", h2_style))
    authors_data = [
        [Paragraph("<b>Sl. No.</b>", table_header), Paragraph("<b>Student Name</b>", table_header), Paragraph("<b>Registration Number</b>", table_header), Paragraph("<b>Department / Role</b>", table_header)],
        [Paragraph("1", table_cell), Paragraph("<b>Tribhuwan Singh</b>", table_cell), Paragraph("2341019538", table_cell), Paragraph("CSE / ML Lead & Pipeline Architect", table_cell)],
        [Paragraph("2", table_cell), Paragraph("<b>Surajit Sahoo</b>", table_cell), Paragraph("2341019165", table_cell), Paragraph("CSE / Preprocessing & Imputation", table_cell)],
        [Paragraph("3", table_cell), Paragraph("<b>Anwesha Srichandan</b>", table_cell), Paragraph("2341019594", table_cell), Paragraph("CSE / Feature Engineering & EDA", table_cell)],
        [Paragraph("4", table_cell), Paragraph("<b>Priti Rani Maity</b>", table_cell), Paragraph("2341013065", table_cell), Paragraph("CSE / Evaluation & Benchmark Analytics", table_cell)]
    ]
    t_authors = Table(authors_data, colWidths=[40, 140, 110, 214])
    t_authors.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_authors)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        "Diabetes mellitus represents one of the most pressing global metabolic health crises. Early diagnosis is vital "
        "to prevent severe secondary microvascular and macrovascular complications. This research project develops an "
        "end-to-end, data-leakage-free machine learning system to predict diabetes onset from physiological and demographic "
        "markers using the Pima Indians Diabetes Database (N = 768). The workflow resolves biologically implausible zero values "
        "in physiological attributes (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) via training-fitted median imputation, "
        "expands raw features into a 24-dimensional domain representation, and benchmarks seven conventional classification baselines "
        "against an optimized Scikit-Learn Multilayer Perceptron (MLPClassifier). The final tuned MLP architecture (24 -> 64 -> 32 -> 1, "
        "ReLU activation, Adam optimizer, alpha=0.001, batch size=32) achieved an untouched test set Accuracy of 82.76%, Sensitivity of 75.00%, "
        "Specificity of 86.84%, F1-Score of 75.00%, and ROC-AUC of 0.8681, successfully demonstrating neural pattern extraction on tabular biometric data."
    )
    story.append(Paragraph(summary_text, body_style))

    # -------------------------------------------------------------
    # 2. Motivation & 3. Problem Statement
    # -------------------------------------------------------------
    story.append(Paragraph("2. Motivation & Clinical Importance", h1_style))
    story.append(Paragraph(
        "<b>2.1 Background:</b> Over 537 million adults globally live with diabetes, a figure projected to reach 783 million by 2045. "
        "Early screening is crucial because clinical onset often remains asymptomatic until irreversible metabolic damage occurs.<br/>"
        "<b>2.2 Limitations of Existing Tools:</b> Traditional clinical risk calculators use static univariate cutoffs that fail to capture "
        "complex non-linear interactions between adiposity, insulin resistance, and pancreatic beta-cell compensation.<br/>"
        "<b>2.3 Need for Research:</b> Many existing ML studies suffer from severe data leakage (fitting imputers/scalers globally prior to splitting) "
        "and mishandle physiological zero readings as true numerical values. This research establishes a leak-free, reproducible neural pipeline.",
        body_style
    ))

    story.append(Paragraph("3. Problem Statement & Research Gap", h1_style))
    gap_table_data = [
        [Paragraph("<b>Current Situation</b>", table_header), Paragraph("High global diabetes prevalence; static univariate cutoffs miss complex metabolic interactions.", table_cell)],
        [Paragraph("<b>Research Gap</b>", table_header), Paragraph("Prior ML studies often exhibit data leakage, ignore zero-value biological artifacts, and lack multi-algorithm benchmarks against neural MLPs.", table_cell)],
        [Paragraph("<b>Proposed Solution</b>", table_header), Paragraph("An authoritative leak-free pipeline with median zero imputation, 24-dim domain engineering, systematic MLP grid search, 8-model benchmark, and cloud deployment.", table_cell)]
    ]
    t_gap = Table(gap_table_data, colWidths=[110, 394])
    t_gap.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), c_accent),
        ('BACKGROUND', (1,0), (1,-1), colors.HexColor("#f1f5f9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_gap)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 4. Research Objectives & Chapter Mapping
    # -------------------------------------------------------------
    story.append(Paragraph("4. Research Objectives (SMART) & Chapter Mapping", h1_style))
    story.append(Paragraph("The research objectives follow the SMART criteria using precise action verbs:", body_style))
    obj_data = [
        [Paragraph("<b>Objective</b>", table_header), Paragraph("<b>Action Verb</b>", table_header), Paragraph("<b>Description</b>", table_header), Paragraph("<b>Chapter</b>", table_header)],
        [Paragraph("O1", table_cell), Paragraph("Analyze", table_cell), Paragraph("Conduct rigorous EDA on feature distributions, skewness, and class imbalance (65.1% vs 34.9%).", table_cell), Paragraph("Section 6", table_cell)],
        [Paragraph("O2", table_cell), Paragraph("Implement", table_cell), Paragraph("Build leak-free zero-detection and training-fitted median imputation pipeline.", table_cell), Paragraph("Section 7", table_cell)],
        [Paragraph("O3", table_cell), Paragraph("Design", table_cell), Paragraph("Formulate 16 domain features (WHO BMI bins, ADA stages, metabolic ratios) -> 24 features.", table_cell), Paragraph("Section 8", table_cell)],
        [Paragraph("O4", table_cell), Paragraph("Develop", table_cell), Paragraph("Implement 6 baseline classifiers (LogReg, KNN, SVM, DT, RF, GBDT) & baseline MLP.", table_cell), Paragraph("Section 9", table_cell)],
        [Paragraph("O5", table_cell), Paragraph("Optimize", table_cell), Paragraph("Execute systematic Grid Search over MLP architectures, L2 penalties, and learning rates.", table_cell), Paragraph("Section 10", table_cell)],
        [Paragraph("O6", table_cell), Paragraph("Evaluate", table_cell), Paragraph("Benchmark all 8 models on untouched test partition (N=116) across 6 evaluation metrics.", table_cell), Paragraph("Section 11", table_cell)],
        [Paragraph("O7", table_cell), Paragraph("Deploy", table_cell), Paragraph("Construct and deploy interactive Streamlit web dashboard with automated verification tests.", table_cell), Paragraph("Section 12", table_cell)]
    ]
    t_obj = Table(obj_data, colWidths=[30, 60, 334, 80])
    t_obj.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_obj)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 5. Literature Survey
    # -------------------------------------------------------------
    story.append(Paragraph("5. Literature Survey", h1_style))
    lit_data = [
        [Paragraph("<b>Author & Year</b>", table_header), Paragraph("<b>Methodology</b>", table_header), Paragraph("<b>Dataset</b>", table_header), Paragraph("<b>Reported Metric</b>", table_header), Paragraph("<b>Identified Limitations</b>", table_header)],
        [Paragraph("Smith et al. (1988)", table_cell), Paragraph("ADAP Perceptron", table_cell), Paragraph("Pima Indians (768)", table_cell), Paragraph("Acc: 76.0%", table_cell), Paragraph("Early neural model; no systematic missing value imputation; high false negatives.", table_cell)],
        [Paragraph("Kavakiotis et al. (2017)", table_cell), Paragraph("Systematic Survey", table_cell), Paragraph("Clinical Databases", table_cell), Paragraph("Acc: 70% - 82%", table_cell), Paragraph("Identified pervasive data leakage and unstandardized evaluation protocols.", table_cell)],
        [Paragraph("Sisodia et al. (2018)", table_cell), Paragraph("DT, SVM, Naive Bayes", table_cell), Paragraph("Pima Indians (768)", table_cell), Paragraph("NB: 76.3%", table_cell), Paragraph("Treated zero values as valid; omitted neural multi-layer architectures.", table_cell)],
        [Paragraph("Zou et al. (2018)", table_cell), Paragraph("DT, Random Forest", table_cell), Paragraph("Hospital Exams", table_cell), Paragraph("RF: 80.84%", table_cell), Paragraph("Evaluated only raw features without metabolic interaction terms.", table_cell)],
        [Paragraph("Patil et al. (2021)", table_cell), Paragraph("MLP & Ensembles", table_cell), Paragraph("Pima Indians (768)", table_cell), Paragraph("MLP: 79.1%", table_cell), Paragraph("Omitted L2 regularization grid search; static decision threshold.", table_cell)]
    ]
    t_lit = Table(lit_data, colWidths=[80, 85, 80, 65, 194])
    t_lit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_lit)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 6. Dataset Description & Preprocessing
    # -------------------------------------------------------------
    story.append(Paragraph("6. Dataset Description & Leak-Free Preprocessing", h1_style))
    story.append(Paragraph(
        "<b>6.1 Pima Indians Diabetes Database:</b> Comprises N = 768 female records with 8 raw attributes and 1 binary target "
        "(Class 0: 500 [65.1%], Class 1: 268 [34.9%]). Biologically impossible zeros in continuous variables were identified: "
        "Glucose (5 zeros), Blood Pressure (35 zeros), Skin Thickness (227 zeros), Insulin (374 zeros), BMI (11 zeros).<br/>"
        "<b>6.2 Preprocessing Workflow:</b> All imputation medians and scaling parameters (mean, standard deviation) were fitted strictly "
        "on the training set (N = 537) to prevent data leakage.",
        body_style
    ))
    
    # Preprocessing Flow Diagram Table
    flow_data = [
        [Paragraph("<b>Step</b>", table_header), Paragraph("<b>Pipeline Stage</b>", table_header), Paragraph("<b>Operation / Formula</b>", table_header), Paragraph("<b>Leakage Prevention Principle</b>", table_header)],
        [Paragraph("1", table_cell), Paragraph("Zero Detection", table_cell), Paragraph("Mask zeros in Glucose, BP, Skin, Insulin, BMI -> NaN", table_cell), Paragraph("Independent sample-level masking", table_cell)],
        [Paragraph("2", table_cell), Paragraph("Median Imputation", table_cell), Paragraph("x_imputed = Median_train if NaN else x", table_cell), Paragraph("Medians computed strictly on D_train", table_cell)],
        [Paragraph("3", table_cell), Paragraph("Feature Engineering", table_cell), Paragraph("Compute 16 domain features on clean imputed values", table_cell), Paragraph("Deterministic deterministic transforms", table_cell)],
        [Paragraph("4", table_cell), Paragraph("Standardization", table_cell), Paragraph("z = (x - mu_train) / sigma_train", table_cell), Paragraph("Fitted StandardScaler applied to Val/Test", table_cell)]
    ]
    t_flow = Table(flow_data, colWidths=[30, 95, 205, 174])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_flow)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 7. Domain-Specific Feature Engineering (24 Dimensions)
    # -------------------------------------------------------------
    story.append(Paragraph("7. Domain-Specific Feature Engineering (24 Features)", h1_style))
    eng_data = [
        [Paragraph("<b>Feature Category</b>", table_header), Paragraph("<b>Engineered Features (16 Total)</b>", table_header), Paragraph("<b>Mathematical Formula / Logic</b>", table_header)],
        [Paragraph("WHO Adiposity Bins", table_cell), Paragraph("BMI_Underweight, BMI_Normal, BMI_Overweight, BMI_Obese", table_cell), Paragraph("1-hot indicators for BMI: <18.5, 18.5-24.9, 25-29.9, >=30.0", table_cell)],
        [Paragraph("ADA Glycemic Stages", table_cell), Paragraph("Glucose_Normal, Glucose_Prediabetes, Glucose_Diabetes", table_cell), Paragraph("1-hot indicators for Glucose: <100, 100-125, >=126 mg/dL", table_cell)],
        [Paragraph("Age Cohorts", table_cell), Paragraph("Age_Young, Age_Middle, Age_Senior", table_cell), Paragraph("1-hot indicators for Age: <30, 30-50, >50 years", table_cell)],
        [Paragraph("Metabolic Interaction", table_cell), Paragraph("Insulin_Resistance_Proxy", table_cell), Paragraph("(Glucose * Insulin) / 405.0", table_cell)],
        [Paragraph("Metabolic Ratio", table_cell), Paragraph("Insulin_Glucose_Ratio", table_cell), Paragraph("Insulin / (Glucose + 1e-5)", table_cell)],
        [Paragraph("Synergistic Risk", table_cell), Paragraph("BMI_Age_Interaction", table_cell), Paragraph("BMI * Age", table_cell)],
        [Paragraph("Gestational Risk", table_cell), Paragraph("Pregnancy_Age_Risk", table_cell), Paragraph("Pregnancies / (Age + 1e-5)", table_cell)],
        [Paragraph("Log Transforms", table_cell), Paragraph("Log_Insulin, Log_DPF", table_cell), Paragraph("ln(1 + Insulin), ln(1 + DPF) (reduces skewness)", table_cell)]
    ]
    t_eng = Table(eng_data, colWidths=[110, 174, 220])
    t_eng.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_eng)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 8. Model Architecture & Mathematical Formulation
    # -------------------------------------------------------------
    story.append(Paragraph("8. Model Architecture & Mathematical Formulation", h1_style))
    story.append(Paragraph(
        "<b>8.1 Network Topology:</b> The authoritative production model is a Feed-Forward Multilayer Perceptron "
        "(Scikit-Learn MLPClassifier) structured as: <b>Input (24) -> Hidden Layer 1 (64) -> Hidden Layer 2 (32) -> Output (1)</b>.<br/>"
        "<b>8.2 Trainable Parameters:</b><br/>"
        "• Layer 1 (24 -> 64): 24 * 64 + 64 = 1,600 parameters<br/>"
        "• Layer 2 (64 -> 32): 64 * 32 + 32 = 2,080 parameters<br/>"
        "• Output Layer (32 -> 1): 32 * 1 + 1 = 33 parameters<br/>"
        "• <b>Total Trainable Parameters = 3,713 weights and biases.</b><br/>"
        "<b>8.3 Objective Function:</b> Binary cross-entropy with L2 weight decay regularization (alpha = 0.001):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>L(W, b) = -1/N * sum[ y*ln(y_hat) + (1-y)*ln(1-y_hat) ] + (alpha / 2N) * sum ||W_l||^2</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # 9. Results & Multi-Model Benchmark
    # -------------------------------------------------------------
    story.append(Paragraph("9. Experimental Results & Benchmark Performance (N = 116)", h1_style))
    story.append(Paragraph("Performance evaluated on the untouched test partition ($N = 116$, 76 Negative, 40 Positive cases):", body_style))
    
    results_data = [
        [Paragraph("<b>Algorithm</b>", table_header), Paragraph("<b>Accuracy</b>", table_header), Paragraph("<b>Sensitivity</b>", table_header), Paragraph("<b>Specificity</b>", table_header), Paragraph("<b>Precision</b>", table_header), Paragraph("<b>F1-Score</b>", table_header), Paragraph("<b>ROC-AUC</b>", table_header), Paragraph("<b>FN</b>", table_header), Paragraph("<b>FP</b>", table_header)],
        [Paragraph("Logistic Regression", table_cell), Paragraph("79.31%", table_cell), Paragraph("72.50%", table_cell), Paragraph("82.89%", table_cell), Paragraph("0.6905", table_cell), Paragraph("0.7073", table_cell), Paragraph("0.8444", table_cell), Paragraph("11", table_cell), Paragraph("13", table_cell)],
        [Paragraph("K-Nearest Neighbors", table_cell), Paragraph("81.90%", table_cell), Paragraph("77.50%", table_cell), Paragraph("84.21%", table_cell), Paragraph("0.7209", table_cell), Paragraph("0.7470", table_cell), Paragraph("0.8954", table_cell), Paragraph("9", table_cell), Paragraph("12", table_cell)],
        [Paragraph("Support Vector Machine", table_cell), Paragraph("83.62%", table_cell), Paragraph("77.50%", table_cell), Paragraph("86.84%", table_cell), Paragraph("0.7561", table_cell), Paragraph("0.7654", table_cell), Paragraph("0.8808", table_cell), Paragraph("9", table_cell), Paragraph("10", table_cell)],
        [Paragraph("Decision Tree", table_cell), Paragraph("87.93%", table_cell), Paragraph("82.50%", table_cell), Paragraph("90.79%", table_cell), Paragraph("0.8250", table_cell), Paragraph("0.8250", table_cell), Paragraph("0.9021", table_cell), Paragraph("7", table_cell), Paragraph("7", table_cell)],
        [Paragraph("Random Forest", table_cell), Paragraph("88.79%", table_cell), Paragraph("87.50%", table_cell), Paragraph("89.47%", table_cell), Paragraph("0.8140", table_cell), Paragraph("0.8434", table_cell), Paragraph("0.9431", table_cell), Paragraph("5", table_cell), Paragraph("8", table_cell)],
        [Paragraph("Gradient Boosting", table_cell), Paragraph("89.66%", table_cell), Paragraph("85.00%", table_cell), Paragraph("92.11%", table_cell), Paragraph("0.8500", table_cell), Paragraph("0.8500", table_cell), Paragraph("0.9579", table_cell), Paragraph("6", table_cell), Paragraph("6", table_cell)],
        [Paragraph("MLP (Baseline)", table_cell), Paragraph("81.03%", table_cell), Paragraph("65.00%", table_cell), Paragraph("89.47%", table_cell), Paragraph("0.7647", table_cell), Paragraph("0.7027", table_cell), Paragraph("0.8625", table_cell), Paragraph("14", table_cell), Paragraph("8", table_cell)],
        [Paragraph("<b>MLP (Optimized)</b>", table_cell), Paragraph("<b>82.76%</b>", table_cell), Paragraph("<b>75.00%</b>", table_cell), Paragraph("<b>86.84%</b>", table_cell), Paragraph("<b>0.7500</b>", table_cell), Paragraph("<b>0.7500</b>", table_cell), Paragraph("<b>0.8681</b>", table_cell), Paragraph("<b>10</b>", table_cell), Paragraph("<b>10</b>", table_cell)]
    ]
    t_res = Table(results_data, colWidths=[104, 52, 54, 54, 48, 48, 50, 24, 24])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor("#f8fafc"), colors.white]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#e0f2fe")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # Embedded Visualizations
    # -------------------------------------------------------------
    roc_img = os.path.join(VIZ_DIR, "roc_curves.png")
    cm_img = os.path.join(VIZ_DIR, "confusion_matrices.png")
    
    if os.path.exists(roc_img) and os.path.exists(cm_img):
        story.append(Paragraph("10. Multi-Model Visual Diagnostics", h1_style))
        viz_table_data = [
            [Image(roc_img, width=3.4*inch, height=2.4*inch), Image(cm_img, width=3.4*inch, height=2.4*inch)],
            [Paragraph("<b>Figure 1:</b> Multi-Model ROC Benchmark Curves", callout_style), Paragraph("<b>Figure 2:</b> Confusion Matrices across Classifiers", callout_style)]
        ]
        t_viz = Table(viz_table_data, colWidths=[252, 252])
        t_viz.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_viz)
        story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 11. Discussion of Results & Clinical Trade-offs
    # -------------------------------------------------------------
    story.append(Paragraph("11. Discussion of Results & Clinical Trade-offs", h1_style))
    story.append(Paragraph(
        "<b>11.1 Non-Linear Pattern Learning:</b> Hyperparameter tuning improved the MLP's sensitivity from 65.00% to 75.00% "
        "while reducing missed cases (False Negatives) from 14 down to 10. The L2 penalty (alpha=0.001) effectively constrained "
        "weight norms, avoiding over-fitting on small sample sizes.<br/>"
        "<b>11.2 Decision Threshold Calibration:</b> In diabetes screening, False Negatives present substantial clinical hazards. "
        "Because the MLP generates smooth continuous probabilities, the decision threshold can be calibrated dynamically: "
        "lowering the threshold to tau=0.35 in the Streamlit application reduces False Negatives from 10 to 4, optimizing early intervention.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # 12. Conclusion & Future Scope
    # -------------------------------------------------------------
    story.append(Paragraph("12. Conclusion & Future Scope", h1_style))
    story.append(Paragraph(
        "<b>12.1 Conclusion:</b> A robust, leak-free machine learning pipeline was established for diabetes onset prediction. "
        "The optimized Scikit-Learn Multilayer Perceptron demonstrated solid performance (82.76% Accuracy, 0.8681 ROC-AUC) "
        "on untouched test data. An enterprise Streamlit Cloud web dashboard was successfully created and deployed.<br/>"
        "<b>12.2 Future Scope:</b><br/>"
        "• Validate on multi-center international clinical cohorts (e.g., NHANES, UK Biobank).<br/>"
        "• Benchmark against tabular attention models (TabNet, FT-Transformer).<br/>"
        "• Integrate explainable AI attribution (SHAP values, Integrated Gradients).<br/>"
        "• Export quantized ONNX models for low-power edge mobile screening devices.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # 13. References
    # -------------------------------------------------------------
    story.append(Paragraph("13. Key References", h1_style))
    refs = [
        "1. Smith, J. W., et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. <i>Proc. Symp. Comput. Appl. Med. Care</i>, 261–265.",
        "2. American Diabetes Association. (2024). Standards of Medical Care in Diabetes—2024. <i>Diabetes Care</i>, 47(Suppl. 1), S1–S343.",
        "3. Kavakiotis, I., et al. (2017). Machine learning and data mining methods in diabetes research. <i>Comput. Struct. Biotechnol. J.</i>, 15, 104–116.",
        "4. Sisodia, D., & Sisodia, D. S. (2018). Prediction of diabetes using classification algorithms. <i>Procedia Computer Science</i>, 132, 1578–1585.",
        "5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', parent=body_style, fontSize=7.5, leading=10, leftIndent=8, spaceAfter=2)))

    # Build Document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF successfully compiled to: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf_report()
