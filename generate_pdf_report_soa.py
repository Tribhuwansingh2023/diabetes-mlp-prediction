"""
Generates the official Laboratory Record document (.pdf) according to the
Siksha 'O' Anusandhan (Deemed to be University) template for CSE 4192: Machine Learning Projects with Python.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Laboratory_Record_CSE4192.pdf")
LOGO_PATH = os.path.join(BASE_DIR, "soa_logo.png")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

class SOALabCanvas(canvas.Canvas):
    """Custom canvas handling SOA University running header ('Diabetes Onset Prediction / Page X') and footers."""
    def __init__(self, *args, **kwargs):
        super(SOALabCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_soa_decorations(num_pages)
            super(SOALabCanvas, self).showPage()
        super(SOALabCanvas, self).save()

    def draw_soa_decorations(self, page_count):
        self.saveState()
        
        # Header (Pages >= 2) - matching university template "Write title here [page_number]"
        if self._pageNumber > 1:
            self.setFont("Times-Roman", 10)
            self.setFillColor(colors.black)
            self.drawString(54, 745, "Diabetes Onset Prediction with Multilayer Perceptron")
            self.drawRightString(612 - 54, 745, str(self._pageNumber - 1))
            self.setStrokeColor(colors.HexColor("#000000"))
            self.setLineWidth(0.5)
            self.line(54, 738, 612 - 54, 738)
            
        self.restoreState()

def build_soa_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Times New Roman Typography Palette
    cov_uni = ParagraphStyle(
        'CovUni',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=19,
        alignment=1, # Center
        textColor=colors.black,
        spaceAfter=3
    )
    
    cov_meta = ParagraphStyle(
        'CovMeta',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=14,
        alignment=0,
        textColor=colors.black
    )
    
    cov_title = ParagraphStyle(
        'CovTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=14,
        leading=18,
        alignment=1,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=4
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        leftIndent=15,
        spaceAfter=2
    )
    
    t_header = ParagraphStyle(
        'THeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=0
    )
    
    t_cell = ParagraphStyle(
        'TCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8,
        leading=10.5,
        textColor=colors.black,
        alignment=0
    )

    story = []

    # =============================================================
    # COVER PAGE (Exact SOA University Template)
    # =============================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("SIKSHA ‘O’ ANUSANDHAN", cov_uni))
    story.append(Paragraph("(DEEMED TO BE UNIVERSITY)", ParagraphStyle('CovSub', parent=cov_uni, fontSize=12, leading=15)))
    story.append(Spacer(1, 20))
    
    # Admission Batch & Session Line
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
    
    # SOA Logo
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

    # =============================================================
    # BODY CONTENT (Sections 1 to 16 & Appendices)
    # =============================================================
    
    # 1 Introduction
    story.append(Paragraph("1 Introduction", h1_style))
    story.append(Paragraph("1.1 Problem Statement", h2_style))
    story.append(Paragraph(
        "Diabetes mellitus is a progressive chronic metabolic disease marked by persistent hyperglycemia due to insulin deficiency "
        "or resistance. Early detection is crucial to prevent microvascular and macrovascular complications. Traditional clinical "
        "cutoffs fail to capture non-linear metabolic risk dynamics, while existing machine learning pipelines frequently suffer "
        "from data leakage and biological zero-value errors. This project establishes an authoritative, leak-free Multilayer Perceptron "
        "(MLP) classification system to accurately predict diabetes onset from patient biometric parameters.",
        body_style
    ))
    story.append(Paragraph("1.2 Aim", h2_style))
    story.append(Paragraph("To develop, optimize, benchmark, and deploy a data-leakage-free Multilayer Perceptron (MLP) binary classifier for diabetes prediction using the Pima Indians Diabetes Database.", body_style))
    story.append(Paragraph("1.3 Objectives", h2_style))
    story.append(Paragraph("• Conduct rigorous Exploratory Data Analysis (EDA) investigating feature distributions, skewness, and class imbalance (65.1% negative vs. 34.9% positive).", bullet_style))
    story.append(Paragraph("• Implement a leak-free preprocessing pipeline that identifies biological zero artifacts and applies training-fitted median imputation.", bullet_style))
    story.append(Paragraph("• Formulate 16 domain-engineered features (expanding 8 raw attributes into 24 dimensions) capturing WHO BMI bins, ADA glucose stages, and metabolic ratios.", bullet_style))
    story.append(Paragraph("• Benchmark 7 baseline algorithms against an optimized Scikit-Learn Multilayer Perceptron (MLPClassifier).", bullet_style))
    story.append(Paragraph("• Deploy the production model into an interactive Streamlit Cloud web application.", bullet_style))

    # 2 Dataset Description
    story.append(Paragraph("2 Dataset Description", h1_style))
    story.append(Paragraph("2.1 Dataset Source", h2_style))
    story.append(Paragraph("The standard Pima Indians Diabetes Database (NIDDK) containing N = 768 female patient records.", body_style))
    story.append(Paragraph("2.2 Dataset Features", h2_style))
    story.append(Paragraph("8 raw physiological inputs: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.", body_style))
    story.append(Paragraph("2.3 Target Variable", h2_style))
    story.append(Paragraph("Outcome: Binary label (0 = Non-Diabetic [500 cases, 65.1%], 1 = Diabetic [268 cases, 34.9%]).", body_style))
    story.append(Paragraph("2.4 Dataset Statistics", h2_style))
    
    # Dataset Table
    ds_data = [
        [Paragraph("<b>Feature</b>", t_header), Paragraph("<b>Type</b>", t_header), Paragraph("<b>Mean ± SD</b>", t_header), Paragraph("<b>Min</b>", t_header), Paragraph("<b>Max</b>", t_header), Paragraph("<b>Zeros / Missing (%)</b>", t_header)],
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
    story.append(Spacer(1, 6))

    # 3 Exploratory Data Analysis & 4 Preprocessing
    story.append(Paragraph("3 Exploratory Data Analysis & 4 Data Preprocessing", h1_style))
    story.append(Paragraph(
        "<b>3.1–3.8 EDA Insights:</b> Insulin (skewness = 2.27) and DPF (skewness = 1.92) exhibit severe positive skewness. "
        "Diabetic individuals show significantly elevated median glucose (140 vs. 107 mg/dL) and BMI (34.3 vs. 30.1 kg/m^2).<br/>"
        "<b>4.1–4.7 Preprocessing:</b> Zeros in continuous features (Glucose, BP, Skin, Insulin, BMI) were replaced with NaN and imputed "
        "using training medians fitted on D_train (N=537): Glucose = 117.0, BP = 72.0, Skin = 29.0, Insulin = 125.0, BMI = 32.0. "
        "16 domain features were engineered (WHO BMI bins, ADA glucose stages, metabolic ratios, interaction terms, log transforms), "
        "expanding the feature representation to 24 dimensions, followed by StandardScaler normalization.",
        body_style
    ))

    # 5 Methodology & 6 Model Development
    story.append(Paragraph("5 Methodology & 6 Model Development", h1_style))
    story.append(Paragraph(
        "<b>5.1–5.4 Architecture:</b> Feed-forward Multilayer Perceptron structured as <b>Input (24) -> Hidden 1 (64, ReLU) -> Hidden 2 (32, ReLU) -> Output (1, Sigmoid)</b>.<br/>"
        "<b>Trainable Parameters:</b> Layer 1: 24*64 + 64 = 1,600; Layer 2: 64*32 + 32 = 2,080; Output: 32*1 + 1 = 33 -> <b>Total = 3,713 parameters</b>.<br/>"
        "<b>6.1–6.5 Baselines:</b> Logistic Regression, KNN (k=7), SVM (RBF), Decision Tree, Random Forest (100 trees), Gradient Boosting, and Baseline MLP.",
        body_style
    ))

    # 7 Model Evaluation & 8 Comparison
    story.append(Paragraph("7 Model Evaluation & 8 Model Comparison", h1_style))
    story.append(Paragraph("Evaluated on the untouched test partition ($N = 116$, 76 Negative, 40 Positive cases):", body_style))
    
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
    story.append(Spacer(1, 6))

    # 9 Hyperparameter Tuning to 12 Deployment
    story.append(Paragraph("9 Hyperparameter Tuning & 10–12 Deployment", h1_style))
    story.append(Paragraph(
        "<b>9.1–9.5 Tuning:</b> Grid Search optimized the MLP: Topology (64, 32), alpha = 0.001, lr = 0.001, batch size = 32 (Validation ROC-AUC = 0.8645).<br/>"
        "<b>10–11 Final Evaluation:</b> Retrained on Train+Val (N=652) and tested on untouched Test (N=116), yielding 82.76% Accuracy, 75.00% Recall, 86.84% Specificity.<br/>"
        "<b>12.1–12.5 Deployment:</b> Deployed to Streamlit Cloud with an interactive multi-tab interface supporting single-patient triage, counterfactual sensitivity simulation, and batch cohort screening. Verified via automated deployment test suite (7/7 tests passed in 3.2s).",
        body_style
    ))

    # 13–16 Discussion, Limitations, Conclusion & Scope
    story.append(Paragraph("13 Results and Discussion & 14–16 Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Discussion:</b> The MLP effectively maps complex metabolic interactions into smooth continuous probabilities. "
        "In clinical triage, False Negatives present high hazard. Dynamic threshold calibration (tau = 0.35) reduces False Negatives from 10 to 4.<br/>"
        "<b>Limitations:</b> Evaluated on Pima Indian female cohort (N=768); multi-ethnic external validation is warranted.<br/>"
        "<b>Future Scope:</b> Tabular Transformers (TabNet), explainable AI (SHAP), and edge mobile deployment.",
        body_style
    ))

    # References
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] J. W. Smith et al., “Using the ADAP learning algorithm to forecast diabetes onset,” Proc. Symp. Comput. Appl. Med. Care, pp. 261–265, 1988.",
        "[2] American Diabetes Association, “Standards of Care in Diabetes—2024,” Diabetes Care, vol. 47, no. Suppl. 1, pp. S1–S343, 2024.",
        "[3] I. Kavakiotis et al., “Machine learning in diabetes research,” Comput. Struct. Biotechnol. J., vol. 15, pp. 104–116, 2017.",
        "[4] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefP', parent=body_style, fontSize=8, leading=10.5, leftIndent=10, spaceAfter=2)))

    # Appendix D: Contributions of Group Members Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("Appendix D: Contributions of Group Members", h1_style))
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

    doc.build(story, canvasmaker=SOALabCanvas)
    print(f"SOA Laboratory Record PDF successfully created at: {PDF_PATH}")

if __name__ == "__main__":
    build_soa_pdf()
