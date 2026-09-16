"""
Advanced Multilayer Perceptron (MLP) Clinical Diabetes Risk Intelligence Suite.
Enterprise clinical decision-support system featuring live neural inference,
biomarker deviation analytics, explainable risk drivers, neural architecture inspection,
and multi-model research benchmarking.
"""

import os
import sys
import base64
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Ensure src modules are resolvable by joblib unpickler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

try:
    from preprocessing import DiabetesPreprocessor
except ImportError:
    pass

# Set Page Configuration
st.set_page_config(
    page_title="EndoPredict AI | Clinical Diabetes MLP Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System & CSS
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
    }
    
    /* Gauge and Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-val {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 600;
    }
    
    /* Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
    }
    .badge-pill-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .badge-pill-mod {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }
    .badge-pill-high {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
    }
    
    /* Risk driver progress bar */
    .driver-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
    }
    .driver-name {
        font-weight: 600;
        font-size: 0.95rem;
        flex: 1;
    }
    .driver-status {
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
    
    /* Architecture Diagram Node */
    .arch-node {
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        color: #e0f2fe;
    }
</style>
""", unsafe_allow_html=True)

# Path Constants
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "eng_preprocessor.joblib")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def compute_engineered_features(data_dict):
    """Generates all 19 feature columns required by the trained model pipeline."""
    df = pd.DataFrame([data_dict])
    
    # 1. BMI Categories
    df['BMI_Underweight'] = (df['BMI'] < 18.5).astype(int)
    df['BMI_Normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25.0)).astype(int)
    df['BMI_Overweight'] = ((df['BMI'] >= 25.0) & (df['BMI'] < 30.0)).astype(int)
    df['BMI_Obese'] = (df['BMI'] >= 30.0).astype(int)
    
    # 2. Glucose Categories
    df['Glucose_Normal'] = (df['Glucose'] < 100).astype(int)
    df['Glucose_Prediabetes'] = ((df['Glucose'] >= 100) & (df['Glucose'] <= 125)).astype(int)
    df['Glucose_Diabetes'] = (df['Glucose'] > 125).astype(int)
    
    # 3. Age Groups
    df['Age_Young'] = (df['Age'] < 30).astype(int)
    df['Age_Middle'] = ((df['Age'] >= 30) & (df['Age'] <= 50)).astype(int)
    df['Age_Senior'] = (df['Age'] > 50).astype(int)
    
    # 4. Clinical Interactions
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1e-5)
    df['Insulin_Resistance_Proxy'] = (df['Glucose'] * df['Insulin']) / 405.0
    df['Pregnancy_Age_Risk'] = df['Pregnancies'] / (df['Age'] + 1e-5)
    df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
    
    # 5. Log Transformations
    df['Log_Insulin'] = np.log1p(np.maximum(0, df['Insulin']))
    df['Log_DPF'] = np.log1p(np.maximum(0, df['DiabetesPedigreeFunction']))
    
    return df

def generate_gauge_svg(prob_pct, risk_color):
    """Renders a modern, animated radial gauge SVG."""
    stroke_offset = 283 - (283 * (prob_pct / 100))
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <svg width="180" height="180" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="42" stroke="rgba(255,255,255,0.1)" stroke-width="8" fill="none" />
            <circle cx="50" cy="50" r="42" stroke="{risk_color}" stroke-width="8" fill="none"
                    stroke-dasharray="264" stroke-dashoffset="{264 - (264 * (prob_pct / 100))}"
                    stroke-linecap="round" transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 0.8s ease;" />
            <text x="50" y="47" font-size="19" font-weight="800" fill="#f8fafc" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif">{prob_pct:.1f}%</text>
            <text x="50" y="63" font-size="7" font-weight="600" fill="#94a3b8" text-anchor="middle" letter-spacing="1">PROBABILITY</text>
        </svg>
    </div>
    """

def render_html_table(df):
    """Pure HTML table generator that eliminates PyArrow DLL dependencies."""
    table_rows = "".join([
        f"""<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            <td style="padding: 10px 16px; font-weight: 600; color: #cbd5e1;">{col}</td>
            <td style="padding: 10px 16px; font-family: 'Courier New', monospace; text-align: right; color: #38bdf8; font-weight: 700;">
                {f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)}
            </td>
        </tr>"""
        for col, val in df.iloc[0].items()
    ])
    return f"""
    <div style="border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; overflow: hidden; margin-top: 12px; background: rgba(15, 23, 42, 0.4);">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
            <thead>
                <tr style="background: rgba(255, 255, 255, 0.06); border-bottom: 2px solid rgba(255, 255, 255, 0.12); text-align: left;">
                    <th style="padding: 12px 16px; font-weight: 700; color: #f1f5f9;">Biomarker / Engineered Feature</th>
                    <th style="padding: 12px 16px; text-align: right; font-weight: 700; color: #f1f5f9;">Computed Value</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🩺 EndoPredict AI | Multilayer Perceptron Suite</div>
        <div class="hero-subtitle">
            Next-generation clinical decision-support and metabolic risk stratification engine. Powered by deep feedforward neural networks (MLP) with Batch Normalization, Dropout regularization, and domain-engineered biometric interactions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model, preprocessor = load_artifacts()
    if model is None or preprocessor is None:
        st.error("⚠️ Model artifacts missing. Please run `python src/train.py` first.")
        return

    # Tabs Interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Live Clinical Risk Assessor",
        "🧠 Neural Architecture & Dynamics",
        "📊 Benchmark & Research Analytics",
        "📁 Batch Cohort Simulator"
    ])
    
    # -------------------------------------------------------------------------------------------------
    # TAB 1: LIVE RISK PREDICTION & CLINICAL REPORT
    # -------------------------------------------------------------------------------------------------
    with tab1:
        st.sidebar.markdown("### 📋 Clinical Presets & Archetypes")
        preset = st.sidebar.selectbox(
            "Select Patient Profile Archetype:",
            [
                "Custom Patient Input",
                "Healthy Young Adult (Low Baseline)",
                "Prediabetic Borderline (Impaired Glucose)",
                "High-Risk Metabolic Patient (Hyperglycemic)",
                "Gestational & Genetic Predisposition Risk"
            ]
        )
        
        # Configure preset parameter values
        if preset == "Healthy Young Adult (Low Baseline)":
            defaults = dict(preg=0, gluc=82, bp=66, skin=18, ins=55, bmi=21.4, dpf=0.20, age=22)
        elif preset == "Prediabetic Borderline (Impaired Glucose)":
            defaults = dict(preg=2, gluc=116, bp=78, skin=27, ins=115, bmi=28.2, dpf=0.46, age=38)
        elif preset == "High-Risk Metabolic Patient (Hyperglycemic)":
            defaults = dict(preg=6, gluc=180, bp=92, skin=38, ins=290, bmi=38.4, dpf=1.12, age=55)
        elif preset == "Gestational & Genetic Predisposition Risk":
            defaults = dict(preg=5, gluc=132, bp=84, skin=32, ins=160, bmi=32.1, dpf=1.45, age=34)
        else:
            defaults = dict(preg=1, gluc=110, bp=72, skin=24, ins=90, bmi=26.0, dpf=0.35, age=30)
            
        st.sidebar.markdown("---")
        st.sidebar.info("💡 **Clinical Tip**: Adjust sliders below or load presets to evaluate neural network confidence and risk sensitivity in real time.")
        
        # Clinical Parameter Inputs
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            st.markdown("#### 🩸 Glycemic & Metabolic Indicators")
            glucose = st.slider("Plasma Glucose Concentration (mg/dL)", 40, 250, int(defaults['gluc']),
                                help="Fasting plasma glucose (mg/dL). Reference: Normal < 100, Prediabetes 100-125, Diabetes >= 126")
            insulin = st.slider("2-Hour Serum Insulin (μU/mL)", 10, 800, int(defaults['ins']),
                                help="2-hour postprandial serum insulin. Normal fasting: 16-166 μU/mL")
            dpf = st.slider("Diabetes Pedigree Function (Genetic Score)", 0.05, 2.50, float(defaults['dpf']), step=0.01,
                            help="Genetic susceptibility function computed from family history pedigree.")
            pregnancies = st.number_input("Pregnancy History (Count)", min_value=0, max_value=20, value=int(defaults['preg']))
            
        with col_in2:
            st.markdown("#### 📏 Physiological & Anthropometric Indicators")
            bmi = st.slider("Body Mass Index (BMI in kg/m²)", 10.0, 65.0, float(defaults['bmi']), step=0.1,
                            help="BMI = Weight(kg) / Height(m)². Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese >=30")
            blood_pressure = st.slider("Diastolic Blood Pressure (mm Hg)", 40, 140, int(defaults['bp']),
                                       help="Diastolic BP (mm Hg). Normal <80, Elevated 80-89, Stage 2 HTN >=90")
            skin_thickness = st.slider("Triceps Skinfold Thickness (mm)", 5, 99, int(defaults['skin']))
            age = st.slider("Patient Age (Years)", 18, 100, int(defaults['age']))
            
        st.markdown("---")
        btn_predict = st.button("⚡ Execute Multilayer Perceptron Inference", type="primary", use_container_width=True)
        
        # Inference pipeline
        patient_dict = {
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }
        df_feat = compute_engineered_features(patient_dict)
        df_scaled = preprocessor.transform(df_feat)
        prob = model.predict_proba(df_scaled)[0, 1]
        pred = int(prob >= 0.5)
        prob_pct = prob * 100
        
        # Diagnostic Assessment Display
        st.markdown("### 📊 Diagnostic Risk Intelligence Summary")
        
        col_res1, col_res2, col_res3 = st.columns([1.2, 1.5, 1.3])
        
        if prob < 0.35:
            risk_color = "#10b981"
            risk_label = "🟢 Low Clinical Risk"
            badge_html = '<span class="badge-pill badge-pill-low">✅ Negative / Non-Diabetic</span>'
            rec_text = "Patient biomarkers are within healthy metabolic baselines. Recommend routine annual checkups and healthy lifestyle maintenance."
        elif prob <= 0.65:
            risk_color = "#f59e0b"
            risk_label = "🟡 Moderate / Prediabetic Risk"
            badge_html = '<span class="badge-pill badge-pill-mod">⚠️ Borderline Prediabetes</span>'
            rec_text = "Elevated glycemic or adiposity indicators detected. Order confirmatory Fasting Plasma Glucose (FPG) and HbA1c testing. Structured lifestyle intervention recommended."
        else:
            risk_color = "#ef4444"
            risk_label = "🔴 High Clinical Risk"
            badge_html = '<span class="badge-pill badge-pill-high">🚨 High Risk / Diabetic</span>'
            rec_text = "Significant hyperglycemia, insulin resistance proxy, and clinical risk factors present. Immediate specialist consultation, Oral Glucose Tolerance Test (OGTT), and metabolic panel advised."

        with col_res1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(generate_gauge_svg(prob_pct, risk_color), unsafe_allow_html=True)
            st.markdown(f'<div class="stat-label">Neural Probability: {prob_pct:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res2:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown(f"**Diagnostic Classification:**<br>{badge_html}", unsafe_allow_html=True)
            st.markdown(f"<br>**Risk Stratification Tier:**<br><span style='color: {risk_color}; font-size: 1.25rem; font-weight: 800;'>{risk_label}</span>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #94a3b8; font-size: 0.9rem; margin-top: 10px; line-height: 1.4;'>{rec_text}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res3:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("**🧪 Key Clinical Biomarkers**")
            homa_proxy = (glucose * insulin) / 405.0
            st.write(f"• **HOMA-IR Proxy:** `{homa_proxy:.2f}` " + ("(Elevated)" if homa_proxy > 2.5 else "(Normal)"))
            st.write(f"• **Glucose Status:** `{glucose} mg/dL` " + ("(High)" if glucose >= 126 else ("(Borderline)" if glucose >= 100 else "(Optimal)")))
            st.write(f"• **Adiposity Index:** `{bmi:.1f} kg/m²` " + ("(Obese)" if bmi >= 30 else ("(Overweight)" if bmi >= 25 else "(Normal)")))
            st.write(f"• **Genetic Pedigree:** `{dpf:.2f}` " + ("(High)" if dpf > 0.8 else "(Moderate/Low)"))
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Clinical Risk Driver Decomposition
        st.markdown("#### 🔬 Key Diagnostic Risk Drivers & Deviations")
        c_drv1, c_drv2 = st.columns(2)
        
        with c_drv1:
            # Glucose status bar
            g_status = "Critical (>=126)" if glucose >= 126 else ("Elevated (100-125)" if glucose >= 100 else "Normal (<100)")
            g_bg = "#ef4444" if glucose >= 126 else ("#f59e0b" if glucose >= 100 else "#10b981")
            st.markdown(f"""
            <div class="driver-row">
                <span class="driver-name">🩸 Plasma Glucose ({glucose} mg/dL)</span>
                <span class="driver-status" style="background: {g_bg}22; color: {g_bg}; border: 1px solid {g_bg}66;">{g_status}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI status bar
            b_status = "Obese (>=30.0)" if bmi >= 30.0 else ("Overweight (25.0-29.9)" if bmi >= 25.0 else "Healthy (18.5-24.9)")
            b_bg = "#ef4444" if bmi >= 30.0 else ("#f59e0b" if bmi >= 25.0 else "#10b981")
            st.markdown(f"""
            <div class="driver-row">
                <span class="driver-name">⚖️ Body Mass Index ({bmi:.1f} kg/m²)</span>
                <span class="driver-status" style="background: {b_bg}22; color: {b_bg}; border: 1px solid {b_bg}66;">{b_status}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with c_drv2:
            # Insulin Resistance status
            i_status = "Severe Resistance (>4.0)" if homa_proxy > 4.0 else ("Moderate Resistance (>2.5)" if homa_proxy > 2.5 else "Optimal (<2.5)")
            i_bg = "#ef4444" if homa_proxy > 4.0 else ("#f59e0b" if homa_proxy > 2.5 else "#10b981")
            st.markdown(f"""
            <div class="driver-row">
                <span class="driver-name">⚡ Insulin Resistance Proxy ({homa_proxy:.2f})</span>
                <span class="driver-status" style="background: {i_bg}22; color: {i_bg}; border: 1px solid {i_bg}66;">{i_status}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Blood pressure status
            bp_status = "Stage 2 HTN (>=90)" if blood_pressure >= 90 else ("Stage 1 HTN (80-89)" if blood_pressure >= 80 else "Normal (<80)")
            bp_bg = "#ef4444" if blood_pressure >= 90 else ("#f59e0b" if blood_pressure >= 80 else "#10b981")
            st.markdown(f"""
            <div class="driver-row">
                <span class="driver-name">💓 Diastolic Blood Pressure ({blood_pressure} mm Hg)</span>
                <span class="driver-status" style="background: {bp_bg}22; color: {bp_bg}; border: 1px solid {bp_bg}66;">{bp_status}</span>
            </div>
            """, unsafe_allow_html=True)

        # Full Feature Vector Expander
        with st.expander("🔍 Inspect Full 19-Dimensional Engineered Feature Vector"):
            st.markdown(render_html_table(df_feat), unsafe_allow_html=True)

    # -------------------------------------------------------------------------------------------------
    # TAB 2: NEURAL ARCHITECTURE & MATHEMATICAL FORMULATION
    # -------------------------------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🧠 Deep Multilayer Perceptron Architecture")
        st.markdown("""
        The clinical prediction model is implemented as a specialized deep Feedforward Artificial Neural Network engineered with Batch Normalization and Dropout layers to prevent overfitting on tabular medical data.
        """)
        
        # Architecture Visual Flow
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0;">
            <div class="arch-node">
                <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 700; text-transform: uppercase;">Input Layer</div>
                <div style="font-size: 1.5rem; font-weight: 800; margin: 4px 0;">19 Features</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Standardized Biometrics & Interactions</div>
            </div>
            <div class="arch-node">
                <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 700; text-transform: uppercase;">Hidden Layer 1</div>
                <div style="font-size: 1.5rem; font-weight: 800; margin: 4px 0;">64 Neurons</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">BatchNorm + ReLU + Dropout (0.2)</div>
            </div>
            <div class="arch-node">
                <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 700; text-transform: uppercase;">Hidden Layer 2</div>
                <div style="font-size: 1.5rem; font-weight: 800; margin: 4px 0;">32 Neurons</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">BatchNorm + ReLU + Dropout (0.2)</div>
            </div>
            <div class="arch-node" style="background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3);">
                <div style="font-size: 0.8rem; color: #34d399; font-weight: 700; text-transform: uppercase;">Output Layer</div>
                <div style="font-size: 1.5rem; font-weight: 800; margin: 4px 0; color: #34d399;">1 Neuron</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Sigmoid Probability P(Diabetic)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mathematical Formulation
        st.markdown("#### 📐 Mathematical Foundations")
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.markdown(r"""
            **1. Forward Propagation:**
            $$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$
            $$\mathbf{h}^{(1)} = \text{ReLU}\left(\text{BatchNorm}(\mathbf{z}^{(1)})\right)$$
            $$\mathbf{z}^{(2)} = \mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)}$$
            $$\hat{y} = \sigma\left(\mathbf{W}^{(3)}\mathbf{h}^{(2)} + b^{(3)}\right) = \frac{1}{1 + e^{-z^{(3)}}}$$
            """)
        with col_math2:
            st.markdown(r"""
            **2. Binary Cross-Entropy Loss & Regularization:**
            $$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right] + \lambda \|\mathbf{W}\|_2^2$$
            **3. Optimization:**
            - **Algorithm:** Adam Optimizer ($\beta_1=0.9, \beta_2=0.999$)
            - **Learning Rate:** $\eta = 0.001$ with weight decay $L_2 = 10^{-4}$
            """)
            
        # Embedded Learning Curve Plot
        lc_path = os.path.join(VIZ_DIR, "mlp_learning_curves.png")
        if os.path.exists(lc_path):
            st.markdown("#### 📉 Neural Training & Validation Learning Curves")
            st.image(lc_path, use_container_width=True, caption="Figure: Binary Cross-Entropy Loss and Accuracy progression across training epochs.")

    # -------------------------------------------------------------------------------------------------
    # TAB 3: BENCHMARK & RESEARCH ANALYTICS
    # -------------------------------------------------------------------------------------------------
    with tab3:
        st.markdown("### 📊 Comprehensive Model Benchmarks on Unseen Test Cohort")
        comp_csv = os.path.join(VIZ_DIR, "model_comparison_results.csv")
        if os.path.exists(comp_csv):
            df_comp = pd.read_csv(comp_csv)
            # Render benchmark table cleanly with HTML
            comp_rows = "".join([
                f"""<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08); background: {'rgba(56, 189, 248, 0.1)' if 'Multilayer' in str(r['Model']) else 'transparent'};">
                    <td style="padding: 10px 14px; font-weight: 700; color: {'#38bdf8' if 'Multilayer' in str(r['Model']) else '#f1f5f9'};">{r['Model']}</td>
                    <td style="padding: 10px 14px; text-align: center;">{float(r['Accuracy'])*100:.2f}%</td>
                    <td style="padding: 10px 14px; text-align: center; color: #34d399; font-weight: 700;">{float(r['Recall (Sensitivity)'])*100:.2f}%</td>
                    <td style="padding: 10px 14px; text-align: center;">{float(r['Specificity'])*100:.2f}%</td>
                    <td style="padding: 10px 14px; text-align: center;">{float(r['Precision'])*100:.2f}%</td>
                    <td style="padding: 10px 14px; text-align: center;">{float(r['F1-Score'])*100:.2f}%</td>
                    <td style="padding: 10px 14px; text-align: center; font-weight: 700;">{float(r['ROC-AUC']):.4f}</td>
                    <td style="padding: 10px 14px; text-align: center; color: #f87171;">{r['False Negatives (FN)']}</td>
                    <td style="padding: 10px 14px; text-align: center; color: #fbbf24;">{r['False Positives (FP)']}</td>
                </tr>"""
                for _, r in df_comp.iterrows()
            ])
            
            st.markdown(f"""
            <div style="border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; overflow: hidden; margin-bottom: 24px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                    <thead>
                        <tr style="background: rgba(255, 255, 255, 0.08); border-bottom: 2px solid rgba(255, 255, 255, 0.15); text-align: center;">
                            <th style="padding: 12px 14px; text-align: left;">Algorithm</th>
                            <th style="padding: 12px 14px;">Accuracy</th>
                            <th style="padding: 12px 14px; color: #34d399;">Sensitivity (Recall)</th>
                            <th style="padding: 12px 14px;">Specificity</th>
                            <th style="padding: 12px 14px;">Precision</th>
                            <th style="padding: 12px 14px;">F1-Score</th>
                            <th style="padding: 12px 14px;">ROC-AUC</th>
                            <th style="padding: 12px 14px; color: #f87171;">FN (Misses)</th>
                            <th style="padding: 12px 14px; color: #fbbf24;">FP (False Alarms)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comp_rows}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        # Clinical Visual Analytics Gallery
        col_v1, col_v2 = st.columns(2)
        roc_path = os.path.join(VIZ_DIR, "roc_curves.png")
        cm_path = os.path.join(VIZ_DIR, "confusion_matrices.png")
        pr_path = os.path.join(VIZ_DIR, "precision_recall_curves.png")
        corr_path = os.path.join(VIZ_DIR, "correlation_heatmap.png")
        
        with col_v1:
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True, caption="Multi-Model ROC Curves")
            if os.path.exists(pr_path):
                st.image(pr_path, use_container_width=True, caption="Precision-Recall Curves")
        with col_v2:
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True, caption="Confusion Matrix Grid")
            if os.path.exists(corr_path):
                st.image(corr_path, use_container_width=True, caption="Feature Correlation Heatmap")

    # -------------------------------------------------------------------------------------------------
    # TAB 4: BATCH PATIENT COHORT SCREENING SIMULATOR
    # -------------------------------------------------------------------------------------------------
    with tab4:
        st.markdown("### 📁 Batch Patient Screening Simulator")
        st.markdown("Evaluate synthetic clinical cohorts simultaneously to assess population-level risk distributions.")
        
        num_sim = st.slider("Simulate Cohort Size (Patients):", 5, 50, 10)
        
        if st.button("🎲 Generate & Screen Simulated Patient Cohort", type="primary"):
            np.random.seed(int(num_sim))
            sim_records = []
            for i in range(num_sim):
                p = {
                    'Patient_ID': f"PT-{1000 + i}",
                    'Pregnancies': int(np.random.choice([0, 1, 2, 3, 5, 7])),
                    'Glucose': int(np.random.normal(120, 35)),
                    'BloodPressure': int(np.random.normal(72, 12)),
                    'SkinThickness': int(np.random.normal(26, 8)),
                    'Insulin': int(np.random.exponential(90) + 15),
                    'BMI': round(float(np.random.normal(29, 6)), 1),
                    'DiabetesPedigreeFunction': round(float(np.random.exponential(0.4) + 0.1), 2),
                    'Age': int(np.random.choice(range(21, 75)))
                }
                # Clip bounds
                p['Glucose'] = max(50, min(240, p['Glucose']))
                p['BloodPressure'] = max(45, min(130, p['BloodPressure']))
                p['BMI'] = max(16.0, min(55.0, p['BMI']))
                p['DiabetesPedigreeFunction'] = max(0.08, min(2.4, p['DiabetesPedigreeFunction']))
                
                feat_df = compute_engineered_features(p)
                scaled_df = preprocessor.transform(feat_df)
                p_prob = model.predict_proba(scaled_df)[0, 1]
                
                p['Predicted_Risk'] = f"{p_prob * 100:.1f}%"
                p['Risk_Tier'] = "High Risk (Diabetic)" if p_prob >= 0.65 else ("Borderline Risk" if p_prob >= 0.35 else "Low Risk (Healthy)")
                sim_records.append(p)
                
            sim_df = pd.DataFrame(sim_records)
            
            # Summary Metrics
            c_s1, c_s2, c_s3 = st.columns(3)
            high_count = (sim_df['Risk_Tier'] == "High Risk (Diabetic)").sum()
            mod_count = (sim_df['Risk_Tier'] == "Borderline Risk").sum()
            low_count = (sim_df['Risk_Tier'] == "Low Risk (Healthy)").sum()
            
            with c_s1:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color: #f87171;">{high_count}</div><div class="stat-label">High Risk Cases</div></div>', unsafe_allow_html=True)
            with c_s2:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color: #fbbf24;">{mod_count}</div><div class="stat-label">Borderline Cases</div></div>', unsafe_allow_html=True)
            with c_s3:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color: #34d399;">{low_count}</div><div class="stat-label">Low Risk Baselines</div></div>', unsafe_allow_html=True)
                
            st.markdown("#### 📋 Cohort Screening Register")
            # Render HTML table of cohort
            cohort_rows = "".join([
                f"""<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                    <td style="padding: 8px 12px; font-weight: 700; color: #38bdf8;">{r['Patient_ID']}</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['Age']}</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['Glucose']} mg/dL</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['BMI']}</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['Insulin']} μU/mL</td>
                    <td style="padding: 8px 12px; text-align: center; font-weight: 700;">{r['Predicted_Risk']}</td>
                    <td style="padding: 8px 12px; font-weight: 700; color: {'#f87171' if 'High' in r['Risk_Tier'] else ('#fbbf24' if 'Borderline' in r['Risk_Tier'] else '#34d399')};">{r['Risk_Tier']}</td>
                </tr>"""
                for _, r in sim_df.iterrows()
            ])
            
            st.markdown(f"""
            <div style="border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; overflow: hidden; margin-top: 12px; max-height: 400px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                    <thead>
                        <tr style="background: rgba(255, 255, 255, 0.08); text-align: center; position: sticky; top: 0;">
                            <th style="padding: 10px 12px; text-align: left;">Patient ID</th>
                            <th style="padding: 10px 12px;">Age</th>
                            <th style="padding: 10px 12px;">Glucose</th>
                            <th style="padding: 10px 12px;">BMI</th>
                            <th style="padding: 10px 12px;">Insulin</th>
                            <th style="padding: 10px 12px;">Predicted Risk</th>
                            <th style="padding: 10px 12px; text-align: left;">Clinical Stratification</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cohort_rows}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption("🔒 **EndoPredict AI Clinical Decision System** • Built with PyTorch, Scikit-Learn, and Streamlit • Developed for Lab Assignment 01 (Multilayer Perceptron for Diabetes Prediction). Strictly for research & academic evaluation.")

if __name__ == "__main__":
    main()
