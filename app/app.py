"""
EndoPredict AI: Academic Multilayer Perceptron Diabetes Prediction Suite.
Interactive academic demonstration tool featuring real-time neural inference,
dynamic model metadata inspection, engineered feature vectors, and batch cohort simulation.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

# Ensure src modules are resolvable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prediction import predict_patient, load_artifacts, RAW_FEATURE_NAMES

# Page Configuration
st.set_page_config(
    page_title="EndoPredict AI | Diabetes MLP Academic Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System
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
        font-size: 1.02rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
    }
    
    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .stat-label {
        font-size: 0.82rem;
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
    }
    .badge-pill-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .badge-pill-high {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
    }
    
    /* Feature row */
    .driver-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
    }
    .driver-name {
        font-weight: 600;
        font-size: 0.92rem;
    }
    .driver-status {
        font-weight: 700;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def generate_gauge_svg(prob_pct, risk_color):
    """Renders a modern, animated radial gauge SVG."""
    stroke_dashoffset = 264 - (264 * (prob_pct / 100.0))
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <svg width="170" height="170" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="42" stroke="rgba(255,255,255,0.1)" stroke-width="8" fill="none" />
            <circle cx="50" cy="50" r="42" stroke="{risk_color}" stroke-width="8" fill="none"
                    stroke-dasharray="264" stroke-dashoffset="{stroke_dashoffset}"
                    stroke-linecap="round" transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 0.8s ease;" />
            <text x="50" y="47" font-size="19" font-weight="800" fill="#f8fafc" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif">{prob_pct:.1f}%</text>
            <text x="50" y="63" font-size="7" font-weight="600" fill="#94a3b8" text-anchor="middle" letter-spacing="1">PROBABILITY</text>
        </svg>
    </div>
    """

def render_html_table(df):
    """Pure HTML table generator eliminating PyArrow DLL dependencies."""
    table_rows = "".join([
        f"""<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            <td style="padding: 9px 14px; font-weight: 600; color: #cbd5e1;">{col}</td>
            <td style="padding: 9px 14px; font-family: monospace; text-align: right; color: #38bdf8; font-weight: 700;">
                {f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)}
            </td>
        </tr>"""
        for col, val in df.iloc[0].items()
    ])
    return f"""
    <div style="border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; overflow: hidden; margin-top: 12px; background: rgba(15, 23, 42, 0.4);">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.90rem;">
            <thead>
                <tr style="background: rgba(255, 255, 255, 0.06); border-bottom: 2px solid rgba(255, 255, 255, 0.12); text-align: left;">
                    <th style="padding: 10px 14px; font-weight: 700; color: #f1f5f9;">Processed Feature Name</th>
                    <th style="padding: 10px 14px; text-align: right; font-weight: 700; color: #f1f5f9;">Normalized Value</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

def main():
    # Hero Header
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🩺 EndoPredict AI | Multilayer Perceptron Suite</div>
        <div class="hero-subtitle">
            Academic Machine Learning demonstration for diabetes onset prediction using a tuned Scikit-Learn Multilayer Perceptron (MLP), structured data imputation, and domain-engineered feature interactions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        model, preprocessor, config = load_artifacts()
    except Exception as e:
        st.error(f"⚠️ Production artifacts missing or invalid: {e}. Please execute `python src/train.py`.")
        return

    # Multi-View Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Live Patient Prediction",
        "ℹ️ Model Architecture & Metadata",
        "📊 Benchmark & Research Analytics",
        "📁 Batch Cohort Simulator"
    ])
    
    # -------------------------------------------------------------------------------------------------
    # TAB 1: LIVE PREDICTION
    # -------------------------------------------------------------------------------------------------
    with tab1:
        st.sidebar.markdown("### 📋 Patient Input Presets")
        preset = st.sidebar.selectbox(
            "Load Sample Input Values:",
            [
                "Custom Input",
                "Demo — Lower Example",
                "Demo — Borderline Example",
                "Demo — Higher Example"
            ]
        )
        
        # Configure preset sample values
        if preset == "Demo — Lower Example":
            defaults = dict(preg=0, gluc=82, bp=66, skin=18, ins=55, bmi=21.4, dpf=0.20, age=22)
        elif preset == "Demo — Borderline Example":
            defaults = dict(preg=2, gluc=118, bp=78, skin=26, ins=110, bmi=28.0, dpf=0.45, age=38)
        elif preset == "Demo — Higher Example":
            defaults = dict(preg=6, gluc=178, bp=90, skin=38, ins=280, bmi=37.5, dpf=1.10, age=54)
        else:
            defaults = dict(preg=1, gluc=110, bp=72, skin=24, ins=90, bmi=26.0, dpf=0.35, age=30)
            
        st.sidebar.markdown("---")
        st.sidebar.info("ℹ️ **Note**: Presets populate sample input values only. All predictions are generated dynamically by the trained MLPClassifier.")
        
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            st.markdown("#### 🩸 Glycemic & Metabolic Inputs")
            glucose = st.slider("Glucose (mg/dL)", 0, 250, int(defaults['gluc']),
                                help="Plasma glucose concentration (2-hour oral glucose tolerance test). Note: 0 values are imputed using training median.")
            insulin = st.slider("Insulin (μU/mL)", 0, 800, int(defaults['ins']),
                                help="2-Hour serum insulin (μU/mL). Note: 0 values are imputed using training median.")
            dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.50, float(defaults['dpf']), step=0.01,
                            help="Genetic pedigree risk score computed from family diabetes history.")
            pregnancies = st.number_input("Pregnancies (Count)", min_value=0, max_value=20, value=int(defaults['preg']))
            
        with col_in2:
            st.markdown("#### 📏 Physiological Inputs")
            bmi = st.slider("Body Mass Index (BMI in kg/m²)", 0.0, 65.0, float(defaults['bmi']), step=0.1,
                            help="Body Mass Index = Weight(kg) / Height(m)². Note: 0 values are imputed using training median.")
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, int(defaults['bp']),
                                       help="Blood Pressure reading (mm Hg). Note: 0 values are imputed using training median.")
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, int(defaults['skin']),
                                       help="Triceps skin fold thickness (mm). Note: 0 values are imputed using training median.")
            age = st.slider("Age (Years)", 18, 100, int(defaults['age']))
            
        st.markdown("---")
        
        # Execute Real Prediction Pipeline
        patient_dict = {
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }
        
        res = predict_patient(patient_dict, threshold=0.50)
        prob = res['probability']
        pred = res['predicted_class']
        prob_pct = prob * 100.0
        actual_dim = res['processed_feature_count']
        
        # Display Prediction Outputs
        st.markdown("### 📊 Model Prediction Summary")
        col_res1, col_res2, col_res3 = st.columns([1.2, 1.5, 1.3])
        
        if pred == 1:
            risk_color = "#ef4444"
            badge_html = '<span class="badge-pill badge-pill-high">⚠️ Positive Prediction (Class 1)</span>'
            summary_desc = "The model estimated a probability exceeding the classification threshold (0.50), indicating positive predicted status for diabetes onset."
        else:
            risk_color = "#10b981"
            badge_html = '<span class="badge-pill badge-pill-low">✅ Negative Prediction (Class 0)</span>'
            summary_desc = "The model estimated a probability below the classification threshold (0.50), indicating negative predicted status for diabetes onset."

        with col_res1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(generate_gauge_svg(prob_pct, risk_color), unsafe_allow_html=True)
            st.markdown(f'<div class="stat-label">Model Probability: {prob_pct:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res2:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown(f"**Predicted Class:**<br>{badge_html}", unsafe_allow_html=True)
            st.markdown(f"<br>**Decision Threshold:** `0.50` | **Raw Probability:** `{prob:.4f}`", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #94a3b8; font-size: 0.88rem; margin-top: 10px; line-height: 1.4;'>{summary_desc}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res3:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("**🔬 Model Input Summary**")
            st.write(f"• **Raw Features:** `{res['raw_feature_count']}`")
            st.write(f"• **Processed Features:** `{actual_dim}`")
            st.write(f"• **Glucose Reading:** `{glucose} mg/dL`")
            st.write(f"• **BMI Value:** `{bmi:.1f} kg/m²`")
            st.write(f"• **Pedigree Function:** `{dpf:.2f}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Engineered Features Section
        st.markdown(f"#### 🧬 Inspect Full {actual_dim}-Dimensional Processed Feature Vector")
        with st.expander("Click to view transformed normalized feature vector"):
            st.markdown(render_html_table(res['processed_features']), unsafe_allow_html=True)

    # -------------------------------------------------------------------------------------------------
    # TAB 2: MODEL ARCHITECTURE & METADATA
    # -------------------------------------------------------------------------------------------------
    with tab2:
        st.markdown("### ℹ️ Production Model Metadata & Architecture")
        st.markdown("All parameters below are read dynamically from the serialized `model_config.json` artifact:")
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-val" style="color: #38bdf8;">{config.get("framework", "scikit-learn").upper()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-label">Framework: {config.get("model_type", "MLPClassifier")}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_m2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-val" style="color: #34d399;">{config.get("processed_feature_count", actual_dim)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">Processed Input Dimensions</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_m3:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-val" style="color: #fbbf24;">{tuple(config.get("hidden_layer_sizes", [64, 32]))}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">Hidden Layer Architecture</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("#### ⚙️ Hyperparameter Configuration")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.write(f"• **Activation Function:** `{config.get('activation', 'relu')}`")
            st.write(f"• **Optimizer (Solver):** `{config.get('solver', 'adam')}`")
            st.write(f"• **Initial Learning Rate:** `{config.get('learning_rate_init', 0.001)}`")
            st.write(f"• **Batch Size:** `{config.get('batch_size', 32)}`")
        with c_p2:
            st.write(f"• **L2 Regularization (Alpha):** `{config.get('alpha', 0.001)}`")
            st.write(f"• **Maximum Iterations:** `{config.get('max_iter', 400)}`")
            st.write(f"• **Classification Threshold:** `{config.get('threshold', 0.50)}`")
            st.write(f"• **Training Dataset Size:** `{config.get('train_samples', 536)} + {config.get('val_samples', 116)} (Train+Val)`")

        # Loss Curve Visualizer
        lc_path = os.path.join(VIZ_DIR, "mlp_learning_curves.png")
        if os.path.exists(lc_path):
            st.markdown("#### 📉 Scikit-Learn MLP Training Loss Progression")
            st.image(lc_path, caption="Figure: Cross-Entropy Loss curve during model training iterations.", use_container_width=True)

    # -------------------------------------------------------------------------------------------------
    # TAB 3: BENCHMARK & RESEARCH ANALYTICS
    # -------------------------------------------------------------------------------------------------
    with tab3:
        st.markdown("### 📊 Comprehensive Model Benchmarks (Untouched Test Set, N=116)")
        comp_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
        if os.path.exists(comp_csv):
            df_comp = pd.read_csv(comp_csv)
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
                <table style="width: 100%; border-collapse: collapse; font-size: 0.90rem;">
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
            
        col_v1, col_v2 = st.columns(2)
        roc_path = os.path.join(VIZ_DIR, "roc_curves.png")
        cm_path = os.path.join(VIZ_DIR, "confusion_matrices.png")
        pr_path = os.path.join(VIZ_DIR, "precision_recall_curves.png")
        corr_path = os.path.join(VIZ_DIR, "correlation_heatmap.png")
        
        with col_v1:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curves for Imbalanced Assessment", use_container_width=True)
        with col_v2:
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix Grid", use_container_width=True)
            if os.path.exists(corr_path):
                st.image(corr_path, caption="Dataset Feature Correlation Matrix", use_container_width=True)

    # -------------------------------------------------------------------------------------------------
    # TAB 4: BATCH COHORT SIMULATOR
    # -------------------------------------------------------------------------------------------------
    with tab4:
        st.markdown("### 📁 Batch Cohort Simulator")
        st.markdown("Simulate and screen multiple patient samples simultaneously.")
        
        num_sim = st.slider("Cohort Sample Count:", 5, 50, 10)
        
        if st.button("🎲 Run Cohort Screening Simulation", type="primary"):
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
                
                res_p = predict_patient(p)
                p['Estimated_Probability'] = f"{res_p['probability'] * 100:.1f}%"
                p['Predicted_Class'] = "Positive (1)" if res_p['predicted_class'] == 1 else "Negative (0)"
                sim_records.append(p)
                
            sim_df = pd.DataFrame(sim_records)
            pos_count = (sim_df['Predicted_Class'] == "Positive (1)").sum()
            neg_count = (sim_df['Predicted_Class'] == "Negative (0)").sum()
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color: #f87171;">{pos_count}</div><div class="stat-label">Positive Class Predictions</div></div>', unsafe_allow_html=True)
            with c_s2:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color: #34d399;">{neg_count}</div><div class="stat-label">Negative Class Predictions</div></div>', unsafe_allow_html=True)
                
            cohort_rows = "".join([
                f"""<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                    <td style="padding: 8px 12px; font-weight: 700; color: #38bdf8;">{r['Patient_ID']}</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['Age']}</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['Glucose']} mg/dL</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['BMI']}</td>
                    <td style="padding: 8px 12px; text-align: center;">{r['Insulin']} μU/mL</td>
                    <td style="padding: 8px 12px; text-align: center; font-weight: 700;">{r['Estimated_Probability']}</td>
                    <td style="padding: 8px 12px; font-weight: 700; color: {'#f87171' if 'Positive' in r['Predicted_Class'] else '#34d399'};">{r['Predicted_Class']}</td>
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
                            <th style="padding: 10px 12px;">Probability</th>
                            <th style="padding: 10px 12px; text-align: left;">Predicted Class</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cohort_rows}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # Academic & Medical Disclaimer Footer
    st.markdown("---")
    st.caption("ℹ️ **Academic & Medical Disclaimer**: This application is an academic machine-learning demonstration developed for Lab Assignment 01 (Predicting Diabetes with Multilayer Perceptron). It is not intended to provide standalone medical diagnosis, medical advice, or treatment plans.")

if __name__ == "__main__":
    main()
