import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from model_pipeline import extract_sparkov_features
from explainability import explain_transaction_risk

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Dark Theme CSS ---
st.markdown("""
<style>
    /* 🤖 Cyber-Sec Neon Theme Setup */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #050505;
        color: #e0e0e0;
        background-image: radial-gradient(circle at 50% 0%, #112233 0%, #050505 70%);
    }
    
    .stAppHeader {
        background-color: transparent !important;
    }

    /* 🔘 Custom Sidebar Navigation Radio Buttons */
    div[role="radiogroup"] label {
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    /* Hide the circular radio dot completely */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Style the Selected Button using the teal accent */
    div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(0, 255, 255, 0.15) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.1) !important;
    }

    /* Make selected text bolder and cyan */
    div[role="radiogroup"] label:has(input:checked) p {
        color: #00f2fe !important;
        font-weight: 800 !important;
        text-shadow: 0 0 5px rgba(0, 242, 254, 0.3) !important;
    }

    /* Hover effect for unselected buttons */
    div[role="radiogroup"] label:not(:has(input:checked)):hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* 🛡️ Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(10, 15, 20, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.15);
        border: 1px solid rgba(0, 255, 255, 0.3);
    }
    
    .metric-card h3 {
        color: #00f2fe;
        font-size: 0.9rem;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-shadow: 0 0 8px rgba(0, 242, 254, 0.3);
    }
    
    .metric-card .value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        word-break: normal;
        overflow-wrap: break-word;
        line-height: 1.2;
    }

    .metric-card .subtext {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 4px;
        font-family: monospace;
    }

    /* 🚨 Neon Risk Badges */
    .badge-high {
        background-color: rgba(255, 8, 68, 0.15);
        color: #ff0844;
        border: 1px solid #ff0844;
        padding: 8px 18px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(255, 8, 68, 0.4);
        text-shadow: 0 0 8px rgba(255, 8, 68, 0.5);
    }
    
    .badge-medium {
        background-color: rgba(255, 170, 0, 0.15);
        color: #ffaa00;
        border: 1px solid #ffaa00;
        padding: 8px 18px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(255, 170, 0, 0.3);
    }

    .badge-low {
        background-color: rgba(11, 163, 96, 0.15);
        color: #0ba360;
        border: 1px solid #0ba360;
        padding: 8px 18px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(11, 163, 96, 0.3);
    }

    /* 📊 Explanation & Terminal Boxes */
    .driver-box {
        background-color: rgba(255, 8, 68, 0.08);
        border-left: 3px solid #ff0844;
        padding: 12px;
        margin-bottom: 10px;
        font-family: monospace;
        color: #ffb3c6;
    }

    .mitigator-box {
        background-color: rgba(11, 163, 96, 0.08);
        border-left: 3px solid #0ba360;
        padding: 12px;
        margin-bottom: 10px;
        font-family: monospace;
        color: #a7f3d0;
    }

    /* ⚡ Action Box */
    .action-box {
        background: rgba(0, 242, 254, 0.05);
        border: 1px dashed #00f2fe;
        border-radius: 4px;
        padding: 16px;
        margin-top: 15px;
        text-align: center;
        font-weight: 700;
        color: #00f2fe;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Pipeline & Models ---
@st.cache_resource
def load_models_and_data():
    model_path = "models/fraud_model_pipeline.pkl"
    metrics_path = "models/metrics.joblib"
    sparkov_data_path = "data/sparkov_fraudTrain.csv"
    data_path = "data/transactions.csv"
    preset_path = "data/preset_scenarios.csv"
    
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        st.error("Model files not found! Running model training pipeline...")
        from model_pipeline import train_and_evaluate_fraud_models
        train_and_evaluate_fraud_models()
        
    pipeline_bundle = joblib.load(model_path)
    metrics_bundle = joblib.load(metrics_path)
    
    df = pd.read_csv(sparkov_data_path) if os.path.exists(sparkov_data_path) else (pd.read_csv(data_path) if os.path.exists(data_path) else None)
    presets = pd.read_csv(preset_path) if os.path.exists(preset_path) else None
    
    return pipeline_bundle, metrics_bundle, df, presets

pipeline_bundle, metrics_bundle, df_transactions, df_presets = load_models_and_data()
best_model_name = pipeline_bundle['model_name']
preprocessor = pipeline_bundle['preprocessor']
model = pipeline_bundle['model']
req_cols = pipeline_bundle['num_features'] + pipeline_bundle['cat_features'] + pipeline_bundle['bin_features']


# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 System Overview"

# --- Full-Page Custom Navigation ---
if not st.session_state.logged_in:
    nav_options = ["🏠 System Overview", "🔑 Login / API Access"]
else:
    nav_options = [
        "🏠 System Overview",
        "⚡ Live Risk Simulator", 
        "📂 Batch CSV Fraud Scanner", 
        "📊 Model Performance & Metrics", 
        "🔍 Fraud Insights & Analytics (EDA)",
        "🚪 Logout"
    ]

# Render Navigation in Sidebar
st.sidebar.markdown("### 🛡️ Navigation")
selected_page = st.sidebar.radio(
    "Navigation", 
    nav_options, 
    index=nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0,
    label_visibility="collapsed"
)

# Handle Logout immediately
if selected_page == "🚪 Logout":
    st.session_state.logged_in = False
    st.session_state.current_page = "🏠 System Overview"
    st.rerun()

# Update page state if changed
if selected_page != st.session_state.current_page and selected_page != "🚪 Logout":
    st.session_state.current_page = selected_page
    st.rerun()

# ==========================================
# TAB OVERVIEW: SYSTEM OVERVIEW
# ==========================================
if st.session_state.current_page == "🏠 System Overview":
    # --- Header Section ---
    st.title("🛡️ AI Financial Fraud Detection & Risk Analysis System")
    st.caption("Real-Time Machine Learning Pipeline for Financial Transaction Risk Scoring, Class Imbalance Mitigation, & Explainable AI")
    
    st.markdown("**How this works:** This app analyzes transaction patterns (like location, time, and purchase history) to flag potentially fraudulent activity before it is approved.")
    st.markdown("---")
    
    # --- Top Banner Stats ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Primary Model</h3>
            <div class="value" style="color: #38bdf8;">{best_model_name}</div>
            <div class="subtext">SMOTE Resampled</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        best_pr = metrics_bundle['results'][best_model_name]['pr_auc']
        best_roc = metrics_bundle['results'][best_model_name]['roc_auc']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model PR-AUC <span title="Precision-Recall Area Under Curve: Measures how well the model catches fraud without making false alarms. Higher is better.">ℹ️</span></h3>
            <div class="value" style="color: #10b981;">{best_pr:.3f}</div>
            <div class="subtext">ROC-AUC <span title="Receiver Operating Characteristic: Overall accuracy of distinguishing fraud from normal.">ℹ️</span>: {best_roc:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        best_rec = metrics_bundle['results'][best_model_name]['recall']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Fraud Recall Rate <span title="Recall: The percentage of actual fraud cases the model successfully catches. Higher recall = catches more real fraud.">ℹ️</span></h3>
            <div class="value" style="color: #f59e0b;">{best_rec*100:.1f}%</div>
            <div class="subtext">Caught Fraud Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        total_len = len(df_transactions) if df_transactions is not None else 50000
        st.markdown(f"""
        <div class="metric-card">
            <h3>Dataset Size</h3>
            <div class="value" style="color: #c084fc;">{total_len:,}</div>
            <div class="subtext">Synthetic data matching real Sparkov schema</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #00f2fe; margin-bottom: 30px; font-weight: 800; font-size: 3.8rem; text-shadow: 0 0 20px rgba(0,242,254,0.6); letter-spacing: -1px;'>Stop fraud before it reaches you.</h1>", unsafe_allow_html=True)
    
    col_o1, col_o2 = st.columns([1.2, 1], gap="large")
    with col_o1:
        st.markdown("### 🛡️ Why Our System?")
        st.markdown("""
        **Real-time request analysis in under 3 milliseconds.**  
        One API call, instant verdict, zero friction for real users. 
        Drop it in today, stop the bleeding tomorrow.
        
        <br>
        
        #### Stops at the Edge:
        - **Account Takeover:** Prevents unauthorized logins.
        - **Card Testing:** Blocks mass automated transaction trials.
        - **Promo Abuse:** Identifies synthetic identities.
        """, unsafe_allow_html=True)
        
    with col_o2:
        st.markdown("### ⚡ Live Verdict Stream (Simulation)")
        st.code("""
12:42:58  ALLOW    US POST /login      1.2ms
12:42:43  ALERT    IN POST /otp        1.4ms
12:42:45  ALLOW    CA POST /payment    1.3ms
12:42:47  BLOCK    RU POST /signup     0.8ms
12:42:49  ALLOW    GB POST /login      1.0ms
        """, language="shell")

# ==========================================
# TAB LOGIN: LOGIN / API ACCESS
# ==========================================
if st.session_state.current_page == "🔑 Login / API Access":
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(2) {
        background-color: #050505;
        background-image: 
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><path d='M50 10 L15 25 L15 55 C15 75 50 90 50 90 C50 90 85 75 85 55 L85 25 Z' fill='none' stroke='%2300f2fe' stroke-width='2' opacity='0.15'/><path d='M35 50 L45 60 L65 35' fill='none' stroke='%2300f2fe' stroke-width='2' opacity='0.15'/></svg>"),
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'><path d='M20 40 L80 40 L80 100 L120 100' fill='none' stroke='%2300f2fe' stroke-width='1.5' opacity='0.4'/><circle cx='20' cy='40' r='4' fill='%2300f2fe' opacity='0.6'/><circle cx='120' cy='100' r='4' fill='%2300f2fe' opacity='0.6'/><path d='M380 60 L300 60 L300 130 L260 130' fill='none' stroke='%2300f2fe' stroke-width='1.5' opacity='0.4'/><circle cx='380' cy='60' r='4' fill='%2300f2fe' opacity='0.6'/><circle cx='260' cy='130' r='4' fill='%2300f2fe' opacity='0.6'/><path d='M30 240 L100 240 L100 160 L140 160' fill='none' stroke='%2300f2fe' stroke-width='1.5' opacity='0.4'/><circle cx='30' cy='240' r='4' fill='%2300f2fe' opacity='0.6'/><circle cx='140' cy='160' r='4' fill='%2300f2fe' opacity='0.6'/><path d='M370 260 L280 260 L280 180 L240 180' fill='none' stroke='%2300f2fe' stroke-width='1.5' opacity='0.4'/><circle cx='370' cy='260' r='4' fill='%2300f2fe' opacity='0.6'/><circle cx='240' cy='180' r='4' fill='%2300f2fe' opacity='0.6'/></svg>"),
            linear-gradient(rgba(14, 42, 56, 0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(14, 42, 56, 0.4) 1px, transparent 1px),
            linear-gradient(135deg, #050505, #0a1622);
        background-size: 
            250px 250px, 
            100% 100%, 
            80px 80px, 
            80px 80px, 
            100% 100%;
        background-position: 
            center center,
            center center,
            top left,
            top left,
            center center;
        background-repeat: no-repeat, no-repeat, repeat, repeat, no-repeat;
        border-radius: 12px;
        padding: 30px;
        border: 1px solid rgba(0, 242, 254, 0.1);
        box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #00f2fe; font-size: 1.5rem;'>Developer Login</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0aec0; margin-bottom: 20px;'>Sign in to access your API keys and production dashboard.</p>", unsafe_allow_html=True)
        
        st.text_input("Work Email", placeholder="developer@company.com")
        st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Log In / Get API Key", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.current_page = "⚡ Live Risk Simulator"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 1: LIVE RISK SIMULATOR
# ==========================================
if st.session_state.current_page == "⚡ Live Risk Simulator":
    st.subheader("⚡ Real-Time Transaction Risk Engine")
    st.markdown("Select a pre-configured transaction scenario or input custom parameters to evaluate the risk score instantly.")
    
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### 📋 Transaction Input Parameters")
        
        preset_choice = st.selectbox(
            "Quick Select Demo Preset Scenario:",
            ["Custom Input"] + (df_presets['scenario'].tolist() if df_presets is not None else ["Custom Input"])
        )
        
        # Default values
        default_amt = 120.0
        default_cat = 'shopping_net'
        default_dist = 12.0
        default_avg_amt = 50.0
        default_hour = 14
        default_age = 42
        default_gender = 'M'
        default_pop = 45000
        
        if preset_choice != "Custom Input" and df_presets is not None:
            preset_row = df_presets[df_presets['scenario'] == preset_choice].iloc[0]
            default_amt = float(preset_row.get('amt', preset_row.get('amount', 120.0)))
            default_cat = str(preset_row.get('category', preset_row.get('merchant_category', 'shopping_net')))
            default_dist = float(preset_row.get('distance_km', preset_row.get('distance_from_home_km', 12.0)))
            default_avg_amt = float(preset_row.get('customer_avg_amount_30d', 50.0))
            default_hour = int(preset_row.get('hour_of_day', 14))
            default_age = int(preset_row.get('age', 42))
            default_gender = str(preset_row.get('gender', 'M'))
            default_pop = int(preset_row.get('city_pop', 45000))

        with st.form("risk_engine_form"):
            c1, c2 = st.columns(2)
            with c1:
                amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=default_amt, step=10.0)
                customer_avg_amount_30d = st.number_input("Customer 30-Day Avg Amount ($)", min_value=1.0, max_value=10000.0, value=default_avg_amt, step=5.0)
                category = st.selectbox("Merchant Category", ['grocery_pos', 'entertainment', 'gas_transport', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home', 'misc_net', 'misc_pos'], index=['grocery_pos', 'entertainment', 'gas_transport', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home', 'misc_net', 'misc_pos'].index(default_cat) if default_cat in ['grocery_pos', 'entertainment', 'gas_transport', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home', 'misc_net', 'misc_pos'] else 0)
                gender = st.selectbox("Cardholder Gender", ['M', 'F'], index=0 if default_gender == 'M' else 1)
                
            with c2:
                hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, default_hour)
                distance_km = st.number_input("Distance to Merchant (km)", 0.0, 10000.0, default_dist)
                age = st.slider("Customer Age", 18, 90, default_age)
                city_pop = st.number_input("City Population", 1000, 5000000, default_pop, step=5000)
                
            submit_btn = st.form_submit_button("🔍 Calculate Risk Score", use_container_width=True)

    with col_output:
        st.markdown("### 🎯 Live Risk Assessment & Explainable AI")
        
        # Build input dict
        amount_to_avg_ratio = amt / (customer_avg_amount_30d if customer_avg_amount_30d > 0 else 1)
        is_night_transaction = 1 if (1 <= hour_of_day <= 5) else 0
        
        txn_dict = {
            'amt': amt,
            'amount': amt,
            'distance_km': distance_km,
            'distance_from_home_km': distance_km,
            'hour_of_day': hour_of_day,
            'age': age,
            'customer_avg_amount_30d': customer_avg_amount_30d,
            'amount_to_avg_ratio': amount_to_avg_ratio,
            'city_pop': city_pop,
            'category': category,
            'merchant_category': category,
            'gender': gender,
            'is_night_transaction': is_night_transaction
        }
        
        # Transform via extract_sparkov_features to guarantee matching columns
        input_raw_df = pd.DataFrame([txn_dict])
        input_feat_df = extract_sparkov_features(input_raw_df)[req_cols]
        input_proc = preprocessor.transform(input_feat_df)
        raw_prob = float(model.predict_proba(input_proc)[0, 1])
        
        # Get explanation
        exp = explain_transaction_risk(txn_dict, pipeline_bundle, raw_prob)
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = exp['risk_score_pct'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Probability (%)", 'font': {'size': 18, 'color': '#94a3b8'}},
            number = {'suffix': "%", 'font': {'size': 36, 'color': exp['risk_color']}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': exp['risk_color']},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk Badge & Action Recommendation
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 15px;">
            <span class="badge-{exp['risk_color']}">Status: {exp['risk_level']}</span>
        </div>
        <div class="action-box">
            System Recommendation:<br><span style="font-size: 1.1rem; color: #f8fafc;">{exp['recommended_action']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🧠 Explainable AI: Key Decision Drivers")
        
        if exp['positive_drivers']:
            st.markdown("**Risk-Increasing Drivers:**")
            for d in exp['positive_drivers']:
                st.markdown(f"""
                <div class="driver-box">
                    <strong>⚠️ {d['factor']} ({d['category']})</strong><br>
                    <span style="font-size: 0.9rem; color: #cbd5e1;">{d['detail']}</span>
                </div>
                """, unsafe_allow_html=True)

        if exp['mitigating_factors']:
            st.markdown("**Risk-Mitigating Factors:**")
            for m in exp['mitigating_factors']:
                st.markdown(f"""
                <div class="mitigator-box">
                    <strong>✅ {m['factor']}</strong><br>
                    <span style="font-size: 0.9rem; color: #cbd5e1;">{m['detail']}</span>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 2: BATCH CSV FRAUD SCANNER
# ==========================================
if st.session_state.current_page == "📂 Batch CSV Fraud Scanner":
    st.subheader("📂 Batch Financial Transaction Fraud Scanner")
    st.markdown("Upload a batch CSV file of transactions (Sparkov Kaggle or standard format) or scan the pre-loaded dataset.")
    
    uploaded_file = st.file_uploader("Upload Transaction Batch (CSV)", type=["csv"])
    
    scan_df = None
    if uploaded_file is not None:
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 150:
                st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). To prevent server memory crashes, analysis is limited to the first 250,000 transactions.")
                scan_df = pd.read_csv(uploaded_file, nrows=250000)
            else:
                scan_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded batch CSV containing {len(scan_df):,} records!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif df_transactions is not None:
        st.info("Using pre-loaded Sparkov 50,000 transaction dataset for demonstration.")
        scan_df = df_transactions.copy()
        
    if scan_df is not None:
        run_batch = st.button("🚀 Run Batch Fraud Analysis", use_container_width=True)
        
        if run_batch or 'batch_scored_df' in st.session_state:
            if run_batch or 'batch_scored_df' not in st.session_state:
                with st.spinner("Processing batch predictions with extract_sparkov_features & model..."):
                    try:
                        scan_feat_df = extract_sparkov_features(scan_df)[req_cols]
                        X_batch_proc = preprocessor.transform(scan_feat_df)
                        batch_probs = model.predict_proba(X_batch_proc)[:, 1]
                        
                        scored_df = scan_df.copy()
                        scored_df['fraud_risk_probability'] = np.round(batch_probs, 4)
                        scored_df['predicted_risk_level'] = np.where(batch_probs >= 0.70, 'HIGH RISK - FLAGGED',
                                                            np.where(batch_probs >= 0.30, 'MEDIUM RISK - 2FA', 'LOW RISK'))
                        st.session_state['batch_scored_df'] = scored_df
                    except Exception as e:
                        st.error(f"Error executing batch predictions: {e}")
                        
            if 'batch_scored_df' in st.session_state:
                scored_df = st.session_state['batch_scored_df']
                
                # Calculate Batch Metrics
                total_txns = len(scored_df)
                flagged_count = (scored_df['predicted_risk_level'] == 'HIGH RISK - FLAGGED').sum()
                medium_count = (scored_df['predicted_risk_level'] == 'MEDIUM RISK - 2FA').sum()
                amt_col = 'amt' if 'amt' in scored_df.columns else ('amount' if 'amount' in scored_df.columns else None)
                fraud_val_at_risk = scored_df[scored_df['predicted_risk_level'] == 'HIGH RISK - FLAGGED'][amt_col].sum() if amt_col else 0
                
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Scanned", f"{total_txns:,}")
                with c2:
                    st.metric("Flagged High Risk Fraud", f"{flagged_count:,}", f"{(flagged_count/total_txns)*100:.2f}% of batch")
                with c3:
                    st.metric("Medium Risk (2FA)", f"{medium_count:,}")
                with c4:
                    st.metric("Prevented Fraud Loss ($)", f"${fraud_val_at_risk:,.2f}")
                    
                st.markdown("### 📋 Filtered Scored Batch Table")
                filter_level = st.radio("Filter by Risk Category:", ["All Transactions", "HIGH RISK - FLAGGED", "MEDIUM RISK - 2FA", "LOW RISK"], horizontal=True)
                
                if filter_level != "All Transactions":
                    display_df = scored_df[scored_df['predicted_risk_level'] == filter_level]
                else:
                    display_df = scored_df
                    
                cols_to_show = [c for c in ['trans_num', 'transaction_id', 'cc_num', 'customer_id', 'amt', 'amount', 'category', 'merchant_category', 'fraud_risk_probability', 'predicted_risk_level'] if c in display_df.columns]
                if not cols_to_show:
                    cols_to_show = display_df.columns.tolist()[:7]
                    
                st.dataframe(
                    display_df[cols_to_show],
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv_bytes = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Transactions CSV",
                    data=csv_bytes,
                    file_name="sparkov_scored_transactions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==========================================
# TAB 3: MODEL PERFORMANCE & METRICS
# ==========================================
if st.session_state.current_page == "📊 Model Performance & Metrics":
    st.subheader("📊 Model Evaluation & Imbalanced Data Metrics")
    st.markdown("Compare benchmarked models on Precision-Recall AUC (PR-AUC), ROC-AUC, and adjust decision thresholds dynamically.")
    
    results = metrics_bundle['results']
    
    # Model Comparison Cards
    c1, c2, c3 = st.columns(3)
    for i, (m_name, res) in enumerate(results.items()):
        col = [c1, c2, c3][i]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{m_name}</h3>
                <div class="value" style="color: {'#38bdf8' if m_name==best_model_name else '#94a3b8'};">PR-AUC: {res['pr_auc']:.3f}</div>
                <div style="margin-top: 10px; text-align: left; font-size: 0.9rem;">
                    • <strong title="The percentage of actual fraud cases the model successfully catches. Higher recall = catches more real fraud.">Recall ℹ️:</strong> {res['recall']*100:.1f}%<br>
                    • <strong title="Of the transactions the model flagged as fraud, how many were actually fraud. Higher precision = fewer false alarms.">Precision ℹ️:</strong> {res['precision']*100:.1f}%<br>
                    • <strong title="A balanced score combining Precision and Recall.">F1-Score ℹ️:</strong> {res['f1']:.3f}<br>
                    • <strong title="Overall accuracy of distinguishing fraud from normal.">ROC-AUC ℹ️:</strong> {res['roc_auc']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Precision-Recall & ROC Curve Charts
    col_pr, col_thresh = st.columns([1, 1], gap="large")
    
    with col_pr:
        st.markdown("#### Precision-Recall Curve Comparison")
        fig_pr = go.Figure()
        for m_name, res in results.items():
            fig_pr.add_trace(go.Scatter(
                x=res['recalls_list'], 
                y=res['precisions_list'],
                mode='lines',
                name=f"{m_name} (PR-AUC={res['pr_auc']:.3f})"
            ))
        fig_pr.update_layout(
            title="Precision vs Recall Trade-off Curve",
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision (Positive Predictive Value)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        
    with col_thresh:
        st.markdown("#### Dynamic Decision Threshold Adjuster")
        threshold = st.slider("Select Classification Threshold", 0.10, 0.90, 0.50, step=0.02)
        
        y_test = metrics_bundle['y_test']
        y_prob = np.array(metrics_bundle['y_prob_best'])
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        prec_t = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec_t = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1_t = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        cm_t = confusion_matrix(y_test, y_pred_thresh)
        tn, fp, fn, tp = cm_t.ravel()
        
        st.markdown(f"""
        **Performance at Threshold = {threshold:.2f}:**
        - **Precision:** `{prec_t*100:.1f}%` (Of flagged items, {prec_t*100:.1f}% were real fraud)
        - **Recall:** `{rec_t*100:.1f}%` (Caught {rec_t*100:.1f}% of total fraud)
        - **F1-Score:** `{f1_t:.3f}`
        - **False Positives (Innocent flagged):** `{fp:,}`
        - **False Negatives (Missed fraud):** `{fn:,}`
        """)
        
        st.markdown(f"*(**Plain English Summary:** At this threshold, out of {tn+fp+fn+tp:,} test transactions, the model correctly caught {tp:,} out of {tp+fn:,} real fraud cases.)*")
        
        # Confusion Matrix Heatmap
        fig_cm = px.imshow(
            cm_t, 
            labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
            x=['Legitimate (0)', 'Fraud (1)'],
            y=['Legitimate (0)', 'Fraud (1)'],
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix (Threshold = {threshold:.2f})"
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# TAB 4: FRAUD INSIGHTS & ANALYTICS (EDA)
# ==========================================
if st.session_state.current_page == "🔍 Fraud Insights & Analytics (EDA)":
    st.subheader("🔍 Exploratory Data Analysis & Feature Importance")
    st.markdown("Visualizing key behavioral patterns and feature importance driving fraud predictions across transactions.")
    
    if df_transactions is not None:
        c_eda1, c_eda2 = st.columns(2)
        
        # Extract hour & category safely
        df_eda = df_transactions.copy()
        if 'hour_of_day' not in df_eda.columns:
            if 'trans_date_trans_time' in df_eda.columns:
                df_eda['hour_of_day'] = pd.to_datetime(df_eda['trans_date_trans_time']).dt.hour
            else:
                df_eda['hour_of_day'] = 12

        cat_col = 'category' if 'category' in df_eda.columns else 'merchant_category'
        
        with c_eda1:
            st.markdown("#### Fraud Spikes by Hour of Day")
            hour_df = df_eda.groupby('hour_of_day')['is_fraud'].agg(['count', 'sum']).reset_index()
            hour_df['fraud_rate'] = (hour_df['sum'] / hour_df['count']) * 100
            
            fig_hour = px.bar(
                hour_df, 
                x='hour_of_day', 
                y='fraud_rate',
                labels={'hour_of_day': 'Hour of Day (0-23)', 'fraud_rate': 'Fraud Rate (%)'},
                title="Fraud Incidence Rate by Hour of Day",
                color='fraud_rate',
                color_continuous_scale='Reds'
            )
            fig_hour.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
            st.plotly_chart(fig_hour, use_container_width=True)
            st.caption("Takeaway: Fraud is highest in the late night / early morning hours (1 AM - 5 AM).")

        with c_eda2:
            st.markdown("#### Fraud Rate by Merchant Category")
            cat_df = df_eda.groupby(cat_col)['is_fraud'].agg(['count', 'sum']).reset_index()
            cat_df['fraud_rate'] = (cat_df['sum'] / cat_df['count']) * 100
            cat_df = cat_df.sort_values(by='fraud_rate', ascending=True)
            
            fig_cat = px.bar(
                cat_df, 
                x='fraud_rate', 
                y=cat_col,
                orientation='h',
                labels={cat_col: 'Category', 'fraud_rate': 'Fraud Rate (%)'},
                title="Fraud Incidence Rate by Merchant Category",
                color='fraud_rate',
                color_continuous_scale='Oranges'
            )
            fig_cat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
            st.plotly_chart(fig_cat, use_container_width=True)
            st.caption("Takeaway: Fraud is highly concentrated in online shopping and grocery transactions.")

        st.markdown("---")
        
        st.markdown("#### 🏆 Global Feature Importance Ranking")
        fi_dict = pipeline_bundle.get('feature_importances', {})
        if fi_dict:
            fi_df = pd.DataFrame(list(fi_dict.items())[:12], columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
            fig_fi = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 12 Most Predictive Risk Features",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_fi.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("Takeaway: The amount of the transaction and geographical distance are the strongest predictors of fraud risk.")
