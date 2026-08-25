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

# --- Splash Screen & Persistent Branding ---
if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = False

if not st.session_state.welcome_shown:
    st.markdown('''<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    
    html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #142B44 !important;
    }
    p, span, div, label {
    font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] > div, .metric-card .value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stSidebar"] * { color: #F5F8FC !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .stAppHeader { display: none !important; }
    body, .stApp { background-color: #0F2740 !important; }
    .splash-starfield { position: fixed; top: -25%; left: -25%; width: 150%; height: 150%; z-index: 0; pointer-events: none; animation: swirl 180s linear infinite; }
    @keyframes swirl { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .splash-star { position: absolute; background: #D4A72C; border-radius: 50%; box-shadow: 0 0 4px 1px rgba(212, 167, 44,0.7); animation: twinkle linear infinite; }
    @keyframes twinkle { 0%, 100% { opacity: 0.15; } 50% { opacity: 1; } }
    @keyframes zi { from { opacity:0; transform:scale(0.85); } to { opacity:1; transform:scale(1); } }
    .splash-container { position: relative; z-index: 10; display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 75vh; text-align: center; animation: zi 0.7s cubic-bezier(0.2,0.8,0.3,1) both; }
    /* Animate the start button below as well */
    .stButton > button { animation: zi 0.7s cubic-bezier(0.2,0.8,0.3,1) both; }
    .shield-glow-container { position: relative; display: inline-block; }
    .shield-glow { position: absolute; width: 700px; height: 700px; background: radial-gradient(circle, rgba(212, 167, 44, 0.3) 0%, rgba(212, 167, 44, 0.1) 30%, transparent 70%); top: 50%; left: 50%; transform: translate(-50%, -50%); border-radius: 50%; animation: pulseGlow 6s infinite ease-in-out; z-index: -1; pointer-events: none; }
    @keyframes pulseGlow { 0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; } 50% { transform: translate(-50%, -50%) scale(1.15); opacity: 1; } }
    .gradient-divider { width: 60%; height: 2px; background: linear-gradient(90deg, transparent 0%, #FFD700 50%, transparent 100%); margin: 20px auto; opacity: 1.0; box-shadow: 0 0 10px rgba(255,215,0,0.8); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .splash-container h1 { color: #FFD700 !important; }
    .splash-container h3 { color: #FFFFFF !important; }
    
    /* Sidebar Starfield */
    [data-testid="stSidebarContent"] { position: relative; overflow: hidden; }
    .sidebar-starfield {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 0;
        pointer-events: none;
    }
</style>
<div class="splash-starfield"><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 4.4%; left: 28.4%; animation-duration: 2.33s; animation-delay: 1.1s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 87.6%; left: 10.3%; animation-duration: 2.63s; animation-delay: 0.04s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 50.5%; left: 4.5%; animation-duration: 2.3s; animation-delay: 0.97s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 23.2%; left: 58.6%; animation-duration: 3.21s; animation-delay: 0.01s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 69.0%; left: 34.7%; animation-duration: 2.23s; animation-delay: 1.44s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 10.9%; left: 11.3%; animation-duration: 3.27s; animation-delay: 0.91s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 72.1%; left: 53.5%; animation-duration: 3.46s; animation-delay: 0.57s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 81.6%; left: 61.4%; animation-duration: 3.29s; animation-delay: 0.87s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 6.4%; left: 23.9%; animation-duration: 2.43s; animation-delay: 0.12s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 11.7%; left: 28.7%; animation-duration: 2.95s; animation-delay: 0.55s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 22.1%; left: 27.6%; animation-duration: 3.4s; animation-delay: 0.97s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 18.4%; left: 72.0%; animation-duration: 2.25s; animation-delay: 0.57s;"></div><div class="splash-star" style="width: 2.0px; height: 2.0px; top: 63.4%; left: 55.5%; animation-duration: 3.03s; animation-delay: 1.26s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 24.0%; left: 5.1%; animation-duration: 2.47s; animation-delay: 0.4s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 92.5%; left: 86.1%; animation-duration: 2.47s; animation-delay: 0.98s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 89.8%; left: 46.0%; animation-duration: 2.4s; animation-delay: 0.37s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 27.2%; left: 58.1%; animation-duration: 3.35s; animation-delay: 0.6s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 97.8%; left: 50.9%; animation-duration: 2.14s; animation-delay: 0.07s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 62.2%; left: 78.0%; animation-duration: 2.63s; animation-delay: 0.1s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 97.6%; left: 52.8%; animation-duration: 3.46s; animation-delay: 1.29s;"></div></div>
<div class="splash-container">
<div class="shield-glow-container" style="display:flex; flex-direction:column; align-items:center;">
<div class="shield-glow"></div>
<svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px; z-index: 1;">
  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" fill="url(#shield-grad)" stroke="#FCEBB8" stroke-width="0.5"/>
  <path d="M9 12l2 2 4-4" stroke="#0F2740" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <defs>
    <linearGradient id="shield-grad" x1="12" y1="2" x2="12" y2="22" gradientUnits="userSpaceOnUse">
      <stop stop-color="#F4D879"/>
      <stop offset="1" stop-color="#B9862A"/>
    </linearGradient>
  </defs>
</svg>
<h1 style="color: #FFD700; font-weight: 800; font-size: 5.5rem; margin-bottom: 0; margin-top: 0; text-shadow: 0 0 30px rgba(212,167,44,0.8); line-height: 1.1;">AUREVIA SHIELD</h1>
</div>
<div class="gradient-divider"></div>
<h3 style="color: #FFFFFF; margin-top: 0; margin-bottom: 40px; font-weight: 400; font-size: 1.6rem;">AI Financial Fraud Detection & Risk Analysis System</div>
</div>''', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Let's Start", use_container_width=True, type="primary"):
            st.session_state.welcome_shown = True
            st.rerun()
            
    st.stop()

# Hide sidebar if not logged in
if not st.session_state.get('logged_in', False):
    st.markdown('''
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
    </style>
    ''', unsafe_allow_html=True)

# Top-Left Persistent Brand Mark
st.markdown('''
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    
    /* :wght@600&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');
    html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #142B44 !important;
    }
    p, span, div, label {
    font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] > div, .metric-card .value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    }
    .top-left-brand {
    position: fixed !important;
    top: 15px !important;
    left: 70px !important;
    z-index: 9999 !important;
    color: #142B44 !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 900 !important;
    font-size: 2.2rem !important;
    letter-spacing: 1.5px !important;
    pointer-events: none !important;
    text-shadow: 2px 2px 4px rgba(20, 43, 68, 0.1) !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    }
    .top-left-brand span {
    font-size: 2.8rem !important;
    }
    
</style>
<div class="top-left-brand"><span>🛡️</span> AUREVIA SHIELD</div>
''', unsafe_allow_html=True)

# --- Premium Light Dashboard Theme ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    
    /* Fix Sidebar text colors */
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    /* Sidebar UI Overhaul */
    [data-testid="stSidebar"] {
    background-color: #0F2740 !important; /* Lighter Navy Blue to match reference */
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    [data-testid="stSidebarNav"] { display: none !important; }
    div[role="radiogroup"] { gap: 4px !important; position: relative; z-index: 10; padding: 0 10px; }
    div[role="radiogroup"] label {
    padding: 14px 20px !important;
    border-radius: 8px !important;
    margin-bottom: 2px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    }
    /* Safely hide the Streamlit radio circles and inputs without hiding the text container */
    div[role="radiogroup"] label input,
    div[role="radiogroup"] label > div:not(:has(p)) { 
        display: none !important; 
    }
    div[role="radiogroup"] label p {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    }
    div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #A88222 0%, #8A6413 100%) !important;
    box-shadow: 0 4px 10px rgba(168, 130, 34, 0.15) !important;
    }
    div[role="radiogroup"] label:has(input:checked) * {
    color: #FFFFFF !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    div[role="radiogroup"] label:not(:has(input:checked)):hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    }
    /* Global Typography Reset */
    html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    }
    /* Fix Input Field Visibility */
    div[data-testid="stNumberInput"] > div > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stSelectbox"] > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stNumberInput"] button,
    div[data-testid="stSelectbox"] * {
        background-color: transparent !important;
        color: #142B44 !important;
    }
    /* Fix File Uploader Visibility */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #142B44 !important;
    }
    /* Beautiful Dashboard Background */
    .stApp {
    background: linear-gradient(135deg, #FDFDFD 0%, #FDF7E2 100%) !important;
    }
    /* Elegant Serif Headings */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: #142B44 !important;
    }
    /* Gradient Main Dashboard Title */
    [data-testid="stHeader"] { background-color: transparent !important; }
    div.block-container > div:first-child [data-testid="stMarkdownContainer"] h1 {
    background: linear-gradient(90deg, #142B44, #D4A72C);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    color: transparent !important;
    text-shadow: 0 4px 15px rgba(212, 167, 44, 0.15);
    }
    p, span, div, label {
    font-family: 'Inter', sans-serif;
    }
    /* Numeric Font for Metrics */
    [data-testid="stMetricValue"] > div, .metric-card .value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #142B44 !important;
    }
    /* 🔘 Premium Sidebar Navigation Radio Buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 12px 16px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(212, 167, 44, 0.4) !important;
    box-shadow: 0 4px 12px rgba(212, 167, 44, 0.15) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #D4A72C !important;
    font-weight: 700 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:not(:has(input:checked)):hover {
    background-color: rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:not(:has(input:checked)) p {
    color: #F8FAFC !important;
    font-weight: 500 !important;
    }
    
    /* Main Page Radio Buttons & DataFrames (High Visibility) */
    [data-testid="stAppViewContainer"] div[role="radiogroup"] label p {
        color: #0F2740 !important;
        font-weight: 600 !important;
    }
    [data-testid="stDataFrame"] * {
        color: #0F2740 !important;
    }
    .metric-card .text-value {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    word-break: normal !important;
    line-height: 1.2 !important;
    }
    /* 🛡️ Premium Elevated Metric Cards */
    .metric-card {
    background-color: #0F2740 !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(212, 167, 44, 0.3) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    text-align: center !important;
    box-shadow: 0 8px 30px rgba(15, 39, 64, 0.06) !important;
    transition: transform 0.3s cubic-bezier(0.2,0.8,0.2,1), box-shadow 0.3s ease !important;
    /* Force all cards to be the exact same size and vertically center content */
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    height: 190px !important;
    }
    .metric-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 15px 40px rgba(15, 39, 64, 0.12) !important;
    border: 1px solid rgba(212, 167, 44, 0.5) !important;
    }
    .metric-card .card-title {
    color: #D4A72C !important;
    font-size: 0.95rem !important;
    margin-bottom: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    }
    .metric-card .value {
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    word-break: normal !important;
    line-height: 1.2 !important;
    }
    .metric-card .subtext {
    color: #94A3B8 !important;
    font-size: 0.9rem !important;
    margin-top: 8px !important;
    font-family: 'Inter', sans-serif !important;
    }
    /* 🚨 Neon Risk Badges */
    .badge-high {
    background-color: rgba(220, 53, 69, 0.15);
    color: #DC3545;
    border: 1px solid #DC3545;
    padding: 8px 18px;
    border-radius: 4px;
    font-weight: 800;
    font-size: 1.2rem;
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 2px;
    box-shadow: 0 0 15px rgba(220, 53, 69, 0.4);
    text-shadow: 0 0 8px rgba(220, 53, 69, 0.5);
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
    background-color: rgba(21, 148, 71, 0.15);
    color: #159447;
    border: 1px solid #159447;
    padding: 8px 18px;
    border-radius: 4px;
    font-weight: 800;
    font-size: 1.2rem;
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 2px;
    box-shadow: 0 0 15px rgba(21, 148, 71, 0.3);
    }
    /* 📊 Explanation & Terminal Boxes */
    .driver-box {
    background-color: rgba(220, 53, 69, 0.08);
    border-left: 3px solid #DC3545;
    padding: 12px;
    margin-bottom: 10px;
    font-family: monospace;
    color: #B91C1C;
    }
    .mitigator-box {
    background-color: rgba(21, 148, 71, 0.08);
    border-left: 3px solid #159447;
    padding: 12px;
    margin-bottom: 10px;
    font-family: monospace;
    color: #047857;
    }
    /* ⚡ Action Box */
    .action-box {
    background: rgba(212, 167, 44, 0.05);
    border: 1px dashed #D4A72C;
    border-radius: 4px;
    padding: 16px;
    margin-top: 15px;
    text-align: center;
    font-weight: 700;
    color: #D4A72C;
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
    st.session_state.current_page = "💠 Dashboard"

# --- Full-Page Custom Navigation ---
if not st.session_state.logged_in:
    nav_options = ["💠 Dashboard", "🔑 Login"]
else:
    nav_options = [
        "💠 Dashboard",
        "⚡ Live Risk Simulator", 
        "📂 Batch CSV Fraud Scanner", 
        "📊 Model Performance & Metrics", 
        "🔍 Fraud Insights & Analytics (EDA)",
        "🚪 Logout"
    ]

# Render Navigation in Sidebar
st.sidebar.markdown("""
<div class="sidebar-starfield">
    <div class="splash-star" style="width: 1.8px; height: 1.8px; top: 4.4%; left: 28.4%; animation-duration: 2.33s; animation-delay: 1.1s;"></div>
    <div class="splash-star" style="width: 1.8px; height: 1.8px; top: 87.6%; left: 10.3%; animation-duration: 2.63s; animation-delay: 0.04s;"></div>
    <div class="splash-star" style="width: 1.6px; height: 1.6px; top: 50.5%; left: 4.5%; animation-duration: 2.3s; animation-delay: 0.97s;"></div>
    <div class="splash-star" style="width: 1.8px; height: 1.8px; top: 23.2%; left: 58.6%; animation-duration: 3.21s; animation-delay: 0.01s;"></div>
    <div class="splash-star" style="width: 1.9px; height: 1.9px; top: 69.0%; left: 34.7%; animation-duration: 2.23s; animation-delay: 1.44s;"></div>
    <div class="splash-star" style="width: 1.7px; height: 1.7px; top: 10.9%; left: 11.3%; animation-duration: 3.27s; animation-delay: 0.91s;"></div>
    <div class="splash-star" style="width: 1.9px; height: 1.9px; top: 72.1%; left: 53.5%; animation-duration: 3.46s; animation-delay: 0.57s;"></div>
    <div class="splash-star" style="width: 1.8px; height: 1.8px; top: 81.6%; left: 61.4%; animation-duration: 3.29s; animation-delay: 0.87s;"></div>
    <div class="splash-star" style="width: 1.9px; height: 1.9px; top: 6.4%; left: 23.9%; animation-duration: 2.43s; animation-delay: 0.12s;"></div>
    <div class="splash-star" style="width: 1.6px; height: 1.6px; top: 11.7%; left: 28.7%; animation-duration: 2.95s; animation-delay: 0.55s;"></div>
    <div class="splash-star" style="width: 1.7px; height: 1.7px; top: 22.1%; left: 27.6%; animation-duration: 3.4s; animation-delay: 0.97s;"></div>
    <div class="splash-star" style="width: 1.8px; height: 1.8px; top: 18.4%; left: 72.0%; animation-duration: 2.25s; animation-delay: 0.57s;"></div>
    <div class="splash-star" style="width: 2.0px; height: 2.0px; top: 63.4%; left: 55.5%; animation-duration: 3.03s; animation-delay: 1.26s;"></div>
    <div class="splash-star" style="width: 1.9px; height: 1.9px; top: 24.0%; left: 5.1%; animation-duration: 2.47s; animation-delay: 0.4s;"></div>
    <div class="splash-star" style="width: 1.6px; height: 1.6px; top: 92.5%; left: 86.1%; animation-duration: 2.47s; animation-delay: 0.98s;"></div>
    <div class="splash-star" style="width: 1.7px; height: 1.7px; top: 89.8%; left: 46.0%; animation-duration: 2.4s; animation-delay: 0.37s;"></div>
    <div class="splash-star" style="width: 1.8px; height: 1.8px; top: 27.2%; left: 58.1%; animation-duration: 3.35s; animation-delay: 0.6s;"></div>
    <div class="splash-star" style="width: 1.6px; height: 1.6px; top: 97.8%; left: 50.9%; animation-duration: 2.14s; animation-delay: 0.07s;"></div>
    <div class="splash-star" style="width: 1.6px; height: 1.6px; top: 62.2%; left: 78.0%; animation-duration: 2.63s; animation-delay: 0.1s;"></div>
    <div class="splash-star" style="width: 1.7px; height: 1.7px; top: 97.6%; left: 52.8%; animation-duration: 3.46s; animation-delay: 1.29s;"></div>
</div>
<div style="text-align: center; padding: 10px 0 30px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px; z-index: 10; position: relative;">
    <div style="margin-bottom: 15px;">
        <svg width="65" height="75" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 10px rgba(212,167,44,0.3));">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" fill="url(#sidebar-shield)" stroke="#FCEBB8" stroke-width="0.5"/>
            <rect x="9.5" y="9.5" width="5" height="4.5" rx="1" fill="#0F2740"/>
            <path d="M10.5 9.5V8a1.5 1.5 0 013 0v1.5" stroke="#0F2740" stroke-width="1.2"/>
            <defs>
                <linearGradient id="sidebar-shield" x1="12" y1="2" x2="12" y2="22" gradientUnits="userSpaceOnUse">
                <stop stop-color="#F4D879"/>
                <stop offset="1" stop-color="#B9862A"/>
                </linearGradient>
            </defs>
        </svg>
    </div>
    <div style="font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 900; line-height: 1.15; letter-spacing: 1px;">
        <span style="color: #FFFFFF;">AUREVIA</span><br>
        <span style="color: #D4A72C;">SHIELD</span>
    </div>
    <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 15px; font-weight: 500; max-width: 220px; margin-left: auto; margin-right: auto; line-height: 1.5;">
        AI Financial Fraud Detection &<br>Risk Analysis System
    </div>
</div>
""", unsafe_allow_html=True)
selected_page = st.sidebar.radio(
    "Navigation", 
    nav_options, 
    index=nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0,
    label_visibility="collapsed"
)

# Handle Logout immediately
if selected_page == "🚪 Logout":
    st.session_state.logged_in = False
    st.session_state.current_page = "💠 Dashboard"
    st.rerun()


# Update page state if changed
if selected_page != st.session_state.current_page and selected_page != "🚪 Logout":
    st.session_state.current_page = selected_page
    st.rerun()

# --- Custom Bottom Security Card ---
st.sidebar.markdown('''
<div style="margin-top: 50px; margin-bottom: 30px; padding: 25px 20px; border: 1px solid rgba(212, 167, 44, 0.3); border-radius: 12px; background: rgba(15, 39, 64, 0.4); text-align: center; position: relative; z-index: 10; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <div style="margin-bottom: 12px;">
        <svg width="24" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke="#D4A72C" stroke-width="1.5"/>
            <rect x="9" y="10" width="6" height="5" rx="1" stroke="#D4A72C" stroke-width="1.2"/>
            <path d="M10 10V8a2 2 0 014 0v2" stroke="#D4A72C" stroke-width="1.2"/>
        </svg>
    </div>
    <div style="color: #D4A72C; font-weight: 800; font-size: 1.15rem; line-height: 1.3; margin-bottom: 15px; font-family: 'Inter', sans-serif;">Stop fraud before<br>it reaches you.</div>
    <div style="color: #E2E8F0; font-size: 0.75rem; line-height: 1.6; font-weight: 400; opacity: 0.9;">Real-time detection.<br>Smart decisions.<br>Secure future.</div>
</div>
''', unsafe_allow_html=True)


# ==========================================
# TAB OVERVIEW: SYSTEM OVERVIEW
# ==========================================
if st.session_state.current_page == "💠 Dashboard":

    st.markdown('''<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    
    /* :wght@600&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');
    html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #142B44 !important;
    }
    p, span, div, label {
    font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] > div, .metric-card .value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    }
    .page-starfield { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 0; pointer-events: none; }
    .splash-star { position: absolute; background: #D4A72C; border-radius: 50%; box-shadow: 0 0 4px 1px rgba(212, 167, 44,0.7); animation: twinkle linear infinite; }
    @keyframes twinkle { 0%, 100% { opacity: 0.15; } 50% { opacity: 1; } }
    
</style>
<div class="page-starfield"><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 4.4%; left: 28.4%; animation-duration: 2.33s; animation-delay: 1.1s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 87.6%; left: 10.3%; animation-duration: 2.63s; animation-delay: 0.04s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 50.5%; left: 4.5%; animation-duration: 2.3s; animation-delay: 0.97s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 23.2%; left: 58.6%; animation-duration: 3.21s; animation-delay: 0.01s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 69.0%; left: 34.7%; animation-duration: 2.23s; animation-delay: 1.44s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 10.9%; left: 11.3%; animation-duration: 3.27s; animation-delay: 0.91s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 72.1%; left: 53.5%; animation-duration: 3.46s; animation-delay: 0.57s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 81.6%; left: 61.4%; animation-duration: 3.29s; animation-delay: 0.87s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 6.4%; left: 23.9%; animation-duration: 2.43s; animation-delay: 0.12s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 11.7%; left: 28.7%; animation-duration: 2.95s; animation-delay: 0.55s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 22.1%; left: 27.6%; animation-duration: 3.4s; animation-delay: 0.97s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 18.4%; left: 72.0%; animation-duration: 2.25s; animation-delay: 0.57s;"></div><div class="splash-star" style="width: 2.0px; height: 2.0px; top: 63.4%; left: 55.5%; animation-duration: 3.03s; animation-delay: 1.26s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 24.0%; left: 5.1%; animation-duration: 2.47s; animation-delay: 0.4s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 92.5%; left: 86.1%; animation-duration: 2.47s; animation-delay: 0.98s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 89.8%; left: 46.0%; animation-duration: 2.4s; animation-delay: 0.37s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 27.2%; left: 58.1%; animation-duration: 3.35s; animation-delay: 0.6s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 97.8%; left: 50.9%; animation-duration: 2.14s; animation-delay: 0.07s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 62.2%; left: 78.0%; animation-duration: 2.63s; animation-delay: 0.1s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 97.6%; left: 52.8%; animation-duration: 3.46s; animation-delay: 1.29s;"></div></div>''', unsafe_allow_html=True)
    # --- Header Section ---
    st.title("AI Financial Fraud Detection & Risk Analysis System")
    st.caption("Real-Time Machine Learning Pipeline for Financial Transaction Risk Scoring, Class Imbalance Mitigation, & Explainable AI")
    
    st.markdown("**How this works:** This app analyzes transaction patterns (like location, time, and purchase history) to flag potentially fraudulent activity before it is approved.")
    st.markdown("---")
    
    # --- Top Banner Stats ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Primary Model</div>
            <div class="text-value" style="color: #D4A72C;">{best_model_name}</div>
            <div class="subtext">SMOTE Resampled</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        best_pr = metrics_bundle['results'][best_model_name]['pr_auc']
        best_roc = metrics_bundle['results'][best_model_name]['roc_auc']
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Model PR-AUC</div>
            <div class="value" style="color: #34D399;">{best_pr:.3f}</div>
            <div class="subtext">ROC-AUC: {best_roc:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    
    with col_m3:
        best_rec = metrics_bundle['results'][best_model_name]['recall']
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Fraud Recall Rate</div>
            <div class="value" style="color: #FBBF24;">{best_rec*100:.1f}%</div>
            <div class="subtext">Caught Fraud Cases</div>
        </div>
        """, unsafe_allow_html=True)

    
    with col_m4:
        total_len = len(df_transactions) if df_transactions is not None else 50000
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Dataset Size</div>
            <div class="value" style="color: #D8B4FE;">{total_len:,}</div>
            <div class="subtext">Synthetic data matching real Sparkov schema</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #D4A72C; margin-bottom: 30px; font-weight: 800; font-size: 3.8rem; text-shadow: 0 0 20px rgba(212, 167, 44, 0.6); letter-spacing: -1px;'>Stop fraud before it reaches you.</h1>", unsafe_allow_html=True)
    
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
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("Access Login Portal →", type="primary", use_container_width=True):
            st.session_state.current_page = "🔑 Login"
            st.rerun()

# ==========================================
# TAB LOGIN: LOGIN / API ACCESS
# ==========================================
if st.session_state.current_page == "🔑 Login":

    st.markdown('''<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    
    /* :wght@600&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');
    html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #142B44 !important;
    }
    p, span, div, label {
    font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] > div, .metric-card .value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    }
    .page-starfield { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 0; pointer-events: none; }
    .splash-star { position: absolute; background: #D4A72C; border-radius: 50%; box-shadow: 0 0 4px 1px rgba(212, 167, 44,0.7); animation: twinkle linear infinite; }
    @keyframes twinkle { 0%, 100% { opacity: 0.15; } 50% { opacity: 1; } }
    
</style>
<div class="page-starfield"><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 4.4%; left: 28.4%; animation-duration: 2.33s; animation-delay: 1.1s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 87.6%; left: 10.3%; animation-duration: 2.63s; animation-delay: 0.04s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 50.5%; left: 4.5%; animation-duration: 2.3s; animation-delay: 0.97s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 23.2%; left: 58.6%; animation-duration: 3.21s; animation-delay: 0.01s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 69.0%; left: 34.7%; animation-duration: 2.23s; animation-delay: 1.44s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 10.9%; left: 11.3%; animation-duration: 3.27s; animation-delay: 0.91s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 72.1%; left: 53.5%; animation-duration: 3.46s; animation-delay: 0.57s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 81.6%; left: 61.4%; animation-duration: 3.29s; animation-delay: 0.87s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 6.4%; left: 23.9%; animation-duration: 2.43s; animation-delay: 0.12s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 11.7%; left: 28.7%; animation-duration: 2.95s; animation-delay: 0.55s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 22.1%; left: 27.6%; animation-duration: 3.4s; animation-delay: 0.97s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 18.4%; left: 72.0%; animation-duration: 2.25s; animation-delay: 0.57s;"></div><div class="splash-star" style="width: 2.0px; height: 2.0px; top: 63.4%; left: 55.5%; animation-duration: 3.03s; animation-delay: 1.26s;"></div><div class="splash-star" style="width: 1.9px; height: 1.9px; top: 24.0%; left: 5.1%; animation-duration: 2.47s; animation-delay: 0.4s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 92.5%; left: 86.1%; animation-duration: 2.47s; animation-delay: 0.98s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 89.8%; left: 46.0%; animation-duration: 2.4s; animation-delay: 0.37s;"></div><div class="splash-star" style="width: 1.8px; height: 1.8px; top: 27.2%; left: 58.1%; animation-duration: 3.35s; animation-delay: 0.6s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 97.8%; left: 50.9%; animation-duration: 2.14s; animation-delay: 0.07s;"></div><div class="splash-star" style="width: 1.6px; height: 1.6px; top: 62.2%; left: 78.0%; animation-duration: 2.63s; animation-delay: 0.1s;"></div><div class="splash-star" style="width: 1.7px; height: 1.7px; top: 97.6%; left: 52.8%; animation-duration: 3.46s; animation-delay: 1.29s;"></div></div>''', unsafe_allow_html=True)
    st.markdown('''
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    
    /* :wght@600&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');
    html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: #142B44 !important;
    }
    p, span, div, label {
    font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] > div, .metric-card .value, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    }
    /* Target the column containing the anchor */
    /* Apply SVG background to the entire page behind the login form */
    [data-testid="stAppViewContainer"]:has(.login-bg-anchor) {
    background-color: #F5F8FC !important;
    }
    
    div[data-testid="stColumn"]:has(.login-bg-anchor) {
    background-color: #0F2740 !important;
    border-radius: 20px;
    padding: 40px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
</style>
''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div class='login-bg-anchor'></div>", unsafe_allow_html=True)
        st.markdown('''
        <style>
    
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-top: -10px;
                font-family: 'Inter', sans-serif;
            }
            .login-logo {
                margin-bottom: 15px;
                filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.4));
                background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%);
                border-radius: 12px;
                padding: 12px;
            }
            .login-title {
                color: #FFFFFF;
                font-size: 1.9rem;
                font-weight: 700;
                margin-bottom: 5px;
            }
            .login-subtitle {
                color: #E2E8F0;
                font-size: 0.95rem;
                margin-bottom: 25px;
            }
    
            .login-tabs {
                display: flex;
                width: 100%;
                gap: 15px;
                margin-bottom: 25px;
            }
            .login-tab {
                flex: 1;
                text-align: center;
                padding: 14px;
                border-radius: 30px;
                font-weight: 600;
                font-size: 0.95rem;
                cursor: pointer;
            }
            .tab-active {
                background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%);
                color: #FFFFFF;
                border: 1px solid #3B82F6;
                box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
            }
            .tab-inactive {
                background: transparent;
                color: #F8FAFC;
                border: 1px solid #334155;
            }
    
            .google-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                padding: 14px;
                border-radius: 30px;
                background: transparent;
                border: 1px solid #334155;
                color: #FFFFFF;
                font-weight: 700;
                font-size: 1rem;
                cursor: pointer;
                margin-bottom: 25px;
                gap: 12px;
                transition: background 0.2s;
            }
            .google-btn:hover { background: rgba(255,255,255,0.05); }
    
            .login-divider {
                display: flex;
                align-items: center;
                text-align: center;
                color: #CBD5E1;
                font-size: 0.85rem;
                margin-bottom: 20px;
                font-family: monospace;
                letter-spacing: 0.5px;
            }
            .login-divider::before, .login-divider::after {
                content: '';
                flex: 1;
                border-bottom: 1px solid #334155;
            }
            .login-divider:not(:empty)::before { margin-right: 1em; }
            .login-divider:not(:empty)::after { margin-left: 1em; }
    
            /* Streamlit Inputs Styling */
            div[data-testid="stTextInput"] {
                margin-bottom: 10px;
            }
            div[data-testid="stTextInput"] label p {
                text-transform: uppercase;
                color: #E2E8F0 !important;
                font-weight: 800;
                font-size: 0.8rem !important;
                letter-spacing: 1px;
                margin-bottom: -2px;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"] {
                background-color: #FFFFFF !important;
                border: 2px solid #CBD5E1 !important;
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
                border: 2px solid #D4A72C !important;
                box-shadow: 0 0 0 2px rgba(212, 167, 44, 0.2) !important;
            }
            div[data-testid="stTextInput"] input {
                background-color: transparent !important;
                color: #0F2740 !important;
                padding: 14px 16px !important;
                font-family: monospace !important;
                font-size: 1.05rem !important;
            }
            div[data-testid="stTextInput"] input::placeholder {
                color: #94A3B8 !important;
            }
    
            /* Main Login Button Override */
            div[data-testid="stButton"] button {
                width: 100% !important;
                background-color: #FFFFFF !important;
                color: #000000 !important;
                border: none !important;
                border-radius: 30px !important;
                padding: 12px 15px !important;
                font-weight: 800 !important;
                font-size: 1.05rem !important;
                margin-top: 15px !important;
                transition: transform 0.2s ease, box-shadow 0.2s ease !important;
            }
            div[data-testid="stButton"] button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255,255,255,0.2) !important;
                color: #000000 !important;
                border: none !important;
            }
            div[data-testid="stButton"] button p {
                font-size: 1.05rem !important;
            }
    
</style>
        
<div class="login-container">
<div class="login-logo" style="background: transparent; padding: 0;">
<svg width="50" height="58" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 10px rgba(212,167,44,0.3));">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" fill="url(#login-shield)" stroke="#FCEBB8" stroke-width="0.5"/>
<rect x="9.5" y="9.5" width="5" height="4.5" rx="1" fill="#0B1121"/>
<path d="M10.5 9.5V8a1.5 1.5 0 013 0v1.5" stroke="#0B1121" stroke-width="1.2"/>
<defs>
<linearGradient id="login-shield" x1="12" y1="2" x2="12" y2="22" gradientUnits="userSpaceOnUse">
<stop stop-color="#F4D879"/>
<stop offset="1" stop-color="#B9862A"/>
</linearGradient>
</defs>
</svg>
</div>
<div class="login-title">Aurevia Access Portal</div>


</div>
        ''', unsafe_allow_html=True)
        
        st.text_input("EMAIL ADDRESS", placeholder="you@example.com")
        st.text_input("PASSWORD", type="password", placeholder="Enter password")
        
        st.markdown('''
            <style>
                button:has(p:contains("hidden_login_trigger")) { display: none !important; }
            </style>
        ''', unsafe_allow_html=True)
        
        btn_main = st.button("Sign In →", use_container_width=True)
        st.markdown("<div style='display: none;'>", unsafe_allow_html=True)
        btn_hidden = st.button("hidden_login_trigger")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if btn_main or btn_hidden:
            st.session_state.logged_in = True
            st.session_state.current_page = "⚡ Live Risk Simulator"
            st.rerun()
            

        
        import streamlit.components.v1 as components
        components.html('''
        <script>
            const hideAndBind = () => {
                const doc = window.parent.document;

                
                let hiddenBtn = null;
                doc.querySelectorAll('button').forEach(btn => {
                    if(btn.innerText.includes('hidden_login_trigger')) {
                        hiddenBtn = btn;
                        // Hide the button's entire container immediately
                        const container = btn.closest('div[data-testid="element-container"]');
                        if (container) container.style.display = 'none';
                        else btn.style.display = 'none';
                    }
                });
                

            };
            
            hideAndBind();
            setTimeout(hideAndBind, 50);
            setTimeout(hideAndBind, 500);
        </script>
        ''', height=0)

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
            title = {'text': "Fraud Risk Probability (%)", 'font': {'size': 18, 'color': '#475569'}},
            number = {'suffix': "%", 'font': {'size': 36, 'color': exp['risk_color']}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': exp['risk_color']},
                'bgcolor': "#F5F8FC",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 10], 'color': 'rgba(21, 148, 71, 0.2)'},
                    {'range': [10, 20], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [20, 100], 'color': 'rgba(220, 53, 69, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#DC3545", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
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
            System Recommendation:<br><span style="font-size: 1.1rem; color: #000000;">{exp['recommended_action']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🧠 Explainable AI: Key Decision Drivers")
        
        if exp['positive_drivers']:
            st.markdown("**Risk-Increasing Drivers:**")
            for d in exp['positive_drivers']:
                st.markdown(f"""
                <div class="driver-box">
                    <strong>⚠️ {d['factor']} ({d['category']})</strong><br>
                    <span style="font-size: 0.9rem; color: #555555;">{d['detail']}</span>
                </div>
                """, unsafe_allow_html=True)

        if exp['mitigating_factors']:
            st.markdown("**Risk-Mitigating Factors:**")
            for m in exp['mitigating_factors']:
                st.markdown(f"""
                <div class="mitigator-box">
                    <strong>✅ {m['factor']}</strong><br>
                    <span style="font-size: 0.9rem; color: #555555;">{m['detail']}</span>
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
            if file_size_mb > 5:
                st.warning(f"⚠️ File size: {file_size_mb:.1f} MB. To ensure real-time performance and prevent server Out-of-Memory crashes, analysis is strictly limited to the first 50,000 transactions.")
            scan_df = pd.read_csv(uploaded_file, nrows=50000)
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
                        scored_df['predicted_risk_level'] = np.where(batch_probs >= 0.20, 'HIGH RISK - FLAGGED',
                                                            np.where(batch_probs >= 0.10, 'MEDIUM RISK - 2FA', 'LOW RISK'))
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
                <div class="card-title">{m_name}</div>
                <div class="value" style="color: {'#D4A72C' if m_name==best_model_name else '#94a3b8'};">PR-AUC: {res['pr_auc']:.3f}</div>
                <div style="margin-top: 10px; text-align: left; font-size: 0.9rem; color: #F8FAFC;">
                    • <strong>Recall:</strong> {res['recall']*100:.1f}%<br>
                    • <strong>Precision:</strong> {res['precision']*100:.1f}%<br>
                    • <strong>F1-Score:</strong> {res['f1']:.3f}<br>
                    • <strong>ROC-AUC:</strong> {res['roc_auc']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.popover("ℹ️ Metrics Info", use_container_width=True):
                st.markdown("**Recall:** The percentage of actual fraud cases the model successfully catches. Higher recall = catches more real fraud.")
                st.markdown("**Precision:** Of the transactions the model flagged as fraud, how many were actually fraud? Higher precision = fewer false alarms.")
                st.markdown("**F1-Score:** A balanced score combining Precision and Recall.")
                st.markdown("**ROC-AUC:** Overall accuracy of distinguishing fraud from normal.")
            
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
            template="plotly_white",
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
            template="plotly_white",
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
                color_continuous_scale='reds'
            )
            fig_hour.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
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
            fig_cat.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
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
            fig_fi.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("Takeaway: The amount of the transaction and geographical distance are the strongest predictors of fraud risk.")
