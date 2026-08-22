# 🛡️ AI-Powered Financial Fraud Detection & Risk Analysis System

An end-to-end Machine Learning pipeline and interactive Streamlit web application for real-time credit card and digital wallet fraud detection. This system addresses severe class imbalance using **SMOTE** and provides human-readable Explainable AI (XAI) risk attributions.

---

## 📌 Key Highlights & Features

- **⚡ Real-Time Transaction Risk Engine**: Predicts instant fraud risk probabilities (0-100%) and provides automated business recommendations (`AUTO-APPROVE`, `2FA PASSCODE`, `BLOCK & ALERT`).
- **📂 Sparkov Kaggle Dataset Schema**: Synthesizes and models transactions following the official Sparkov Credit Card Fraud schema (`kartik2112/fraud-detection`).
- **⚖️ Class Imbalance Mitigation (SMOTE)**: Solves the severe 98.3% / 1.67% class imbalance using Synthetic Minority Over-sampling Technique (SMOTE) and cost-sensitive class weighting.
- **🧠 Explainable AI (XAI)**: Generates human-readable local risk driver breakdowns explaining *why* a transaction was flagged, syncing perfectly with the inputs provided.
- **🎨 Premium UI/UX**: Features a highly polished, dark-mode glassmorphism UI with interactive popovers, sidebar navigation, and a blurred cyber-network background.

---

## 🏆 Model Performance Benchmarks

Models evaluated on a 20% stratified test set (10,000 transactions) using **PR-AUC** and **Recall**:

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (SMOTE)** | 25.2% | **90.0%** | 0.393 | 0.9528 | **0.4663** | Selected Baseline |
| **Random Forest** | 46.2% | 86.5% | 0.603 | 0.9575 | 0.4385 | Evaluated |
| **XGBoost Classifier** | **45.8%** | 86.5% | **0.599** | **0.9530** | 0.4342 | Evaluated |

---

## 🖥️ Streamlit Web Application Tabs

The dashboard is navigated via a clean sidebar menu:
1. **🏠 System Overview**: Top-level KPI metrics (PR-AUC, Recall) with interactive tooltips and system architecture details.
2. **🔑 Login**: Secure developer login portal granting access to the fraud analysis tools.
3. **⚡ Live Risk Simulator**: Test single transactions or choose demo presets (*Normal Morning Grocery*, *Suspicious Midnight Shopping*, *High Value Overseas Travel*).
4. **📂 Batch CSV Fraud Scanner**: Drag-and-drop CSV batch upload, real-time risk predictions, summary KPIs (Flagged Count, Fraud Dollars Saved), and filterable CSV export.
5. **📊 Model Performance & Metrics**: Precision-Recall & ROC curves, dynamic threshold slider (0.10 - 0.90), and confusion matrix heatmaps.
6. **🔍 Fraud Insights & Analytics (EDA)**: Interactive Plotly charts of fraud by hour of day (night spikes), merchant category, and feature importances.

---

## 🛠️ Feature Engineering Pipeline

1. **Haversine Distance (`distance_km`)**: Calculates great-circle geographical distance between cardholder `(lat, long)` and merchant `(merch_lat, merch_long)`.
2. **Customer Age (`age`)**: Computed dynamically from cardholder birthdate (`dob`) and transaction timestamp.
3. **Amount-to-Baseline Ratio (`amount_to_avg_ratio`)**: Ratio of current transaction amount (`amt`) vs. customer's historical 30-day average.
4. **Night Time Anomaly (`is_night_transaction`)**: Binary indicator for transactions between 1 AM and 5 AM.

---

## 💻 Local Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/veepurimanogna-ctrl/ai-financial-fraud-detection.git
   cd ai-financial-fraud-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate synthetic dataset & train ML models**:
   ```bash
   python data_generator.py
   python model_pipeline.py
   ```

4. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 📜 Project Structure

```text
├── assets/
│   └── world_map.png             # UI Background assets
├── data/
│   ├── sparkov_fraudTrain.csv    # 50,000 transaction dataset (Kaggle Sparkov schema)
│   └── preset_scenarios.csv      # Demo test scenarios for UI simulator
├── models/
│   ├── fraud_model_pipeline.pkl  # Trained ML pipeline binary (Preprocessor + Classifier)
│   └── metrics.joblib            # Evaluation metrics & test predictions cache
├── app.py                        # Streamlit multi-tab web dashboard application
├── data_generator.py             # Synthetic transaction dataset generator
├── model_pipeline.py             # Preprocessing, feature engineering & model training
├── explainability.py             # Local XAI risk factor attribution module
├── Run_App.bat                   # 1-click Windows batch file launcher
├── requirements.txt              # Required Python packages list
└── README.md                     # Documentation
```
