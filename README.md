# 🛡️ AI-Powered Financial Fraud Detection & Risk Analysis System

🔗 **[Try the live app here](https://ai-financial-fraud-detection-vrmx4fnmax9yyq6cdptyik.streamlit.app/)**

An end-to-end Machine Learning pipeline and interactive Streamlit web application for real-time credit card and digital wallet fraud detection. This system addresses severe class imbalance using **SMOTE** and provides human-readable Explainable AI (XAI) risk attributions.

---

## 👥 Team

This project was built by a 4-member team:

| Name |  Focus Area |
|------|------------|
| **[Tanuja Neela Devi]** | Data Collection & Cleaning |
| **[Veepuri Manogna]** | EDA & Visualization |
| **[Ajay Kumar]** | Feature Engineering & Model Training |
| **[Sravani]** | Model Evaluation & Dashboard |

---

## 📌 Key Highlights & Features

- **⚡ Real-Time Transaction Risk Engine**: Predicts instant fraud risk probabilities (0-100%) and provides automated business recommendations (`AUTO-APPROVE`, `2FA PASSCODE`, `BLOCK & ALERT`).
- **⚖️ Class Imbalance Mitigation (SMOTE)**: Solves the severe 98.3% / 1.67% class imbalance using Synthetic Minority Over-sampling Technique (SMOTE) and cost-sensitive class weighting.
- **🧠 Explainable AI (XAI)**: Generates human-readable local risk driver breakdowns explaining *why* a transaction was flagged, syncing perfectly with the inputs provided.
- **🎨 Premium UI/UX**: Features a highly polished, dark-mode glassmorphism UI with interactive popovers, sidebar navigation, and a blurred cyber-network background.

---

## 📂 Dataset

This project uses a **synthetic dataset** generated to match the schema of the real-world 
[Sparkov Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) 
on Kaggle. The generator (`data_generator.py`) creates 50,000 transactions with realistic 
column structure (merchant, category, location, job, amount) and a rule-based fraud labeling 
system, preserving a realistic class imbalance (~1.6% fraud) without requiring the full 
1.3M-row original download.

---

## 🏆 Model Performance Benchmarks

Models evaluated on a 20% stratified test set (10,000 transactions) using **PR-AUC** and **Recall**:

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (SMOTE)** | 25.2% | **90.0%** | 0.393 | 0.9528 | **0.4663** | Selected Baseline |
| **Random Forest** | 46.2% | 86.5% | 0.603 | 0.9575 | 0.4385 | Evaluated |
| **XGBoost Classifier** | **45.8%** | 86.5% | **0.599** | **0.9530** | 0.4342 | Evaluated |

### 🧠 Model Selection Rationale

Three models were trained and compared — Logistic Regression, Random Forest, and XGBoost — 
using SMOTE to address class imbalance. **Logistic Regression was selected as the primary 
model** based on PR-AUC score (0.466) and highest recall (90.0%), prioritizing catching real 
fraud cases over minimizing false alarms, since missing fraud is typically costlier than a 
false positive requiring manual review.

---

## ⚠️ Known Limitations

- **Login page is for demonstration only** — it does not perform real authentication or 
  issue actual API keys.
- **Dataset is synthetic**, generated to match the real Sparkov schema rather than using 
  the original downloaded data.
- **The explainability engine is a separate rule-based layer**, not derived directly from 
  the trained model's internal weights — it interprets the same input features to produce 
  human-readable reasoning.

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

## 💻 Setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/veepurimanogna-ctrl/ai-financial-fraud-detection.git
   cd ai-financial-fraud-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**:
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
