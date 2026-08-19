import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def extract_sparkov_features(df):
    """
    Extracts & engineers features from the Sparkov Kaggle Credit Card Fraud Dataset.
    Works for both sparkov_fraudTrain.csv and standard transaction files.
    """
    df_feat = pd.DataFrame()
    
    # Amount
    if 'amt' in df.columns:
        df_feat['amt'] = df['amt'].astype(float)
    elif 'amount' in df.columns:
        df_feat['amt'] = df['amount'].astype(float)
    else:
        df_feat['amt'] = 50.0

    # Distance km
    if 'lat' in df.columns and 'long' in df.columns and 'merch_lat' in df.columns and 'merch_long' in df.columns:
        df_feat['distance_km'] = np.round(haversine_np(df['lat'], df['long'], df['merch_lat'], df['merch_long']), 2)
    elif 'distance_from_home_km' in df.columns:
        df_feat['distance_km'] = df['distance_from_home_km'].astype(float)
    else:
        df_feat['distance_km'] = 10.0

    # Timestamp & Hour of Day & Night flag
    if 'trans_date_trans_time' in df.columns:
        dt_col = pd.to_datetime(df['trans_date_trans_time'])
        df_feat['hour_of_day'] = dt_col.dt.hour
    elif 'timestamp' in df.columns:
        dt_col = pd.to_datetime(df['timestamp'])
        df_feat['hour_of_day'] = dt_col.dt.hour
    elif 'hour_of_day' in df.columns:
        df_feat['hour_of_day'] = df['hour_of_day'].astype(int)
    else:
        df_feat['hour_of_day'] = 12

    df_feat['is_night_transaction'] = np.where((df_feat['hour_of_day'] >= 1) & (df_feat['hour_of_day'] <= 5), 1, 0)

    # Customer Age from dob
    if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
        dob_dt = pd.to_datetime(df['dob'])
        trans_dt = pd.to_datetime(df['trans_date_trans_time'])
        df_feat['age'] = (trans_dt - dob_dt).dt.days // 365
    elif 'age' in df.columns:
        df_feat['age'] = df['age'].astype(int)
    else:
        df_feat['age'] = 40

    # Amount to customer baseline average ratio
    cc_id = 'cc_num' if 'cc_num' in df.columns else ('customer_id' if 'customer_id' in df.columns else None)
    if cc_id:
        cust_avg = df.groupby(cc_id)['amt'].transform('mean')
        df_feat['customer_avg_amount_30d'] = np.round(cust_avg, 2)
        df_feat['amount_to_avg_ratio'] = np.round(df_feat['amt'] / np.maximum(cust_avg, 1.0), 2)
    else:
        df_feat['customer_avg_amount_30d'] = 50.0
        df_feat['amount_to_avg_ratio'] = 1.0

    # City population
    df_feat['city_pop'] = df['city_pop'].astype(float) if 'city_pop' in df.columns else 50000.0

    # Categoricals
    df_feat['category'] = df['category'].astype(str) if 'category' in df.columns else (df['merchant_category'].astype(str) if 'merchant_category' in df.columns else 'shopping_pos')
    df_feat['gender'] = df['gender'].astype(str) if 'gender' in df.columns else 'M'

    return df_feat

def train_and_evaluate_fraud_models(data_path="data/sparkov_fraudTrain.csv", model_dir="models"):
    """
    Trains models on the Sparkov Kaggle Credit Card Fraud dataset structure using SMOTE & PR-AUC.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(data_path):
        data_path = "data/transactions.csv"
        
    print(f"Loading Sparkov transaction dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    y = df_raw['is_fraud'].values
    X_feat = extract_sparkov_features(df_raw)
    
    num_features = ['amt', 'distance_km', 'hour_of_day', 'age', 'customer_avg_amount_30d', 'amount_to_avg_ratio', 'city_pop']
    cat_features = ['category', 'gender']
    bin_features = ['is_night_transaction']
    
    X_cols = num_features + cat_features + bin_features
    X = X_feat[X_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} fraud)")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features),
            ('bin', 'passthrough', bin_features)
        ]
    )
    
    print("Fitting preprocessor and transforming training features...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    all_feature_names = num_features + encoded_cat_names + bin_features
    
    print("Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced_subsample', n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    }
    
    results = {}
    best_model_name = None
    best_pr_auc = -1.0
    best_model_obj = None
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == 'Logistic Regression':
            model.fit(X_train_res, y_train_res)
        else:
            model.fit(X_train_proc, y_train)
            
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        pr_auc_score = auc(recalls, precisions)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        print(f"Results for {name}:")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  PR-AUC:    {pr_auc_score:.4f}")
        
        results[name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc_score),
            'confusion_matrix': cm,
            'precisions_list': precisions.tolist(),
            'recalls_list': recalls.tolist(),
            'thresholds_list': thresholds.tolist(),
            'y_prob': y_prob.tolist()
        }
        
        if pr_auc_score > best_pr_auc:
            best_pr_auc = pr_auc_score
            best_model_name = name
            best_model_obj = model
            
    print(f"\n[BEST MODEL] Selected: {best_model_name} with PR-AUC = {best_pr_auc:.4f}")
    
    feature_importances = {}
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        feature_importances = dict(sorted(zip(all_feature_names, [float(x) for x in importances]), key=lambda t: t[1], reverse=True))
    elif hasattr(best_model_obj, 'coef_'):
        importances = np.abs(best_model_obj.coef_[0])
        feature_importances = dict(sorted(zip(all_feature_names, [float(x) for x in importances]), key=lambda t: t[1], reverse=True))
        
    pipeline_bundle = {
        'preprocessor': preprocessor,
        'model': best_model_obj,
        'model_name': best_model_name,
        'feature_names': all_feature_names,
        'num_features': num_features,
        'cat_features': cat_features,
        'bin_features': bin_features,
        'feature_importances': feature_importances
    }
    
    pipeline_save_path = os.path.join(model_dir, "fraud_model_pipeline.pkl")
    joblib.dump(pipeline_bundle, pipeline_save_path)
    
    metrics_save_path = os.path.join(model_dir, "metrics.joblib")
    metrics_bundle = {
        'results': results,
        'best_model_name': best_model_name,
        'X_test': X_test.copy(),
        'y_test': y_test.copy(),
        'y_prob_best': results[best_model_name]['y_prob'],
        'feature_importances': feature_importances
    }
    joblib.dump(metrics_bundle, metrics_save_path)
    print("Saved pipeline and metrics successfully!")
    
    return pipeline_bundle, metrics_bundle

if __name__ == "__main__":
    train_and_evaluate_fraud_models()
