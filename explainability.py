import numpy as np
import pandas as pd

def explain_transaction_risk(txn_dict, pipeline_bundle, risk_score):
    """
    Computes local feature risk contributions and natural language explanation factors
    for a single financial transaction.
    
    Parameters:
        txn_dict (dict): Feature key-value dictionary for the transaction.
        pipeline_bundle (dict): Loaded pipeline containing model and preprocessor.
        risk_score (float): Model predicted fraud probability (0.0 to 1.0).
        
    Returns:
        dict: Breakdown of positive risk drivers, mitigating factors, and summary text.
    """
    drivers = []
    mitigators = []
    
    amount = float(txn_dict.get('amount', 0))
    avg_amt = float(txn_dict.get('customer_avg_amount_30d', 1))
    ratio = float(txn_dict.get('amount_to_avg_ratio', amount / (avg_amt if avg_amt > 0 else 1)))
    v1h = int(txn_dict.get('velocity_1h', 1))
    v24h = int(txn_dict.get('velocity_24h', 1))
    is_night = int(txn_dict.get('is_night_transaction', 0))
    merchant = str(txn_dict.get('merchant_category', ''))
    device = str(txn_dict.get('device_type', ''))
    card_present = int(txn_dict.get('is_card_present', 0))
    location = str(txn_dict.get('location_country', ''))
    dist = float(txn_dict.get('distance_from_home_km', 0))
    failed = int(txn_dict.get('failed_attempts_last_24h', 0))
    ip_risk = float(txn_dict.get('ip_risk_score', 0))
    
    # 1. Transaction Amount Anomaly
    if ratio >= 4.0:
        drivers.append({
            'factor': 'Unusual Transaction Amount',
            'detail': f'Amount (${amount:,.2f}) is {ratio:.1f}x higher than customer 30-day baseline (${avg_amt:,.2f})',
            'impact_weight': 0.35,
            'category': 'High Impact'
        })
    elif ratio >= 2.5:
        drivers.append({
            'factor': 'Elevated Transaction Amount',
            'detail': f'Amount (${amount:,.2f}) is {ratio:.1f}x customer 30-day average (${avg_amt:,.2f})',
            'impact_weight': 0.20,
            'category': 'Medium Impact'
        })
    elif ratio <= 1.2:
        mitigators.append({
            'factor': 'Normal Purchase Amount',
            'detail': f'Amount is consistent with customer 30-day baseline (${avg_amt:,.2f})',
            'impact_weight': -0.15
        })

    # 2. Hourly / Daily Velocity Spikes (Removed - Not collected in app)

    # 3. Night & High Risk Merchant Combination
    high_risk_merchants = ['travel', 'shopping_net', 'misc_net']
    if is_night == 1 and merchant in high_risk_merchants:
        drivers.append({
            'factor': 'Late-Night High-Risk Merchant',
            'detail': f'Transaction to {merchant} initiated during early morning hours (1 AM - 5 AM)',
            'impact_weight': 0.30,
            'category': 'High Impact'
        })
    elif merchant in high_risk_merchants:
        drivers.append({
            'factor': 'High Risk Merchant Category',
            'detail': f'Merchant type is classified as {merchant}',
            'impact_weight': 0.15,
            'category': 'Medium Impact'
        })

    # 4. Geographic Distance Rule
    if dist > 100:
        drivers.append({
            'factor': 'Long Distance From Home',
            'detail': f'Transaction occurred {dist:,.1f} km away from customer home',
            'impact_weight': 0.25,
            'category': 'High Impact'
        })
    elif dist < 20:
        mitigators.append({
            'factor': 'Nearby Merchant Location',
            'detail': f'Transaction occurred locally ({dist:,.1f} km from home)',
            'impact_weight': -0.15
        })

    # 5. Location, Card Present, IP Risk, Failed Auth (Removed - Not collected in app)

    # Risk Assessment Categorization
    if risk_score >= 0.20:
        risk_level = "HIGH RISK - FLAGGED"
        risk_color = "red"
        action = "🛑 BLOCK TRANSACTION & ALERT FRAUD INVESTIGATION TEAM"
    elif risk_score >= 0.10:
        risk_level = "MEDIUM RISK - REVIEW"
        risk_color = "orange"
        action = "⚠️ PROMPT SECOND-FACTOR AUTHENTICATION (2FA / OTP)"
    else:
        risk_level = "LOW RISK - CLEARED"
        risk_color = "green"
        action = "✅ AUTOMATICALLY APPROVE TRANSACTION"

    return {
        'risk_score_pct': round(risk_score * 100, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommended_action': action,
        'positive_drivers': drivers,
        'mitigating_factors': mitigators
    }
