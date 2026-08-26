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
            'detail': f'The transaction amount (₹{amount:,.2f}) is alarmingly high—{ratio:.1f}x greater than the customer\\'s normal 30-day baseline (₹{avg_amt:,.2f}). Sudden, massive deviations in spending behavior are strong leading indicators of potential account takeover or compromised credentials, severely amplifying the overall risk score.',
            'impact_weight': 0.35,
            'category': 'High Impact'
        })
    elif ratio >= 2.5:
        drivers.append({
            'factor': 'Elevated Transaction Amount',
            'detail': f'The amount requested (₹{amount:,.2f}) is noticeably elevated at {ratio:.1f}x the customer\\'s typical 30-day average (₹{avg_amt:,.2f}). While not necessarily fraudulent on its own, this abnormal deviation flags the transaction for closer scrutiny and moderately increases the risk score.',
            'impact_weight': 0.20,
            'category': 'Medium Impact'
        })
    elif ratio <= 1.2:
        mitigators.append({
            'factor': 'Normal Purchase Amount',
            'detail': f'The transaction amount is highly consistent with the customer\\'s historical 30-day baseline (₹{avg_amt:,.2f}). Predictable, routine spending patterns indicate legitimate account usage, thereby lowering the transaction\\'s overall risk profile.',
            'impact_weight': -0.15
        })

    # 2. Hourly / Daily Velocity Spikes (Removed - Not collected in app)

    # 3. Night & High Risk Merchant Combination
    high_risk_merchants = ['travel', 'shopping_net', 'misc_net']
    if is_night == 1 and merchant in high_risk_merchants:
        drivers.append({
            'factor': 'Late-Night High-Risk Merchant',
            'detail': f'This transaction to a historically high-risk merchant category ({merchant}) was initiated during early morning hours (1 AM - 5 AM). Fraudsters frequently operate during these off-hours to exploit delayed customer awareness, making this a critical risk amplifier.',
            'impact_weight': 0.30,
            'category': 'High Impact'
        })
    elif merchant in high_risk_merchants:
        drivers.append({
            'factor': 'High Risk Merchant Category',
            'detail': f'The merchant type is classified as "{merchant}". Historical network data shows this category suffers from a statistically higher rate of fraudulent chargebacks and stolen card testing, contributing to an elevated risk score.',
            'impact_weight': 0.15,
            'category': 'Medium Impact'
        })

    # 4. Geographic Distance Rule
    if dist > 100:
        drivers.append({
            'factor': 'Long Distance From Home',
            'detail': f'The transaction occurred {dist:,.1f} km away from the customer\\'s registered home address. Without corresponding travel flags, large geographic anomalies heavily suggest the card is being used by an unauthorized third party in a different region.',
            'impact_weight': 0.25,
            'category': 'High Impact'
        })
    elif dist < 20:
        mitigators.append({
            'factor': 'Nearby Merchant Location',
            'detail': f'The transaction occurred locally (only {dist:,.1f} km from the customer\\'s home). Transactions made within a known, familiar geographic radius strongly correlate with legitimate, routine customer behavior, acting as a powerful risk mitigator.',
            'impact_weight': -0.15
        })

    # 5. Location, Card Present, IP Risk, Failed Auth (Removed - Not collected in app)

    # Risk Assessment Categorization
    if risk_score >= 0.70:
        risk_level = "HIGH RISK - FLAGGED"
        risk_color = "#DC3545"
        action = "🛑 BLOCK TRANSACTION & ALERT FRAUD INVESTIGATION TEAM"
    elif risk_score >= 0.30:
        risk_level = "MEDIUM RISK - REVIEW"
        risk_color = "#D4A72C"
        action = "⚠️ PROMPT SECOND-FACTOR AUTHENTICATION (2FA / OTP)"
    else:
        risk_level = "LOW RISK - CLEARED"
        risk_color = "#159447"
        action = "✅ AUTOMATICALLY APPROVE TRANSACTION"

    return {
        'risk_score_pct': round(risk_score * 100, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommended_action': action,
        'positive_drivers': drivers,
        'mitigating_factors': mitigators
    }
