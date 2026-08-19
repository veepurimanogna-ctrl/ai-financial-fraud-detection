import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def generate_sparkov_dataset(num_records=50000, random_seed=42, output_dir="data"):
    """
    Generates a dataset following the exact Sparkov Kaggle Credit Card Fraud schema
    (kartik2112/fraud-detection dataset format).
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_records} synthetic transactions in Sparkov Kaggle format...")

    num_customers = 2000
    cc_numbers = [1000000000000000 + i * 987654 for i in range(1, num_customers + 1)]
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    genders = ['M', 'F']
    jobs = ['Software Engineer', 'Accountant', 'Teacher', 'Nurse', 'Manager', 'Sales Associate', 'Driver', 'Consultant', 'Doctor', 'Executive']
    
    # Assign customer profiles
    cust_profiles = {}
    for cc in cc_numbers:
        lat = np.random.uniform(25.0, 48.0)
        long = np.random.uniform(-122.0, -70.0)
        dob = datetime(1955, 1, 1) + timedelta(days=int(np.random.uniform(0, 18000)))
        cust_profiles[cc] = {
            'first': np.random.choice(first_names),
            'last': np.random.choice(last_names),
            'gender': np.random.choice(genders),
            'street': f"{np.random.randint(100, 9999)} Main St",
            'city': 'Metropolis',
            'state': 'NY',
            'zip': np.random.randint(10000, 99999),
            'lat': lat,
            'long': long,
            'city_pop': int(np.random.lognormal(mean=9.5, sigma=1.2)),
            'job': np.random.choice(jobs),
            'dob': dob.strftime('%Y-%m-%d'),
            'dob_dt': dob,
            'avg_amt': float(np.random.lognormal(mean=3.8, sigma=0.6))
        }

    categories = [
        'grocery_pos', 'entertainment', 'gas_transport', 'shopping_net', 
        'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 
        'travel', 'kids_pets', 'home', 'misc_net', 'misc_pos'
    ]
    cat_weights = [0.22, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01]

    assigned_cc = np.random.choice(cc_numbers, size=num_records)
    merchant_cats = np.random.choice(categories, size=num_records, p=cat_weights)
    merchants = [f"fraud_{np.random.choice(last_names)}, {c.replace('_', ' ').title()}" for c in merchant_cats]

    # Timestamps over 30 days
    start_date = datetime(2026, 7, 1, 0, 0, 0)
    random_seconds = np.random.randint(0, 30 * 24 * 3600, size=num_records)
    trans_dates = [start_date + timedelta(seconds=int(s)) for s in random_seconds]

    df = pd.DataFrame({
        'trans_date_trans_time': [t.strftime('%Y-%m-%d %H:%M:%S') for t in trans_dates],
        'cc_num': assigned_cc,
        'merchant': merchants,
        'category': merchant_cats,
        'trans_num': [f"txn_{1000000 + i}" for i in range(num_records)],
        'unix_time': [int(t.timestamp()) for t in trans_dates]
    })

    # Sort chronologically for velocity
    df['trans_date_dt'] = trans_dates
    df = df.sort_values(by=['cc_num', 'trans_date_dt']).reset_index(drop=True)
    df['trans_date_trans_time'] = df['trans_date_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['unix_time'] = df['trans_date_dt'].astype('int64') // 10**9

    # Add customer profile info
    df['first'] = df['cc_num'].map(lambda c: cust_profiles[c]['first'])
    df['last'] = df['cc_num'].map(lambda c: cust_profiles[c]['last'])
    df['gender'] = df['cc_num'].map(lambda c: cust_profiles[c]['gender'])
    df['street'] = df['cc_num'].map(lambda c: cust_profiles[c]['street'])
    df['city'] = df['cc_num'].map(lambda c: cust_profiles[c]['city'])
    df['state'] = df['cc_num'].map(lambda c: cust_profiles[c]['state'])
    df['zip'] = df['cc_num'].map(lambda c: cust_profiles[c]['zip'])
    df['lat'] = df['cc_num'].map(lambda c: cust_profiles[c]['lat'])
    df['long'] = df['cc_num'].map(lambda c: cust_profiles[c]['long'])
    df['city_pop'] = df['cc_num'].map(lambda c: cust_profiles[c]['city_pop'])
    df['job'] = df['cc_num'].map(lambda c: cust_profiles[c]['job'])
    df['dob'] = df['cc_num'].map(lambda c: cust_profiles[c]['dob'])
    
    # Generate amounts & merchant coordinates
    avg_amts = df['cc_num'].map(lambda c: cust_profiles[c]['avg_amt']).values
    amt_mults = np.random.lognormal(mean=0.0, sigma=0.5, size=num_records)
    df['amt'] = np.round(avg_amts * amt_mults, 2)
    df['amt'] = np.maximum(df['amt'], 1.0)

    # Merchant lat/long (near customer lat/long with some outliers)
    lat_offsets = np.random.normal(loc=0.0, scale=0.15, size=num_records)
    long_offsets = np.random.normal(loc=0.0, scale=0.15, size=num_records)
    df['merch_lat'] = np.round(df['lat'] + lat_offsets, 5)
    df['merch_long'] = np.round(df['long'] + long_offsets, 5)

    # Calculate distance km
    df['distance_km'] = np.round(haversine_np(df['lat'], df['long'], df['merch_lat'], df['merch_long']), 2)

    # Feature engineering for ground truth rules
    df['hour_of_day'] = df['trans_date_dt'].dt.hour
    df['is_night_transaction'] = np.where((df['hour_of_day'] >= 1) & (df['hour_of_day'] <= 5), 1, 0)
    df['amount_to_avg_ratio'] = np.round(df['amt'] / avg_amts, 2)

    # Ground Truth Fraud Labels
    risk_score = np.zeros(num_records)
    risk_score += np.where(df['amount_to_avg_ratio'] > 5.0, 0.35, 0.0)
    risk_score += np.where(df['amount_to_avg_ratio'] > 3.0, 0.20, 0.0)
    
    high_risk_cats = ['shopping_net', 'travel', 'misc_net']
    risk_score += np.where((df['is_night_transaction'] == 1) & (df['category'].isin(high_risk_cats)), 0.30, 0.0)
    risk_score += np.where(df['distance_km'] > 150.0, 0.35, 0.0)
    risk_score += np.where(df['amt'] > 1000.0, 0.25, 0.0)

    prob = 1.0 / (1.0 + np.exp(-(risk_score - 0.70) * 4.5))
    target_fraud_count = int(num_records * 0.016)
    top_indices = np.argsort(prob)[-target_fraud_count:]

    is_fraud = np.zeros(num_records, dtype=int)
    is_fraud[top_indices] = 1

    noise_mask = np.random.rand(num_records) < 0.001
    is_fraud = np.where(noise_mask, 1 - is_fraud, is_fraud)
    df['is_fraud'] = is_fraud

    # Clean intermediate columns
    df_output = df.drop(columns=['trans_date_dt', 'distance_km', 'hour_of_day', 'is_night_transaction', 'amount_to_avg_ratio'])

    output_path = os.path.join(output_dir, "sparkov_fraudTrain.csv")
    df_output.to_csv(output_path, index=False)
    print(f"Saved Sparkov Kaggle format dataset to {output_path} (Fraud count: {is_fraud.sum()})")

    # Save preset scenarios for UI
    create_sparkov_presets(output_dir)
    return df_output

def create_sparkov_presets(output_dir="data"):
    samples = [
        {
            'scenario': '🟢 Normal Morning Grocery Purchase',
            'amt': 18.50,
            'category': 'grocery_pos',
            'merchant': 'fraud_Smith, Grocery Pos',
            'distance_km': 2.4,
            'hour_of_day': 10,
            'is_night_transaction': 0,
            'customer_avg_amount_30d': 25.00,
            'amount_to_avg_ratio': 0.74,
            'city_pop': 45000,
            'age': 38,
            'gender': 'F'
        },
        {
            'scenario': '🔴 Suspicious Midnight Online Shopping Spike',
            'amt': 1850.00,
            'category': 'shopping_net',
            'merchant': 'fraud_Kaggle, Shopping Net',
            'distance_km': 480.0,
            'hour_of_day': 3,
            'is_night_transaction': 1,
            'customer_avg_amount_30d': 45.00,
            'amount_to_avg_ratio': 41.1,
            'city_pop': 12000,
            'age': 54,
            'gender': 'M'
        },
        {
            'scenario': '🚨 High Value Overseas Travel Transfer',
            'amt': 3200.00,
            'category': 'travel',
            'merchant': 'fraud_Airlines, Travel',
            'distance_km': 1250.0,
            'hour_of_day': 2,
            'is_night_transaction': 1,
            'customer_avg_amount_30d': 120.00,
            'amount_to_avg_ratio': 26.6,
            'city_pop': 850000,
            'age': 29,
            'gender': 'F'
        }
    ]
    preset_df = pd.DataFrame(samples)
    preset_path = os.path.join(output_dir, "preset_scenarios.csv")
    preset_df.to_csv(preset_path, index=False)

if __name__ == "__main__":
    generate_sparkov_dataset(num_records=50000)
