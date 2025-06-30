import pytest
import pandas as pd
import os
import sys
# Add the src directory to the path for module imports
sys.path.append(os.path.abspath("../src"))

from src.data_processing import (
    extract_date_features,
    build_aggregate_features,
    clean_data,
    build_preprocessor
)


# ✅ Sample minimal dummy data for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'BatchId': ['B1', 'B1', 'B2'],
        'AccountId': ['A1', 'A2', 'A1'],
        'SubscriptionId': ['S1', 'S2', 'S1'],
        'CustomerId': ['C1', 'C2', 'C1'],
        'TransactionStartTime': ['2024-01-01T10:00:00Z', '2024-01-02T15:00:00Z', '2024-01-03T18:30:00Z'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX'],
        'Amount': [1000, 2000, 3000],
        'Value': [1000, 2000, 3000],
        'ProductCategory': ['airtime', 'financial_services', 'airtime'],
        'ChannelId': ['ChannelId_1', 'ChannelId_2', 'ChannelId_1'],
        'ProviderId': ['ProviderId_1', 'ProviderId_2', 'ProviderId_1'],
        'PricingStrategy': [1, 2, 1],
        'FraudResult': [0, 1, 0]
    })
    return data


# ✅ Test date feature extraction
def test_extract_date_features(sample_data):
    df = extract_date_features(sample_data)

    assert 'transaction_year' in df.columns
    assert 'transaction_month' in df.columns
    assert 'transaction_day' in df.columns
    assert 'transaction_hour' in df.columns
    assert 'transaction_dayofweek' in df.columns
    assert df['transaction_year'].iloc[0] == 2024
    assert df['transaction_month'].iloc[0] == 1


# ✅ Test aggregate feature generation
def test_build_aggregate_features(sample_data):
    df = build_aggregate_features(sample_data)

    assert 'transaction_count' in df.columns
    assert 'total_amount' in df.columns
    assert 'avg_amount' in df.columns
    assert 'std_amount' in df.columns

    c1 = df[df['CustomerId'] == 'C1']
    assert c1['transaction_count'].iloc[0] == 2
    assert c1['total_amount'].iloc[0] == 4000


# ✅ Test data cleaning
def test_clean_data(sample_data):
    df = clean_data(sample_data)

    dropped_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'TransactionStartTime', 'CurrencyCode']
    for col in dropped_cols:
        assert col not in df.columns


# ✅ Test preprocessor pipeline
def test_build_preprocessor(sample_data):
    df = extract_date_features(sample_data)
    df = build_aggregate_features(df)
    df = clean_data(df)

    X = df.drop(columns=['FraudResult'])

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    # Check shape — should match (rows, total features)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] > 0  # Should have some features

