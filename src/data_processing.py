import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


def extract_date_features(df):
    """ Extracts date-related features from the 'TransactionStartTime' column.
    Args:
        df (pd.DataFrame): DataFrame containing 'TransactionStartTime' column.
    Returns:
        pd.DataFrame: DataFrame with new date features.
    """
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

    df['transaction_year'] = df['TransactionStartTime'].dt.year
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek

    return df


def build_aggregate_features(df):
    """
    Builds aggregate features from the 'TransactionId' and 'Amount' columns, grouped by 'CustomerId'.
    The following features are computed: transaction count, total amount, average amount, and standard deviation of amount.

    Args:
        df (pd.DataFrame): DataFrame containing 'TransactionId', 'Amount', and 'CustomerId' columns.

    Returns:
        pd.DataFrame: DataFrame with the computed aggregate features merged in.
    """
    df = df.copy()

    agg = df.groupby('CustomerId').agg(
        transaction_count=('TransactionId', 'count'),
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        std_amount=('Amount', 'std')
    ).reset_index()

    df = df.merge(agg, on='CustomerId', how='left')
    return df


def clean_data(df):
    """
    Removes unnecessary columns from the DataFrame to reduce dimensionality and improve
    computational efficiency.

    The following columns are removed:

    - TransactionId
    - BatchId
    - AccountId
    - SubscriptionId
    - TransactionStartTime
    - CurrencyCode
    - CountryCode

    Args:
        df (pd.DataFrame): DataFrame containing the columns to be cleaned.

    Returns:
        pd.DataFrame: DataFrame with the specified columns removed.
    """
    df = df.copy()
    drop_cols = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
        'TransactionStartTime', 'CurrencyCode', 'CountryCode'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

def safe_log_transform(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def build_preprocessor():
    """
    Builds a preprocessor for the dataset using pipelines for numerical and categorical features.

    The preprocessor applies the following transformations:
    - For numeric features: imputes missing values with the median, applies a log transformation, and scales them using standard scaling.
    - For categorical features: imputes missing values with the most frequent category and encodes them using one-hot encoding.

    Returns:
        ColumnTransformer: A scikit-learn ColumnTransformer object that applies the specified
        transformations to the columns of the input DataFrame.
    """

    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy']
    numeric_features = [
        'Amount', 'Value',
        'transaction_count', 'total_amount', 'avg_amount', 'std_amount',
        'transaction_year', 'transaction_month', 'transaction_day', 'transaction_hour', 'transaction_dayofweek'
    ]

    # log_transformer = FunctionTransformer(np.log1p)
    log_transformer = FunctionTransformer(safe_log_transform)


    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor
