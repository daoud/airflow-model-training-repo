import pytest
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from joblib import load
# Import necessary functions from your training script
from bank_campaign_model_training import (
    encode_categorical, preprocess_features, apply_bucketing,
    train_model, get_classification_report,
    # load_model_artifact, # No longer needed for this specific test
)
import pandas as pd
from imblearn.over_sampling import RandomOverSampler # Added for oversampling in test

@pytest.fixture
def dummy_data():
    # Prepare dummy data for testing
    data = {
        'age': [30, 40, 50, 60, 35, 45, 55, 65], # Increased data points for better split
        'job': ['admin.', 'technician', 'self-employed', 'management', 'services', 'retired', 'student', 'unemployed'],
        'marital': ['married', 'single', 'married', 'divorced', 'single', 'married', 'single', 'divorced'],
        'education': ['university.degree', 'basic.9y', 'high.school', 'basic.4y', 'professional.course', 'unknown', 'illiterate', 'university.degree'],
        'default': ['no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes'],
        'housing': ['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no'],
        'loan': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
        'contact': ['cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'telephone'],
        'month': ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri', 'mon', 'tue', 'wed'],
        'duration': [200, 300, 400, 500, 250, 350, 450, 550],
        'campaign': [1, 2, 3, 4, 1, 2, 3, 4],
        'pdays': [20, 30, 40, 999, 15, 25, 35, 999],
        'previous': [1, 2, 3, 4, 1, 2, 3, 4],
        'poutcome': ['success', 'failure', 'nonexistent', 'failure', 'success', 'nonexistent', 'failure', 'success'],
        'emp.var.rate': [1.1, 2.2, 3.3, 4.4, 1.0, 2.0, 3.0, 4.0],
        'cons.price.idx': [90.1, 90.2, 90.3, 90.4, 90.0, 90.1, 90.2, 90.3],
        'cons.conf.idx': [-30.1, -30.2, -30.3, -30.4, -30.0, -30.1, -30.2, -30.3],
        'euribor3m': [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
        'nr.employed': [5000, 6000, 7000, 8000, 5500, 6500, 7500, 8500],
        'y': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'] # Balanced for dummy data
    }
    return pd.DataFrame(data)

def test_preprocess_features(dummy_data):
    df = dummy_data
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                        'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    # The number of columns changes due to 'duration' and 'pdays' being dropped and 'pdays_bucketed' added
    # Original 21 columns - 2 dropped + 1 added = 20 columns after bucketing.
    # After dropping 'y', X should have 19 columns.
    assert X.shape == (8, 19) # Updated shape for 8 rows of dummy data
    assert y.shape == (8,) # Updated shape for 8 rows of dummy data
    
def test_data_loading(dummy_data):
    df = dummy_data
    assert len(df.columns) == 21

def test_categorical_encoding(dummy_data):
    # Test categorical encoding function
    df = dummy_data
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    encoded_df = encode_categorical(df, categorical_cols)
    assert encoded_df.shape == df.shape  # Check if the shape is preserved after encoding
    
def test_get_classification_report(dummy_data):
    df = dummy_data
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                        'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)

    # Apply oversampling, similar to main script, to ensure consistent data for training
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Train a model directly within the test, instead of loading from GCS
    # This makes the test self-contained and independent of external artifacts
    model = train_model('xgboost', X_resampled, y_resampled) # Use the train_model function
    
    report = get_classification_report(model, X_resampled, y_resampled) # Use X_resampled, y_resampled for consistency
    assert isinstance(report, dict)
    assert '0' in report.keys()
    assert '1' in report.keys() # Ensure both classes are in the report

def test_train_model(dummy_data):
    df = dummy_data
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                        'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)

    # Apply oversampling for training data consistency
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    model = train_model('xgboost', X_resampled, y_resampled)
    assert isinstance(model, Pipeline)

