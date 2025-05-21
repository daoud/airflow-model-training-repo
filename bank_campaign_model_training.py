# !pip install imblearn
# !pip install xgboost

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
from google.cloud import storage
import json
from google.cloud import bigquery
from datetime import datetime
from sklearn.pipeline import make_pipeline

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket("airflow-ml-ops")

def load_data(path):
    """
    Loads data from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(path, sep=";")

def encode_categorical(df, categorical_cols):
    """
    Encodes categorical columns using LabelEncoder.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_cols (list): A list of column names to encode.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    return df

def preprocess_features(df):
    """
    Separates features (X) and target (y), and scales numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the processed features (X) and target (y).
    """
    X = df.drop('y', axis=1)
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    return X, y

def bucket_pdays(pdays):
    """
    Buckets the 'pdays' column into categories.

    Args:
        pdays (int): The value of the 'pdays' column.

    Returns:
        int: The bucketed category (0, 1, or 2).
    """
    if pdays == 999:
        return 0
    elif pdays <= 30:
        return 1
    else:
        return 2

def apply_bucketing(df):
    """
    Applies bucketing to 'pdays' and drops 'pdays' and 'duration' columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with 'pdays_bucketed' and dropped columns.
    """
    df['pdays_bucketed'] = df['pdays'].apply(bucket_pdays)
    df = df.drop('pdays', axis=1)
    df = df.drop('duration', axis=1)
    return df

def train_model(model_name, x_train, y_train):
    """
    Trains a machine learning model based on the given model name.

    Args:
        model_name (str): The name of the model to train ('logistic', 'random_forest', 'knn', 'xgboost').
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        sklearn.pipeline.Pipeline: The trained model pipeline.
    """
    if model_name == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'xgboost':
        # Added eval_metric to suppress the UserWarning about use_label_encoder
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Invalid model name. Choose from 'logistic', 'random_forest', 'knn', 'xgboost'.")

    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline

def get_classification_report(pipeline, X_test, y_test):
    """
    Generates a classification report for the model.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: A dictionary containing the classification report metrics.
    """
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def save_model_artifact(model_name, pipeline):
    """
    Saves the trained model pipeline as a joblib file and uploads it to GCS.

    Args:
        model_name (str): The name of the model.
        pipeline (sklearn.pipeline.Pipeline): The trained model pipeline.
    """
    artifact_name = model_name + '_model.joblib'
    dump(pipeline, artifact_name)
    # Upload the model artifact to Google Cloud Storage
    model_artifact = bucket.blob('bank_campaign_artifact/' + artifact_name)
    model_artifact.upload_from_filename(artifact_name)
    print(f"Model artifact saved to gs://{bucket.name}/bank_campaign_artifact/{artifact_name}")

def load_model_artifact(file_name):
    """
    Downloads a model artifact from GCS and loads it.
    CORRECTED: The GCS path now matches where models are saved.

    Args:
        file_name (str): The name of the model file (e.g., 'xgboost_model.joblib').

    Returns:
        object: The loaded model pipeline.
    """
    # CORRECTED PATH: Ensure this matches the save_model_artifact path
    blob = bucket.blob("bank_campaign_artifact/" + file_name)
    blob.download_to_filename(file_name)
    print(f"Model artifact downloaded from gs://{bucket.name}/bank_campaign_artifact/{file_name}")
    return load(file_name)

def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    """
    Writes model training metrics to a BigQuery table.

    Args:
        algo_name (str): The name of the algorithm used.
        training_time (datetime): The timestamp of when the model was trained.
        model_metrics (dict): A dictionary containing the model's classification report.
    """
    client = bigquery.Client()
    # Ensure this project ID matches your actual Google Cloud project ID
    table_id = "serious-studio-456210-r8.ml_ops.bank_campaign_model_metrics"
    table = client.get_table(table_id) # Get the table object to ensure it exists

    row = {
        "algo_name": algo_name,
        "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'),
        "model_metrics": json.dumps(model_metrics)
    }

    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)

def main():
    """
    Main function to orchestrate data loading, preprocessing, model training,
    evaluation, and artifact saving.
    """
    input_data_path = "gs://airflow-ml-ops/bank_campaign_data/bank-campaign-training-data.csv"
    model_name = 'xgboost'

    print("Starting model training process...")
    df = load_data(input_data_path)
    print("Data loaded.")

    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    pipeline = train_model(model_name, X_train, y_train)
    accuracy_metrics = get_classification_report(pipeline, X_test, y_test)
    training_time = datetime.now()
    
    write_metrics_to_bigquery(model_name, training_time, accuracy_metrics)
    save_model_artifact(model_name, pipeline)
    print("Model training process completed.")

if __name__ == "__main__":
    main()
