import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import boto3
import yaml
import os

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def load_data():
    df = pd.read_csv(cfg["local"]["raw_data"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def clean_data(df):
    # Drop customerID — not useful for prediction
    df = df.drop(columns=["customerID"])

    # TotalCharges has some spaces — fix and convert to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert target: Yes -> 1, No -> 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Convert SeniorCitizen is already 0/1, leave it
    # Binary Yes/No columns -> 1/0
    yes_no_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # One-hot encode remaining categorical columns
    cat_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convert all boolean columns to int (get_dummies creates bool in newer pandas)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")
    return df

def split_and_save(df):
    # SageMaker XGBoost needs target column FIRST
    target = "Churn"
    feature_cols = [c for c in df.columns if c != target]
    df = df[[target] + feature_cols]

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=df[target]
    )

    os.makedirs("data", exist_ok=True)
    train_df.to_csv(cfg["local"]["train_data"], index=False, header=False)
    test_df.to_csv(cfg["local"]["test_data"], index=False, header=False)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df

def upload_to_s3():
    s3 = boto3.client("s3", region_name=cfg["aws"]["region"])
    bucket = cfg["aws"]["bucket_name"]

    files = [
        (cfg["local"]["train_data"], cfg["s3"]["train_key"]),
        (cfg["local"]["test_data"],  cfg["s3"]["test_key"]),
        (cfg["local"]["test_data"],  cfg["s3"]["batch_input_key"]),
    ]

    for local_path, s3_key in files:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded {local_path} -> s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    split_and_save(df)
    upload_to_s3()
    print("Preprocessing complete!")