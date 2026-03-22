import boto3
import yaml
import pandas as pd

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def predict_single_customer():
    # Sample customer — one row from test data (no label, no header)
    # Format: tenure, MonthlyCharges, TotalCharges, SeniorCitizen, ... all features
    df = pd.read_csv("data/test.csv", header=None)
    sample = ",".join(map(str, df.iloc[0, 1:].tolist()))

    client = boto3.client("sagemaker-runtime", region_name=cfg["aws"]["region"])

    response = client.invoke_endpoint(
        EndpointName=cfg["sagemaker"]["endpoint_name"],
        ContentType="text/csv",
        Body=sample
    )

    score = float(response["Body"].read().decode("utf-8").strip())

    print("\n--- Real-time Inference Result ---")
    print(f"Churn probability: {score:.4f}")
    if score >= 0.5:
        print("Prediction: WILL CHURN (high risk)")
    else:
        print("Prediction: WILL NOT CHURN (low risk)")
    print("----------------------------------\n")
    return score

if __name__ == "__main__":
    predict_single_customer()