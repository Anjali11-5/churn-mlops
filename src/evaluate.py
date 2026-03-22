import boto3
import sagemaker
from sagemaker.model import Model
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import mlflow
import yaml
import json
import time
import os

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def deploy_endpoint(model_uri):
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=cfg["aws"]["region"])
    )

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=cfg["aws"]["region"],
        version="1.7-1"
    )

    model = Model(
        image_uri=image_uri,
        model_data=model_uri,
        role=cfg["aws"]["role_arn"],
        sagemaker_session=session,
    )

    print(f"Deploying endpoint: {cfg['sagemaker']['endpoint_name']}")
    
    # delete endpoint if it already exists from a previous run
    try:
        sm_client = boto3.client("sagemaker", region_name=cfg["aws"]["region"])
        sm_client.delete_endpoint(EndpointName=cfg["sagemaker"]["endpoint_name"])
        print("Deleted existing endpoint, redeploying...")
        import time
        time.sleep(10)
    except Exception:
        pass

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=cfg["sagemaker"]["endpoint_instance_type"],
        endpoint_name=cfg["sagemaker"]["endpoint_name"],
    )

    # make sure predictor is not None
    if predictor is None:
        from sagemaker.predictor import Predictor
        predictor = Predictor(
            endpoint_name=cfg["sagemaker"]["endpoint_name"],
            sagemaker_session=session,
        )

    print("Endpoint deployed successfully!")
    return predictor

def evaluate_model(predictor):
    # Load test data — no header, first column is label
    test_df = pd.read_csv(cfg["local"]["test_data"], header=None)
    y_true = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    print(f"Scoring {len(X_test)} test rows...")

    # Score in batches of 100
    predictions = []
    batch_size = 100
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        csv_batch = "\n".join([",".join(map(str, row)) for row in batch])
        response = predictor.predict(
            csv_batch,
            initial_args={"ContentType": "text/csv"}
        )
        scores = [float(x) for x in response.decode("utf-8").strip().split("\n")]
        predictions.extend(scores)

    auc = roc_auc_score(y_true, predictions)
    print(f"\nAUC Score:  {auc:.4f}")
    print(f"Threshold:  {cfg['model']['auc_threshold']}")

    result = {
        "auc": round(auc, 4),
        "threshold": cfg["model"]["auc_threshold"],
        "passed": bool(auc >= cfg["model"]["auc_threshold"])
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to evaluation_results.json")
    return auc

def delete_endpoint():
    sm_client = boto3.client("sagemaker", region_name=cfg["aws"]["region"])
    
    # Delete endpoint
    try:
        sm_client.delete_endpoint(EndpointName=cfg["sagemaker"]["endpoint_name"])
        print(f"Endpoint deleted: {cfg['sagemaker']['endpoint_name']}")
    except Exception as e:
        print(f"Could not delete endpoint: {e}")

    # Delete endpoint config
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=cfg["sagemaker"]["endpoint_name"])
        print(f"Endpoint config deleted: {cfg['sagemaker']['endpoint_name']}")
    except Exception as e:
        print(f"Could not delete endpoint config: {e}")

    # Delete model
    try:
        sm_client.delete_model(ModelName=cfg["sagemaker"]["endpoint_name"])
        print(f"Model deleted: {cfg['sagemaker']['endpoint_name']}")
    except Exception as e:
        print(f"Could not delete model: {e}")

if __name__ == "__main__":
    # Setup MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Read model URI saved by train.py
    if not os.path.exists("model_uri.txt"):
        print("ERROR: model_uri.txt not found.")
        print("You must run train.py first before evaluate.py")
        exit(1)

    with open("model_uri.txt", "r") as f:
        model_uri = f.read().strip()

    if not model_uri:
        print("ERROR: model_uri.txt is empty. Run train.py first.")
        exit(1)

    print(f"Using model: {model_uri}")

    # Deploy endpoint
    predictor = deploy_endpoint(model_uri)

    # Evaluate model
    auc = evaluate_model(predictor)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metric("test_auc", auc)

    # Check threshold
    threshold = cfg["model"]["auc_threshold"]
    if auc < threshold:
        print(f"\nFAILED: AUC {auc:.4f} is below threshold {threshold}")
        print("Deleting endpoint to avoid AWS charges...")
        delete_endpoint()
        exit(1)
    else:
        print(f"\nPASSED: AUC {auc:.4f} >= threshold {threshold}")
        print(f"Endpoint is live: {cfg['sagemaker']['endpoint_name']}")