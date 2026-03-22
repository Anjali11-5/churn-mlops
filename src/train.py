import boto3
import sagemaker
from sagemaker.estimator import Estimator
import mlflow
import yaml
import json
import os

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def get_xgboost_image_uri():
    return sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=cfg["aws"]["region"],
        version="1.7-1"
    )

def run_training():
    # Setup MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=cfg["aws"]["region"])
    )
    bucket = cfg["aws"]["bucket_name"]

    # XGBoost hyperparameters
    hyperparams = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "num_round":        100,
        "max_depth":        6,
        "eta":              0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": 3,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)
        print(f"MLflow run ID: {run.info.run_id}")

        estimator = Estimator(
            image_uri=get_xgboost_image_uri(),
            role=cfg["aws"]["role_arn"],
            instance_count=1,
            instance_type=cfg["sagemaker"]["instance_type"],
            output_path=f"s3://{bucket}/{cfg['s3']['model_prefix']}",
            sagemaker_session=session,
            hyperparameters=hyperparams,
        )

        train_input = sagemaker.inputs.TrainingInput(
            f"s3://{bucket}/{cfg['s3']['train_key']}",
            content_type="text/csv"
        )
        test_input = sagemaker.inputs.TrainingInput(
            f"s3://{bucket}/{cfg['s3']['test_key']}",
            content_type="text/csv"
        )

        print("Starting SageMaker training job...")
        estimator.fit({"train": train_input, "validation": test_input})

        # Save model URI for evaluate.py
        model_uri = estimator.model_data
        print(f"Model saved to: {model_uri}")

        with open("model_uri.txt", "w") as f:
            f.write(model_uri)

        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("instance_type", cfg["sagemaker"]["instance_type"])
        print("Training complete! Check MLflow UI at http://127.0.0.1:5000")

if __name__ == "__main__":
    run_training()