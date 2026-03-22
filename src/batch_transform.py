import boto3
import sagemaker
import pandas as pd
import yaml
import time

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def prepare_batch_input():
    # Remove the label column (first column) for batch transform
    test_df = pd.read_csv(cfg["local"]["test_data"], header=None)
    test_df = test_df.iloc[:, 1:]  # drop first column (Churn label)
    
    batch_input_path = "data/batch_input.csv"
    test_df.to_csv(batch_input_path, index=False, header=False)
    
    # Upload to S3
    s3 = boto3.client("s3", region_name=cfg["aws"]["region"])
    bucket = cfg["aws"]["bucket_name"]
    s3.upload_file(batch_input_path, bucket, cfg["s3"]["batch_input_key"])
    print(f"Uploaded batch input to s3://{bucket}/{cfg['s3']['batch_input_key']}")

def run_batch_transform():
    with open("model_uri.txt", "r") as f:
        model_uri = f.read().strip()

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=cfg["aws"]["region"])
    )

    bucket = cfg["aws"]["bucket_name"]
    input_s3  = f"s3://{bucket}/{cfg['s3']['batch_input_key']}"
    output_s3 = f"s3://{bucket}/{cfg['s3']['batch_output_prefix']}"

    sm_client = boto3.client("sagemaker", region_name=cfg["aws"]["region"])

    model_name = f"churn-batch-model-{int(time.time())}"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=cfg["aws"]["region"],
        version="1.7-1"
    )

    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_uri,
        },
        ExecutionRoleArn=cfg["aws"]["role_arn"],
    )

    transformer = sagemaker.transformer.Transformer(
        model_name=model_name,
        instance_count=1,
        instance_type=cfg["sagemaker"]["instance_type"],
        output_path=output_s3,
        sagemaker_session=session,
    )

    print(f"Starting batch transform...")
    print(f"Input:  {input_s3}")
    print(f"Output: {output_s3}")

    transformer.transform(
        data=input_s3,
        content_type="text/csv",
        split_type="Line",
        wait=True,
    )

    print(f"Batch transform complete!")
    print(f"Results at: {output_s3}")

if __name__ == "__main__":
    prepare_batch_input()
    run_batch_transform()
