import boto3
sm = boto3.client("sagemaker", region_name="us-east-1")
sm.delete_endpoint(EndpointName="churn-mlops-endpoint")
print("Endpoint deleted — no more charges")