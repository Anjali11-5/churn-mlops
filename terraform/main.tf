terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ─── S3 Bucket ───────────────────────────────────────────────
resource "aws_s3_bucket" "churn_bucket" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = {
    Project     = "churn-mlops"
    Environment = "dev"
  }
}

resource "aws_s3_bucket_versioning" "churn_bucket_versioning" {
  bucket = aws_s3_bucket.churn_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ─── IAM Role for SageMaker ──────────────────────────────────
resource "aws_iam_role" "sagemaker_role" {
  name = "churn-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Project = "churn-mlops"
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_full" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# ─── Outputs ─────────────────────────────────────────────────
output "bucket_name" {
  value       = aws_s3_bucket.churn_bucket.bucket
  description = "Copy this into config.yaml -> aws.bucket_name"
}

output "sagemaker_role_arn" {
  value       = aws_iam_role.sagemaker_role.arn
  description = "Copy this into config.yaml -> aws.role_arn"
}