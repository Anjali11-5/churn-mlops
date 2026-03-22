variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "bucket_name" {
  description = "S3 bucket name — must be globally unique. Change the suffix."
  type        = string
  default     = "churn-mlops-bucket-anjali-2026"
}