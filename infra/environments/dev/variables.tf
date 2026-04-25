variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-west-1"
}

variable "env" {
  description = "Environment name. Locked to 'dev' for this stack."
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "Primary CIDR for the dev VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "alert_email" {
  description = "Email address subscribed to the SNS alerts topic. Confirmation email must be acknowledged after apply."
  type        = string
}

variable "ml_image" {
  description = "Container image for ml-service (e.g. <account>.dkr.ecr.<region>.amazonaws.com/voiceflow-dev/ml-service:sha-xxx)."
  type        = string
}

variable "inference_image" {
  description = "Container image for inference-engine."
  type        = string
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for the HTTPS listener. Set null to expose HTTP-only (NOT recommended)."
  type        = string
  default     = null
}

variable "rds_min_capacity" {
  description = "Aurora Serverless v2 minimum ACU."
  type        = number
  default     = 0.5
}

variable "rds_max_capacity" {
  description = "Aurora Serverless v2 maximum ACU."
  type        = number
  default     = 2.0
}

variable "ml_desired_count" {
  description = "Desired Fargate tasks for ml-service."
  type        = number
  default     = 1
}

variable "ml_min_capacity" {
  description = "Min auto-scaling capacity for ml-service."
  type        = number
  default     = 1
}

variable "ml_max_capacity" {
  description = "Max auto-scaling capacity for ml-service."
  type        = number
  default     = 2
}

variable "inference_desired_count" {
  description = "Desired Fargate tasks for inference-engine."
  type        = number
  default     = 1
}

variable "inference_min_capacity" {
  description = "Min auto-scaling capacity for inference-engine."
  type        = number
  default     = 1
}

variable "inference_max_capacity" {
  description = "Max auto-scaling capacity for inference-engine."
  type        = number
  default     = 2
}

variable "log_retention_days" {
  description = "CloudWatch log retention (days)."
  type        = number
  default     = 30
}
