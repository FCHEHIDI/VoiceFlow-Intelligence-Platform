variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-west-1"
}

variable "env" {
  description = "Environment name. Locked to 'prod' for this stack."
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "Primary CIDR for the prod VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "alert_email" {
  description = "Email address subscribed to the SNS alerts topic."
  type        = string
}

variable "ml_image" {
  description = "Container image for ml-service."
  type        = string
}

variable "inference_image" {
  description = "Container image for inference-engine."
  type        = string
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for the HTTPS listener. REQUIRED in production."
  type        = string

  validation {
    condition     = var.acm_certificate_arn != null && can(regex("^arn:aws:acm:", var.acm_certificate_arn))
    error_message = "acm_certificate_arn must be a valid ACM certificate ARN in production."
  }
}

variable "rds_min_capacity" {
  description = "Aurora Serverless v2 minimum ACU."
  type        = number
  default     = 1.0
}

variable "rds_max_capacity" {
  description = "Aurora Serverless v2 maximum ACU."
  type        = number
  default     = 8.0
}

variable "ml_desired_count" {
  description = "Desired Fargate tasks for ml-service."
  type        = number
  default     = 2
}

variable "ml_min_capacity" {
  description = "Min auto-scaling capacity for ml-service."
  type        = number
  default     = 2
}

variable "ml_max_capacity" {
  description = "Max auto-scaling capacity for ml-service."
  type        = number
  default     = 10
}

variable "inference_desired_count" {
  description = "Desired Fargate tasks for inference-engine."
  type        = number
  default     = 2
}

variable "inference_min_capacity" {
  description = "Min auto-scaling capacity for inference-engine."
  type        = number
  default     = 2
}

variable "inference_max_capacity" {
  description = "Max auto-scaling capacity for inference-engine."
  type        = number
  default     = 10
}

variable "log_retention_days" {
  description = "CloudWatch log retention (days)."
  type        = number
  default     = 90
}

variable "redis_node_type" {
  description = "ElastiCache node type for prod."
  type        = string
  default     = "cache.t4g.small"
}

variable "redis_num_cache_clusters" {
  description = "Number of Redis nodes in the replication group (>=2 required for HA)."
  type        = number
  default     = 2
}
