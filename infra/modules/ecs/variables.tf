variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "aws_region" {
  description = "AWS region (used for awslogs driver options)."
  type        = string
}

############################
# Networking inputs
############################

variable "vpc_id" {
  description = "VPC ID."
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnets where the ALB lives."
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Private subnets where Fargate tasks run."
  type        = list(string)
}

variable "alb_security_group_id" {
  description = "Security group for the ALB."
  type        = string
}

variable "ml_service_security_group_id" {
  description = "Security group attached to ml-service tasks."
  type        = string
}

variable "inference_security_group_id" {
  description = "Security group attached to inference-engine tasks."
  type        = string
}

############################
# Container images
############################

variable "ml_image" {
  description = "Full container image reference for ml-service (e.g. 123.dkr.ecr.<region>.amazonaws.com/voiceflow-<env>/ml-service:sha-abc)."
  type        = string
}

variable "inference_image" {
  description = "Full container image reference for inference-engine."
  type        = string
}

############################
# Secrets to expose to tasks
############################

variable "jwt_secret_arn" {
  description = "ARN of the JWT secret in Secrets Manager."
  type        = string
}

variable "redis_auth_secret_arn" {
  description = "ARN of the Redis AUTH secret in Secrets Manager (JSON with auth_token field)."
  type        = string
}

variable "db_master_user_secret_arn" {
  description = "ARN of the RDS-managed master user secret (JSON with username/password fields)."
  type        = string
}

variable "extra_secret_arns" {
  description = "Additional secret ARNs the task execution role should be allowed to read (e.g. CMK ARNs not covered by default Secrets Manager permissions)."
  type        = list(string)
  default     = []
}

variable "secrets_kms_key_arn" {
  description = "Optional CMK ARN used to encrypt the secrets. If provided, kms:Decrypt is granted to the execution role on this key."
  type        = string
  default     = null
}

############################
# S3 buckets
############################

variable "models_bucket_arn" {
  description = "ARN of the models bucket (read-only access)."
  type        = string
}

variable "models_bucket_name" {
  description = "Name of the models bucket (used as the MODELS_BUCKET environment variable)."
  type        = string
}

variable "audio_bucket_arn" {
  description = "ARN of the audio bucket (read/write access)."
  type        = string
}

variable "audio_bucket_name" {
  description = "Name of the audio bucket (used as the AUDIO_BUCKET environment variable)."
  type        = string
}

############################
# Service config
############################

variable "ml_cpu" {
  description = "Fargate CPU units for ml-service task."
  type        = number
  default     = 1024
}

variable "ml_memory" {
  description = "Fargate memory (MiB) for ml-service task."
  type        = number
  default     = 2048
}

variable "inference_cpu" {
  description = "Fargate CPU units for inference-engine task."
  type        = number
  default     = 512
}

variable "inference_memory" {
  description = "Fargate memory (MiB) for inference-engine task."
  type        = number
  default     = 1024
}

variable "ml_desired_count" {
  description = "Desired count for ml-service service."
  type        = number
  default     = 1
}

variable "ml_min_capacity" {
  description = "Minimum auto-scaling capacity for ml-service."
  type        = number
  default     = 1
}

variable "ml_max_capacity" {
  description = "Maximum auto-scaling capacity for ml-service."
  type        = number
  default     = 2
}

variable "inference_desired_count" {
  description = "Desired count for inference-engine service."
  type        = number
  default     = 1
}

variable "inference_min_capacity" {
  description = "Minimum auto-scaling capacity for inference-engine."
  type        = number
  default     = 1
}

variable "inference_max_capacity" {
  description = "Maximum auto-scaling capacity for inference-engine."
  type        = number
  default     = 2
}

variable "cpu_target_utilization" {
  description = "Target CPU utilization percentage for service auto-scaling."
  type        = number
  default     = 70
}

############################
# Database / cache wiring
############################

variable "db_endpoint" {
  description = "Aurora writer endpoint hostname."
  type        = string
}

variable "db_port" {
  description = "Aurora port."
  type        = number
  default     = 5432
}

variable "db_name" {
  description = "Initial database name."
  type        = string
}

variable "db_master_username" {
  description = "Database master username (the password is injected via secrets)."
  type        = string
}

variable "redis_endpoint" {
  description = "Redis primary endpoint hostname."
  type        = string
}

variable "redis_port" {
  description = "Redis port."
  type        = number
  default     = 6379
}

############################
# Logging
############################

variable "ml_log_group_name" {
  description = "CloudWatch log group for ml-service (provisioned in monitoring module)."
  type        = string
}

variable "inference_log_group_name" {
  description = "CloudWatch log group for inference-engine (provisioned in monitoring module)."
  type        = string
}

############################
# ALB / TLS
############################

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for the HTTPS listener. When null, only HTTP listener is created (NOT recommended outside dev)."
  type        = string
  default     = null
}

variable "ssl_policy" {
  description = "ALB SSL policy."
  type        = string
  default     = "ELBSecurityPolicy-TLS13-1-2-2021-06"
}

variable "enable_alb_access_logs" {
  description = "If true, enable ALB access logs (requires alb_access_logs_bucket)."
  type        = bool
  default     = false
}

variable "alb_access_logs_bucket" {
  description = "S3 bucket for ALB access logs (must have a bucket policy allowing the ELB account)."
  type        = string
  default     = null
}

variable "deletion_protection" {
  description = "ALB deletion protection. Set true in prod."
  type        = bool
  default     = false
}

variable "stickiness_cookie_duration_seconds" {
  description = "Duration (seconds) of the lb_cookie used for ALB stickiness on the ml/inference target groups."
  type        = number
  default     = 86400
}
