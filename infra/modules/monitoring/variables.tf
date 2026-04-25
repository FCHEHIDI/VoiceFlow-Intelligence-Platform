variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "alert_email" {
  description = "Email address subscribed to the SNS alerts topic. Operator must confirm the subscription via the AWS confirmation email."
  type        = string
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days."
  type        = number
  default     = 30
}

variable "alb_arn_suffix" {
  description = "ALB ARN suffix for CloudWatch metric dimensions (LoadBalancer dimension)."
  type        = string
}

variable "ecs_cluster_name" {
  description = "ECS cluster name (used as the ClusterName dimension)."
  type        = string
}

variable "inference_service_name" {
  description = "Name of the inference-engine ECS service (used for low-task alarm)."
  type        = string
}

variable "ml_service_name" {
  description = "Name of the ml-service ECS service (used for monitoring outputs and dashboards)."
  type        = string
}

variable "error_rate_threshold_percent" {
  description = "Threshold for the 5xx-as-percent-of-requests alarm (e.g. 5 = 5%)."
  type        = number
  default     = 5
}

variable "p99_latency_threshold_seconds" {
  description = "Threshold (seconds) for the P99 ALB latency alarm. ALB metrics are in seconds; 0.1 = 100 ms."
  type        = number
  default     = 0.1
}

variable "inference_min_running_tasks" {
  description = "Minimum number of running inference tasks before triggering the low-task alarm."
  type        = number
  default     = 2
}
