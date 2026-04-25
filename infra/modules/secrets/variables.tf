variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "create_kms_key" {
  description = "If true, provision a customer-managed KMS key for these secrets. If false, use the AWS-managed aws/secretsmanager key."
  type        = bool
  default     = true
}

variable "kms_deletion_window_days" {
  description = "Pending deletion window for the customer-managed KMS key (days, between 7 and 30)."
  type        = number
  default     = 30
}

variable "recovery_window_days" {
  description = "Recovery window for deleted secrets (days, 0 to delete immediately, otherwise 7-30)."
  type        = number
  default     = 7
}

variable "jwt_secret_length" {
  description = "Length of the auto-generated JWT secret (minimum 32)."
  type        = number
  default     = 48

  validation {
    condition     = var.jwt_secret_length >= 32
    error_message = "JWT secret must be at least 32 characters long."
  }
}

variable "redis_auth_token_length" {
  description = "Length of the auto-generated Redis auth token (minimum 32, max 128 per ElastiCache)."
  type        = number
  default     = 48

  validation {
    condition     = var.redis_auth_token_length >= 32 && var.redis_auth_token_length <= 128
    error_message = "Redis auth token length must be between 32 and 128 characters."
  }
}
