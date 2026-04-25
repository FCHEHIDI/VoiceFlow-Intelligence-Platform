variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "db_subnet_ids" {
  description = "IDs of the DB subnets the Aurora cluster lives in (must span at least 2 AZs)."
  type        = list(string)
}

variable "rds_security_group_id" {
  description = "Security group ID attached to the Aurora cluster."
  type        = string
}

variable "engine_version" {
  description = "Aurora PostgreSQL engine version."
  type        = string
  default     = "15.4"
}

variable "database_name" {
  description = "Initial database name created in the Aurora cluster."
  type        = string
  default     = "voiceflow"
}

variable "master_username" {
  description = "Master username for Aurora. The password is managed by RDS itself (manage_master_user_password)."
  type        = string
  default     = "voiceflow_admin"
}

variable "min_capacity" {
  description = "Aurora Serverless v2 minimum ACU (0.5 - 128)."
  type        = number
  default     = 0.5
}

variable "max_capacity" {
  description = "Aurora Serverless v2 maximum ACU (0.5 - 128)."
  type        = number
  default     = 2.0
}

variable "instance_count" {
  description = "Number of writer/reader instances. dev=1, prod>=2 for HA."
  type        = number
  default     = 1
}

variable "deletion_protection" {
  description = "If true, the cluster cannot be destroyed via Terraform without setting this to false first."
  type        = bool
  default     = false
}

variable "skip_final_snapshot" {
  description = "If true, no final snapshot is taken on destroy. Set false in prod."
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days."
  type        = number
  default     = 7
}

variable "performance_insights_enabled" {
  description = "Enable RDS Performance Insights on the writer."
  type        = bool
  default     = true
}

variable "kms_key_id" {
  description = "Optional KMS key ARN to use for storage and master-user-password encryption. If null, AWS-managed keys are used."
  type        = string
  default     = null
}
