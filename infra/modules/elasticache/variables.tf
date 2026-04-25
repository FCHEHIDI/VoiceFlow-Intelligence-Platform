variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "db_subnet_ids" {
  description = "Subnet IDs for the ElastiCache subnet group (must span at least 2 AZs)."
  type        = list(string)
}

variable "redis_security_group_id" {
  description = "Security group ID attached to the Redis cluster."
  type        = string
}

variable "node_type" {
  description = "ElastiCache node instance type."
  type        = string
  default     = "cache.t4g.micro"
}

variable "engine_version" {
  description = "Redis engine version (7.x)."
  type        = string
  default     = "7.1"
}

variable "parameter_group_family" {
  description = "Redis parameter group family."
  type        = string
  default     = "redis7"
}

variable "num_cache_clusters" {
  description = "Total number of nodes in the replication group (1 = primary only, 2 = primary + 1 replica, etc.). For prod, use >= 2 for HA."
  type        = number
  default     = 1
}

variable "automatic_failover_enabled" {
  description = "Whether to automatically promote a replica on primary failure. Requires num_cache_clusters >= 2."
  type        = bool
  default     = false
}

variable "multi_az_enabled" {
  description = "Multi-AZ for replication group. Requires num_cache_clusters >= 2 and automatic_failover_enabled."
  type        = bool
  default     = false
}

variable "auth_token" {
  description = "Redis AUTH token (32-128 printable ASCII chars). Sourced from the secrets module."
  type        = string
  sensitive   = true
}

variable "kms_key_id" {
  description = "Optional KMS key ARN used for at-rest encryption. If null, AWS-managed key is used."
  type        = string
  default     = null
}

variable "snapshot_retention_limit" {
  description = "Number of days to retain automatic snapshots (0 disables snapshots)."
  type        = number
  default     = 1
}
