variable "env" {
  description = "Environment name (e.g. dev, prod). Used for naming and tags."
  type        = string
}

variable "vpc_cidr" {
  description = "Primary CIDR block for the VoiceFlow VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_multi_az_nat" {
  description = "If true, provision one NAT Gateway per AZ (high availability, prod). If false, a single NAT Gateway is shared by all private subnets (cost-conscious, dev)."
  type        = bool
  default     = false
}

variable "enable_flow_logs" {
  description = "If true, enable VPC Flow Logs to CloudWatch for security auditing."
  type        = bool
  default     = true
}

variable "flow_logs_retention_days" {
  description = "CloudWatch log retention (days) for VPC Flow Logs."
  type        = number
  default     = 30
}
