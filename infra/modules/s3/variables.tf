variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "models_bucket_name_override" {
  description = "Optional override for the models bucket name. If null, a name is auto-generated."
  type        = string
  default     = null
}

variable "audio_bucket_name_override" {
  description = "Optional override for the audio uploads bucket name. If null, a name is auto-generated."
  type        = string
  default     = null
}

variable "audio_expiration_days" {
  description = "Lifecycle rule expiration (days) for the audio bucket."
  type        = number
  default     = 7
}

variable "force_destroy" {
  description = "If true, allows Terraform to delete non-empty buckets. Use with caution; should be false in prod."
  type        = bool
  default     = false
}
