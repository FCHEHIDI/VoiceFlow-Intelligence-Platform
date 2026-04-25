variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "image_tag_mutability" {
  description = "ECR image tag mutability. Use MUTABLE in dev, IMMUTABLE in prod."
  type        = string
  default     = "MUTABLE"

  validation {
    condition     = contains(["MUTABLE", "IMMUTABLE"], var.image_tag_mutability)
    error_message = "image_tag_mutability must be one of MUTABLE or IMMUTABLE."
  }
}

variable "scan_on_push" {
  description = "Enable basic image scanning on push."
  type        = bool
  default     = true
}

variable "max_image_count" {
  description = "Number of most recent images to retain via lifecycle policy."
  type        = number
  default     = 20
}

variable "force_delete" {
  description = "If true, ECR repos may be deleted even when they contain images. Use with caution; should be false in prod."
  type        = bool
  default     = false
}
