######################################################################
# VoiceFlow S3 module
#
#   - models bucket: versioning enabled, AES256 SSE, public access fully
#                    blocked. Used for ONNX model weights.
#   - audio bucket : versioning suspended, AES256 SSE, 7-day lifecycle
#                    expiration, public access fully blocked. Used for
#                    short-lived user audio uploads.
######################################################################

locals {
  name_prefix        = "voiceflow-${var.env}"
  models_bucket_name = var.models_bucket_name_override != null ? var.models_bucket_name_override : "${local.name_prefix}-models-${random_id.suffix.hex}"
  audio_bucket_name  = var.audio_bucket_name_override != null ? var.audio_bucket_name_override : "${local.name_prefix}-audio-${random_id.suffix.hex}"
}

resource "random_id" "suffix" {
  byte_length = 4
  keepers = {
    env = var.env
  }
}

######################################################################
# Models bucket
######################################################################

resource "aws_s3_bucket" "models" {
  bucket        = local.models_bucket_name
  force_destroy = var.force_destroy

  tags = {
    Name      = local.models_bucket_name
    Component = "models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

######################################################################
# Audio bucket
######################################################################

resource "aws_s3_bucket" "audio" {
  bucket        = local.audio_bucket_name
  force_destroy = var.force_destroy

  tags = {
    Name      = local.audio_bucket_name
    Component = "audio-uploads"
  }
}

resource "aws_s3_bucket_versioning" "audio" {
  bucket = aws_s3_bucket.audio.id

  versioning_configuration {
    status = "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "audio" {
  bucket = aws_s3_bucket.audio.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "audio" {
  bucket = aws_s3_bucket.audio.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "audio" {
  bucket = aws_s3_bucket.audio.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "audio" {
  bucket = aws_s3_bucket.audio.id

  rule {
    id     = "expire-after-${var.audio_expiration_days}d"
    status = "Enabled"

    filter {}

    expiration {
      days = var.audio_expiration_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }

    noncurrent_version_expiration {
      noncurrent_days = 1
    }
  }
}
