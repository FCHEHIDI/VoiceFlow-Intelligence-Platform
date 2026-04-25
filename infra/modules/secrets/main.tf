######################################################################
# VoiceFlow secrets module
#
# Provisions:
#   * (optional) a customer-managed KMS key used to encrypt all secrets
#   * a Secrets Manager secret for the JWT signing key (auto-generated, 32+ chars)
#   * a Secrets Manager secret for the Redis AUTH token (auto-generated)
#
# Note: the database master password is owned by RDS itself
# (manage_master_user_password = true). The RDS module exposes the
# generated secret ARN as an output so consumers can wire it through.
# This module only references the JWT and Redis secrets.
######################################################################

data "aws_caller_identity" "current" {}

locals {
  name_prefix = "voiceflow-${var.env}"
}

######################################################################
# KMS key
######################################################################

data "aws_iam_policy_document" "kms_default" {
  statement {
    sid     = "EnableRootAccountAdmin"
    actions = ["kms:*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    resources = ["*"]
  }

  statement {
    sid = "AllowSecretsManagerUseOfTheKey"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey*",
      "kms:ReEncrypt*",
    ]
    principals {
      type        = "Service"
      identifiers = ["secretsmanager.amazonaws.com"]
    }
    resources = ["*"]
  }
}

resource "aws_kms_key" "secrets" {
  count = var.create_kms_key ? 1 : 0

  description             = "VoiceFlow ${var.env} CMK for Secrets Manager secrets"
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.kms_default.json

  tags = {
    Name = "${local.name_prefix}-secrets-cmk"
  }
}

resource "aws_kms_alias" "secrets" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${local.name_prefix}-secrets"
  target_key_id = aws_kms_key.secrets[0].key_id
}

locals {
  kms_key_id = var.create_kms_key ? aws_kms_key.secrets[0].arn : null
}

######################################################################
# JWT signing key
######################################################################

resource "random_password" "jwt" {
  length           = var.jwt_secret_length
  special          = true
  override_special = "!@#$%^&*()-_=+"
}

resource "aws_secretsmanager_secret" "jwt" {
  name                    = "${local.name_prefix}/jwt-secret-key"
  description             = "VoiceFlow ${var.env} JWT signing key"
  kms_key_id              = local.kms_key_id
  recovery_window_in_days = var.recovery_window_days

  tags = {
    Name      = "${local.name_prefix}-jwt-secret-key"
    Component = "auth"
  }
}

resource "aws_secretsmanager_secret_version" "jwt" {
  secret_id     = aws_secretsmanager_secret.jwt.id
  secret_string = random_password.jwt.result
}

######################################################################
# Redis AUTH token
######################################################################

# ElastiCache requires only printable ASCII and disallows @, ", /
resource "random_password" "redis" {
  length           = var.redis_auth_token_length
  special          = true
  override_special = "!#$%^&*()-_=+[]{}<>?"
}

resource "aws_secretsmanager_secret" "redis" {
  name                    = "${local.name_prefix}/redis-auth-token"
  description             = "VoiceFlow ${var.env} ElastiCache Redis AUTH token"
  kms_key_id              = local.kms_key_id
  recovery_window_in_days = var.recovery_window_days

  tags = {
    Name      = "${local.name_prefix}-redis-auth-token"
    Component = "cache"
  }
}

resource "aws_secretsmanager_secret_version" "redis" {
  secret_id = aws_secretsmanager_secret.redis.id
  secret_string = jsonencode({
    auth_token = random_password.redis.result
  })
}
