output "kms_key_arn" {
  description = "ARN of the customer-managed KMS key (null when create_kms_key = false)."
  value       = var.create_kms_key ? aws_kms_key.secrets[0].arn : null
}

output "kms_key_id" {
  description = "ID of the customer-managed KMS key (null when create_kms_key = false)."
  value       = var.create_kms_key ? aws_kms_key.secrets[0].key_id : null
}

output "jwt_secret_arn" {
  description = "ARN of the JWT signing key secret (consumed by JWT_SECRET_ARN)."
  value       = aws_secretsmanager_secret.jwt.arn
}

output "jwt_secret_name" {
  description = "Name of the JWT signing key secret."
  value       = aws_secretsmanager_secret.jwt.name
}

output "redis_auth_secret_arn" {
  description = "ARN of the Redis AUTH token secret (JSON with key auth_token)."
  value       = aws_secretsmanager_secret.redis.arn
}

output "redis_auth_token" {
  description = "The Redis AUTH token (sensitive). Used by the elasticache module to enable AUTH on the replication group."
  value       = random_password.redis.result
  sensitive   = true
}
