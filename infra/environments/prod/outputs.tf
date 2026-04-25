output "vpc_id" {
  description = "VPC ID."
  value       = module.networking.vpc_id
}

output "alb_dns_name" {
  description = "Public DNS of the ALB."
  value       = module.ecs.alb_dns_name
}

output "alb_zone_id" {
  description = "Hosted zone ID of the ALB (for Route53 alias records)."
  value       = module.ecs.alb_zone_id
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = module.ecs.cluster_name
}

output "ml_service_name" {
  description = "ECS service name for ml-service."
  value       = module.ecs.ml_service_name
}

output "inference_service_name" {
  description = "ECS service name for inference-engine."
  value       = module.ecs.inference_service_name
}

output "ml_repository_url" {
  description = "ECR repository URL for ml-service."
  value       = module.ecr.ml_repository_url
}

output "inference_repository_url" {
  description = "ECR repository URL for inference-engine."
  value       = module.ecr.inference_repository_url
}

output "models_bucket_name" {
  description = "S3 bucket for ONNX models."
  value       = module.s3.models_bucket_name
}

output "audio_bucket_name" {
  description = "S3 bucket for audio uploads."
  value       = module.s3.audio_bucket_name
}

output "jwt_secret_arn" {
  description = "ARN of the JWT signing key secret."
  value       = module.secrets.jwt_secret_arn
}

output "redis_auth_secret_arn" {
  description = "ARN of the Redis AUTH secret."
  value       = module.secrets.redis_auth_secret_arn
}

output "db_master_user_secret_arn" {
  description = "ARN of the RDS-managed master user secret."
  value       = module.rds.master_user_secret_arn
}

output "db_writer_endpoint" {
  description = "Aurora writer endpoint."
  value       = module.rds.writer_endpoint
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint."
  value       = module.elasticache.primary_endpoint_address
}

output "sns_alerts_topic_arn" {
  description = "SNS topic ARN for alerts."
  value       = module.monitoring.sns_topic_arn
}
