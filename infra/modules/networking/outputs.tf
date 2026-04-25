output "vpc_id" {
  description = "ID of the VoiceFlow VPC."
  value       = aws_vpc.this.id
}

output "vpc_cidr" {
  description = "Primary CIDR block of the VPC."
  value       = aws_vpc.this.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the two public subnets (used by the ALB)."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the two private subnets (used by ECS Fargate tasks)."
  value       = aws_subnet.private[*].id
}

output "db_subnet_ids" {
  description = "IDs of the two DB subnets (used by RDS and ElastiCache)."
  value       = aws_subnet.db[*].id
}

output "alb_security_group_id" {
  description = "Security group ID for the public-facing ALB."
  value       = aws_security_group.alb.id
}

output "ml_service_security_group_id" {
  description = "Security group ID attached to ml-service ECS tasks."
  value       = aws_security_group.ml_service.id
}

output "inference_security_group_id" {
  description = "Security group ID attached to inference-engine ECS tasks."
  value       = aws_security_group.inference.id
}

output "rds_security_group_id" {
  description = "Security group ID attached to the Aurora cluster."
  value       = aws_security_group.rds.id
}

output "redis_security_group_id" {
  description = "Security group ID attached to the ElastiCache replication group."
  value       = aws_security_group.redis.id
}

output "availability_zones" {
  description = "AZs used by this VPC."
  value       = local.azs
}
