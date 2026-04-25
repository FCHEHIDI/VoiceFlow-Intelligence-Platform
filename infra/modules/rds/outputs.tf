output "cluster_id" {
  description = "Aurora cluster identifier."
  value       = aws_rds_cluster.this.id
}

output "cluster_arn" {
  description = "ARN of the Aurora cluster."
  value       = aws_rds_cluster.this.arn
}

output "writer_endpoint" {
  description = "Aurora writer endpoint hostname."
  value       = aws_rds_cluster.this.endpoint
}

output "reader_endpoint" {
  description = "Aurora read-only endpoint hostname."
  value       = aws_rds_cluster.this.reader_endpoint
}

output "port" {
  description = "Database port."
  value       = aws_rds_cluster.this.port
}

output "database_name" {
  description = "Initial database name."
  value       = aws_rds_cluster.this.database_name
}

output "master_username" {
  description = "Master username."
  value       = aws_rds_cluster.this.master_username
}

# RDS-managed secret containing { username, password }. Use this ARN in
# ECS task definitions: valueFrom = "${arn}:password::" to pluck the password.
output "master_user_secret_arn" {
  description = "ARN of the RDS-managed master user secret in Secrets Manager."
  value       = try(aws_rds_cluster.this.master_user_secret[0].secret_arn, null)
}
