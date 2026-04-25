output "replication_group_id" {
  description = "ID of the ElastiCache replication group."
  value       = aws_elasticache_replication_group.this.id
}

output "primary_endpoint_address" {
  description = "DNS name of the primary endpoint (use this for writes)."
  value       = aws_elasticache_replication_group.this.primary_endpoint_address
}

output "reader_endpoint_address" {
  description = "DNS name of the reader endpoint (use this for read traffic when replicas exist)."
  value       = aws_elasticache_replication_group.this.reader_endpoint_address
}

output "port" {
  description = "Redis port (6379)."
  value       = aws_elasticache_replication_group.this.port
}
