######################################################################
# VoiceFlow ElastiCache module
#
# Single-shard Redis 7.x replication group with:
#   - Transit and at-rest encryption
#   - AUTH token sourced from the secrets module
#   - Subnet group restricted to the DB subnets
######################################################################

locals {
  name_prefix = "voiceflow-${var.env}"
  cluster_id  = "${local.name_prefix}-redis"
}

resource "aws_elasticache_subnet_group" "this" {
  name        = "${local.name_prefix}-redis-subnet-group"
  description = "VoiceFlow ${var.env} ElastiCache subnet group"
  subnet_ids  = var.db_subnet_ids

  tags = {
    Name = "${local.name_prefix}-redis-subnet-group"
  }
}

resource "aws_elasticache_parameter_group" "this" {
  name        = "${local.name_prefix}-redis-pg"
  description = "VoiceFlow ${var.env} Redis parameter group"
  family      = var.parameter_group_family

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id = local.cluster_id
  description          = "VoiceFlow ${var.env} Redis replication group"

  engine                     = "redis"
  engine_version             = var.engine_version
  node_type                  = var.node_type
  num_cache_clusters         = var.num_cache_clusters
  parameter_group_name       = aws_elasticache_parameter_group.this.name
  port                       = 6379
  subnet_group_name          = aws_elasticache_subnet_group.this.name
  security_group_ids         = [var.redis_security_group_id]
  automatic_failover_enabled = var.automatic_failover_enabled
  multi_az_enabled           = var.multi_az_enabled

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  kms_key_id                 = var.kms_key_id
  auth_token                 = var.auth_token

  snapshot_retention_limit = var.snapshot_retention_limit
  snapshot_window          = "03:00-04:00"
  maintenance_window       = "sun:05:00-sun:06:00"
  apply_immediately        = false

  tags = {
    Name = local.cluster_id
  }

  lifecycle {
    ignore_changes = [
      num_cache_clusters,
    ]
  }
}
