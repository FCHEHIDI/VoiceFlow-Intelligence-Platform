######################################################################
# VoiceFlow RDS module
#
# Aurora PostgreSQL Serverless v2.
# - Master password is owned by RDS (manage_master_user_password = true);
#   the generated secret ARN is exported as an output so the ECS module
#   can wire it into the task definition's `secrets` block.
# - Storage is encrypted with the supplied KMS key (or default RDS CMK).
# - The cluster is placed in the DB subnets only and reachable from the
#   ml-service security group exclusively.
######################################################################

locals {
  name_prefix = "voiceflow-${var.env}"
  cluster_id  = "${local.name_prefix}-aurora"
}

resource "aws_db_subnet_group" "this" {
  name        = "${local.name_prefix}-db-subnet-group"
  description = "VoiceFlow ${var.env} Aurora DB subnet group"
  subnet_ids  = var.db_subnet_ids

  tags = {
    Name = "${local.name_prefix}-db-subnet-group"
  }
}

resource "aws_rds_cluster_parameter_group" "this" {
  name_prefix = "${local.name_prefix}-aurora-pg-"
  description = "VoiceFlow ${var.env} Aurora PostgreSQL cluster parameter group"
  family      = "aurora-postgresql15"

  parameter {
    name  = "log_statement"
    value = "ddl"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_rds_cluster" "this" {
  cluster_identifier              = local.cluster_id
  engine                          = "aurora-postgresql"
  engine_mode                     = "provisioned"
  engine_version                  = var.engine_version
  database_name                   = var.database_name
  master_username                 = var.master_username
  manage_master_user_password     = true
  master_user_secret_kms_key_id   = var.kms_key_id
  db_subnet_group_name            = aws_db_subnet_group.this.name
  vpc_security_group_ids          = [var.rds_security_group_id]
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.this.name
  storage_encrypted               = true
  kms_key_id                      = var.kms_key_id
  backup_retention_period         = var.backup_retention_days
  preferred_backup_window         = "02:00-03:00"
  preferred_maintenance_window    = "sun:04:00-sun:05:00"
  copy_tags_to_snapshot           = true
  deletion_protection             = var.deletion_protection
  skip_final_snapshot             = var.skip_final_snapshot
  final_snapshot_identifier       = var.skip_final_snapshot ? null : "${local.cluster_id}-final-${formatdate("YYYYMMDDhhmmss", timestamp())}"
  enabled_cloudwatch_logs_exports = ["postgresql"]

  serverlessv2_scaling_configuration {
    min_capacity = var.min_capacity
    max_capacity = var.max_capacity
  }

  lifecycle {
    ignore_changes = [
      final_snapshot_identifier,
    ]
  }

  tags = {
    Name = local.cluster_id
  }
}

resource "aws_rds_cluster_instance" "this" {
  count = var.instance_count

  identifier                            = "${local.cluster_id}-${count.index + 1}"
  cluster_identifier                    = aws_rds_cluster.this.id
  instance_class                        = "db.serverless"
  engine                                = aws_rds_cluster.this.engine
  engine_version                        = aws_rds_cluster.this.engine_version
  db_subnet_group_name                  = aws_db_subnet_group.this.name
  publicly_accessible                   = false
  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_kms_key_id       = var.performance_insights_enabled ? var.kms_key_id : null
  performance_insights_retention_period = var.performance_insights_enabled ? 7 : null
  auto_minor_version_upgrade            = true

  tags = {
    Name = "${local.cluster_id}-${count.index + 1}"
    Role = count.index == 0 ? "writer" : "reader"
  }
}
