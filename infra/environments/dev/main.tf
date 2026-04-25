######################################################################
# VoiceFlow — dev environment
#
# Cost-conscious sizing:
#   * single NAT Gateway shared across both private subnets
#   * Aurora Serverless v2 0.5–2.0 ACU
#   * single-node Redis
#   * 1 desired task per ECS service, max 2
#   * ECR tags MUTABLE (dev iteration)
######################################################################

locals {
  name_prefix              = "voiceflow-${var.env}"
  ml_log_group_name        = "/ecs/voiceflow-ml-${var.env}"
  inference_log_group_name = "/ecs/voiceflow-inference-${var.env}"
}

module "networking" {
  source              = "../../modules/networking"
  env                 = var.env
  vpc_cidr            = var.vpc_cidr
  enable_multi_az_nat = false
  enable_flow_logs    = true
}

module "secrets" {
  source                  = "../../modules/secrets"
  env                     = var.env
  create_kms_key          = true
  recovery_window_days    = 0
  jwt_secret_length       = 48
  redis_auth_token_length = 48
}

module "s3" {
  source                = "../../modules/s3"
  env                   = var.env
  audio_expiration_days = 7
  force_destroy         = true
}

module "ecr" {
  source               = "../../modules/ecr"
  env                  = var.env
  image_tag_mutability = "MUTABLE"
  scan_on_push         = true
  max_image_count      = 20
  force_delete         = true
}

module "rds" {
  source                       = "../../modules/rds"
  env                          = var.env
  db_subnet_ids                = module.networking.db_subnet_ids
  rds_security_group_id        = module.networking.rds_security_group_id
  engine_version               = "15.4"
  database_name                = "voiceflow"
  master_username              = "voiceflow_admin"
  min_capacity                 = var.rds_min_capacity
  max_capacity                 = var.rds_max_capacity
  instance_count               = 1
  deletion_protection          = false
  skip_final_snapshot          = true
  backup_retention_days        = 1
  performance_insights_enabled = false
  kms_key_id                   = module.secrets.kms_key_arn
}

module "elasticache" {
  source                     = "../../modules/elasticache"
  env                        = var.env
  db_subnet_ids              = module.networking.db_subnet_ids
  redis_security_group_id    = module.networking.redis_security_group_id
  node_type                  = "cache.t4g.micro"
  engine_version             = "7.1"
  num_cache_clusters         = 1
  automatic_failover_enabled = false
  multi_az_enabled           = false
  auth_token                 = module.secrets.redis_auth_token
  kms_key_id                 = module.secrets.kms_key_arn
  snapshot_retention_limit   = 0
}

module "monitoring" {
  source                        = "../../modules/monitoring"
  env                           = var.env
  alert_email                   = var.alert_email
  log_retention_days            = var.log_retention_days
  alb_arn_suffix                = module.ecs.alb_arn_suffix
  ecs_cluster_name              = module.ecs.cluster_name
  ml_service_name               = module.ecs.ml_service_name
  inference_service_name        = module.ecs.inference_service_name
  error_rate_threshold_percent  = 5
  p99_latency_threshold_seconds = 0.1
  inference_min_running_tasks   = 1
}

module "ecs" {
  source = "../../modules/ecs"

  env        = var.env
  aws_region = var.aws_region

  vpc_id                       = module.networking.vpc_id
  public_subnet_ids            = module.networking.public_subnet_ids
  private_subnet_ids           = module.networking.private_subnet_ids
  alb_security_group_id        = module.networking.alb_security_group_id
  ml_service_security_group_id = module.networking.ml_service_security_group_id
  inference_security_group_id  = module.networking.inference_security_group_id

  ml_image        = var.ml_image
  inference_image = var.inference_image

  jwt_secret_arn            = module.secrets.jwt_secret_arn
  redis_auth_secret_arn     = module.secrets.redis_auth_secret_arn
  db_master_user_secret_arn = module.rds.master_user_secret_arn
  secrets_kms_key_arn       = module.secrets.kms_key_arn

  models_bucket_arn  = module.s3.models_bucket_arn
  models_bucket_name = module.s3.models_bucket_name
  audio_bucket_arn   = module.s3.audio_bucket_arn
  audio_bucket_name  = module.s3.audio_bucket_name

  ml_cpu                  = 1024
  ml_memory               = 2048
  ml_desired_count        = var.ml_desired_count
  ml_min_capacity         = var.ml_min_capacity
  ml_max_capacity         = var.ml_max_capacity
  inference_cpu           = 512
  inference_memory        = 1024
  inference_desired_count = var.inference_desired_count
  inference_min_capacity  = var.inference_min_capacity
  inference_max_capacity  = var.inference_max_capacity
  cpu_target_utilization  = 70

  db_endpoint        = module.rds.writer_endpoint
  db_port            = module.rds.port
  db_name            = module.rds.database_name
  db_master_username = module.rds.master_username

  redis_endpoint = module.elasticache.primary_endpoint_address
  redis_port     = module.elasticache.port

  ml_log_group_name        = local.ml_log_group_name
  inference_log_group_name = local.inference_log_group_name

  acm_certificate_arn = var.acm_certificate_arn
  deletion_protection = false
}
