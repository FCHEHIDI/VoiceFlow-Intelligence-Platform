######################################################################
# VoiceFlow ECS module
#
# Provisions:
#   * an ECS cluster with Container Insights enabled
#   * task execution + task IAM roles (least privilege; secrets are pulled
#     via the execution role, business permissions via the task role)
#   * ml-service task definition + service (Fargate, awsvpc)
#   * inference-engine task definition + service
#   * ALB + listeners (80 redirect → 443) + target groups with sticky
#     sessions for both services (required for WebSocket)
#   * listener rules: /api/* and /ws/* -> ml TG, /infer* -> inference TG
#   * Application Auto Scaling target tracking on CPU 70%
######################################################################

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name_prefix   = "voiceflow-${var.env}"
  account_id    = data.aws_caller_identity.current.account_id
  partition     = data.aws_partition.current.partition
  has_tls       = var.acm_certificate_arn != null
  ml_container  = "ml-service"
  inf_container = "inference-engine"
  ml_port       = 8000
  inf_port      = 3000
  task_secret_arns = compact(distinct(concat([
    var.jwt_secret_arn,
    var.redis_auth_secret_arn,
    var.db_master_user_secret_arn,
  ], var.extra_secret_arns)))
}

######################################################################
# Cluster
######################################################################

resource "aws_ecs_cluster" "this" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${local.name_prefix}-cluster"
  }
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
}

######################################################################
# IAM: task execution role
######################################################################

data "aws_iam_policy_document" "tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name_prefix}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  statement {
    sid       = "ReadSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = local.task_secret_arns
  }

  dynamic "statement" {
    for_each = var.secrets_kms_key_arn == null ? [] : [1]
    content {
      sid       = "DecryptSecretsCMK"
      actions   = ["kms:Decrypt"]
      resources = [var.secrets_kms_key_arn]
    }
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "${local.name_prefix}-execution-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

######################################################################
# IAM: task role (application-level permissions)
######################################################################

resource "aws_iam_role" "ml_task" {
  name               = "${local.name_prefix}-ml-task"
  assume_role_policy = data.aws_iam_policy_document.tasks_assume.json
}

resource "aws_iam_role" "inference_task" {
  name               = "${local.name_prefix}-inference-task"
  assume_role_policy = data.aws_iam_policy_document.tasks_assume.json
}

# ml-service: read models bucket, read+write audio bucket
data "aws_iam_policy_document" "ml_task" {
  statement {
    sid     = "ReadModelsBucket"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      var.models_bucket_arn,
      "${var.models_bucket_arn}/*",
    ]
  }

  statement {
    sid = "AudioBucketRW"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:AbortMultipartUpload",
    ]
    resources = [
      var.audio_bucket_arn,
      "${var.audio_bucket_arn}/*",
    ]
  }

  statement {
    sid = "CloudWatchLogs"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:${var.ml_log_group_name}:*"]
  }
}

resource "aws_iam_role_policy" "ml_task" {
  name   = "${local.name_prefix}-ml-task"
  role   = aws_iam_role.ml_task.id
  policy = data.aws_iam_policy_document.ml_task.json
}

# inference-engine: read models bucket only
data "aws_iam_policy_document" "inference_task" {
  statement {
    sid     = "ReadModelsBucket"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      var.models_bucket_arn,
      "${var.models_bucket_arn}/*",
    ]
  }

  statement {
    sid = "CloudWatchLogs"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:${var.inference_log_group_name}:*"]
  }
}

resource "aws_iam_role_policy" "inference_task" {
  name   = "${local.name_prefix}-inference-task"
  role   = aws_iam_role.inference_task.id
  policy = data.aws_iam_policy_document.inference_task.json
}

######################################################################
# ALB
######################################################################

resource "aws_lb" "this" {
  name                       = "${local.name_prefix}-alb"
  internal                   = false
  load_balancer_type         = "application"
  security_groups            = [var.alb_security_group_id]
  subnets                    = var.public_subnet_ids
  enable_deletion_protection = var.deletion_protection
  drop_invalid_header_fields = true
  idle_timeout               = 4000

  dynamic "access_logs" {
    for_each = var.enable_alb_access_logs && var.alb_access_logs_bucket != null ? [1] : []
    content {
      bucket  = var.alb_access_logs_bucket
      prefix  = "alb/${local.name_prefix}"
      enabled = true
    }
  }

  tags = {
    Name = "${local.name_prefix}-alb"
  }
}

resource "aws_lb_target_group" "ml" {
  name        = "${local.name_prefix}-ml-tg"
  port        = local.ml_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    path                = "/health"
    matcher             = "200-299"
    protocol            = "HTTP"
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = var.stickiness_cookie_duration_seconds
    enabled         = true
  }

  deregistration_delay = 30

  tags = {
    Name = "${local.name_prefix}-ml-tg"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_lb_target_group" "inference" {
  name        = "${local.name_prefix}-inf-tg"
  port        = local.inf_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    path                = "/health"
    matcher             = "200-299"
    protocol            = "HTTP"
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = var.stickiness_cookie_duration_seconds
    enabled         = true
  }

  deregistration_delay = 30

  tags = {
    Name = "${local.name_prefix}-inf-tg"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# When TLS is configured: HTTP (80) redirects to HTTPS (443),
# and HTTPS hosts the routing listener.
resource "aws_lb_listener" "http_redirect" {
  count = local.has_tls ? 1 : 0

  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  count = local.has_tls ? 1 : 0

  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = var.ssl_policy
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "application/json"
      message_body = "{\"error\":\"not_found\"}"
      status_code  = "404"
    }
  }
}

# Dev-only fallback: when no ACM cert is provided (var.acm_certificate_arn = "")
# we expose an HTTP listener so engineers can hit the ALB without TLS. In every
# environment with `has_tls = true` (staging/prod) this resource has count = 0
# and the routing listener is the HTTPS one above, which uses
# ELBSecurityPolicy-TLS13-1-2-2021-06. This file is allow-listed in
# .semgrepignore for that reason.
resource "aws_lb_listener" "http_route" {
  count = local.has_tls ? 0 : 1

  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "application/json"
      message_body = "{\"error\":\"not_found\"}"
      status_code  = "404"
    }
  }
}

locals {
  routed_listener_arn = local.has_tls ? aws_lb_listener.https[0].arn : aws_lb_listener.http_route[0].arn
}

resource "aws_lb_listener_rule" "ml_api" {
  listener_arn = local.routed_listener_arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ml.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

resource "aws_lb_listener_rule" "ml_ws" {
  listener_arn = local.routed_listener_arn
  priority     = 110

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ml.arn
  }

  condition {
    path_pattern {
      values = ["/ws/*"]
    }
  }
}

resource "aws_lb_listener_rule" "inference_infer" {
  listener_arn = local.routed_listener_arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.inference.arn
  }

  condition {
    path_pattern {
      values = ["/infer*"]
    }
  }
}

######################################################################
# Task definitions
######################################################################

locals {
  ml_container_def = [
    {
      name      = local.ml_container
      image     = var.ml_image
      essential = true
      portMappings = [
        {
          containerPort = local.ml_port
          hostPort      = local.ml_port
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "ENV", value = var.env },
        { name = "AWS_REGION", value = var.aws_region },
        { name = "POSTGRES_HOST", value = var.db_endpoint },
        { name = "POSTGRES_PORT", value = tostring(var.db_port) },
        { name = "POSTGRES_DB", value = var.db_name },
        { name = "POSTGRES_USER", value = var.db_master_username },
        { name = "REDIS_HOST", value = var.redis_endpoint },
        { name = "REDIS_PORT", value = tostring(var.redis_port) },
        { name = "REDIS_TLS", value = "true" },
        { name = "RUST_SERVICE_URL", value = "http://${local.inf_container}.${local.name_prefix}.local:${local.inf_port}" },
        { name = "MODELS_BUCKET", value = var.models_bucket_name },
        { name = "AUDIO_BUCKET", value = var.audio_bucket_name },
        { name = "JWT_SECRET_ARN", value = var.jwt_secret_arn },
        { name = "REDIS_AUTH_SECRET_ARN", value = var.redis_auth_secret_arn },
        { name = "DB_SECRET_ARN", value = var.db_master_user_secret_arn },
      ]
      secrets = [
        {
          name      = "JWT_SECRET_KEY"
          valueFrom = var.jwt_secret_arn
        },
        {
          name      = "POSTGRES_PASSWORD"
          valueFrom = "${var.db_master_user_secret_arn}:password::"
        },
        {
          name      = "REDIS_PASSWORD"
          valueFrom = "${var.redis_auth_secret_arn}:auth_token::"
        },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = var.ml_log_group_name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -fsS http://localhost:${local.ml_port}/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ]

  inference_container_def = [
    {
      name      = local.inf_container
      image     = var.inference_image
      essential = true
      portMappings = [
        {
          containerPort = local.inf_port
          hostPort      = local.inf_port
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "ENV", value = var.env },
        { name = "AWS_REGION", value = var.aws_region },
        { name = "RUST_LOG", value = "info" },
        { name = "BIND_ADDR", value = "0.0.0.0:${local.inf_port}" },
        { name = "MODELS_BUCKET", value = var.models_bucket_name },
      ]
      secrets = []
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = var.inference_log_group_name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -fsS http://localhost:${local.inf_port}/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ]
}

resource "aws_ecs_task_definition" "ml" {
  family                   = "${local.name_prefix}-ml-service"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ml_cpu
  memory                   = var.ml_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.ml_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode(local.ml_container_def)

  tags = {
    Name = "${local.name_prefix}-ml-task"
  }
}

resource "aws_ecs_task_definition" "inference" {
  family                   = "${local.name_prefix}-inference-engine"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.inference_cpu
  memory                   = var.inference_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.inference_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode(local.inference_container_def)

  tags = {
    Name = "${local.name_prefix}-inference-task"
  }
}

######################################################################
# Service Discovery (Cloud Map) for internal ml -> inference calls
######################################################################

resource "aws_service_discovery_private_dns_namespace" "this" {
  name        = "${local.name_prefix}.local"
  description = "VoiceFlow ${var.env} internal service discovery"
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "inference" {
  name = local.inf_container

  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.this.id
    routing_policy = "MULTIVALUE"

    dns_records {
      ttl  = 10
      type = "A"
    }
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

######################################################################
# ECS services
######################################################################

resource "aws_ecs_service" "ml" {
  name                               = "${local.name_prefix}-ml-service"
  cluster                            = aws_ecs_cluster.this.id
  task_definition                    = aws_ecs_task_definition.ml.arn
  desired_count                      = var.ml_desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 90
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  enable_execute_command             = false
  propagate_tags                     = "SERVICE"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ml_service_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ml.arn
    container_name   = local.ml_container
    container_port   = local.ml_port
  }

  lifecycle {
    ignore_changes = [desired_count, task_definition]
  }

  depends_on = [
    aws_lb_listener.http_redirect,
    aws_lb_listener.http_route,
    aws_lb_listener.https,
  ]

  tags = {
    Name = "${local.name_prefix}-ml-service"
  }
}

resource "aws_ecs_service" "inference" {
  name                               = "${local.name_prefix}-inference-engine"
  cluster                            = aws_ecs_cluster.this.id
  task_definition                    = aws_ecs_task_definition.inference.arn
  desired_count                      = var.inference_desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  health_check_grace_period_seconds  = 90
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  enable_execute_command             = false
  propagate_tags                     = "SERVICE"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.inference_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.inference.arn
    container_name   = local.inf_container
    container_port   = local.inf_port
  }

  service_registries {
    registry_arn = aws_service_discovery_service.inference.arn
  }

  lifecycle {
    ignore_changes = [desired_count, task_definition]
  }

  depends_on = [
    aws_lb_listener.http_redirect,
    aws_lb_listener.http_route,
    aws_lb_listener.https,
  ]

  tags = {
    Name = "${local.name_prefix}-inference-engine"
  }
}

######################################################################
# Application Auto Scaling (target tracking on CPU)
######################################################################

resource "aws_appautoscaling_target" "ml" {
  max_capacity       = var.ml_max_capacity
  min_capacity       = var.ml_min_capacity
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.ml.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ml_cpu" {
  name               = "${local.name_prefix}-ml-cpu-target"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ml.resource_id
  scalable_dimension = aws_appautoscaling_target.ml.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ml.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = var.cpu_target_utilization
    scale_in_cooldown  = 120
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_appautoscaling_target" "inference" {
  max_capacity       = var.inference_max_capacity
  min_capacity       = var.inference_min_capacity
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.inference.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "inference_cpu" {
  name               = "${local.name_prefix}-inference-cpu-target"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.inference.resource_id
  scalable_dimension = aws_appautoscaling_target.inference.scalable_dimension
  service_namespace  = aws_appautoscaling_target.inference.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = var.cpu_target_utilization
    scale_in_cooldown  = 120
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
