# AGENT 5 — Terraform AWS Infrastructure
**Role** : Senior AWS Solutions Architect + Terraform Engineer
**Duree estimee** : 6-8h
**Prerequis** : Agents 1, 2, 3, 4 termines

---

## Contexte

2 services a deployer sur AWS ECS Fargate (ADR-003) :
- `voiceflow-ml` : FastAPI port 8000 — batch jobs
- `voiceflow-inference` : Rust Axum port 3000 — WebSocket streaming

---

## Structure Terraform a creer

```
infra/
├── modules/
│   ├── networking/     # VPC, subnets, SGs, NAT GW
│   ├── ecs/            # Fargate cluster, task defs, services, IAM
│   ├── rds/            # Aurora PostgreSQL Serverless v2
│   ├── elasticache/    # Redis 7.x
│   ├── s3/             # Buckets models + audio-uploads
│   ├── secrets/        # Secrets Manager (JWT, DB, Redis)
│   ├── ecr/            # 2 repos ECR
│   └── monitoring/     # CloudWatch dashboards + alarms
├── environments/
│   ├── dev/  (main.tf, variables.tf, terraform.tfvars.example)
│   └── prod/ (main.tf, variables.tf, terraform.tfvars.example)
└── README.md
```

---

## Module `networking/`

```hcl
# VPC : 10.0.0.0/16
# Subnets publics ALB  : 10.0.1.0/24, 10.0.2.0/24 (2 AZ minimum)
# Subnets prives ECS   : 10.0.10.0/24, 10.0.11.0/24
# Subnets DB           : 10.0.20.0/24, 10.0.21.0/24
# NAT Gateway par AZ publique

# Security Groups (principe least privilege)
# alb_sg           : ingress 80/443 from 0.0.0.0/0
# ml_service_sg    : ingress 8000 from alb_sg ONLY
# inference_sg     : ingress 3000 from alb_sg + ml_service_sg
# rds_sg           : ingress 5432 from ml_service_sg ONLY
# redis_sg         : ingress 6379 from ml_service_sg + inference_sg
```

---

## Module `ecs/` — Dimensionnement

| Service | CPU | Memory | Min tasks | Max tasks |
|---------|-----|--------|-----------|-----------|
| ml-service | 1024 | 2048 MB | 1 | 4 |
| inference-engine | 512 | 1024 MB | 2 | 10 |

### Task Definition ml-service (extrait critique)

```hcl
secrets = [
  { name = "JWT_SECRET_KEY",    valueFrom = aws_secretsmanager_secret.jwt.arn },
  { name = "POSTGRES_PASSWORD", valueFrom = "${aws_secretsmanager_secret.db.arn}:password::" },
  { name = "REDIS_PASSWORD",    valueFrom = "${aws_secretsmanager_secret.redis.arn}:auth_token::" }
]
# JAMAIS de credentials dans environment{} — toujours dans secrets{}

health_check = {
  command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
  interval    = 30
  timeout     = 10
  retries     = 3
  startPeriod = 60
}

log_configuration = {
  log_driver = "awslogs"
  options = {
    "awslogs-group"         = "/voiceflow/ml-service"
    "awslogs-region"        = var.aws_region
    "awslogs-stream-prefix" = "ecs"
  }
}
```

### ALB — sticky sessions WebSocket

```hcl
resource "aws_lb_target_group" "inference" {
  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400  # 24h — requis pour WebSocket sessions
    enabled         = true
  }
}
```

### Auto Scaling inference-engine

```hcl
# Scale out si websocket_connections_active > 200 par task
resource "aws_appautoscaling_policy" "inference_websocket" {
  policy_type = "StepScaling"
  step_scaling_policy_configuration {
    adjustment_type = "ChangeInCapacity"
    step_adjustment {
      scaling_adjustment          = 2
      metric_interval_lower_bound = 0
    }
  }
}
```

---

## Module `rds/` — Aurora PostgreSQL Serverless v2

```hcl
resource "aws_rds_cluster" "voiceflow" {
  engine         = "aurora-postgresql"
  engine_version = "15.4"

  serverlessv2_scaling_configuration {
    min_capacity = 0.5  # dev : economique
    max_capacity = 4.0  # prod : ajuster
  }

  manage_master_user_password   = true  # Secrets Manager integre
  master_user_secret_kms_key_id = aws_kms_key.rds.arn

  # Subnet groupe prive uniquement
  db_subnet_group_name = aws_db_subnet_group.db.name
  vpc_security_group_ids = [aws_security_group.rds.id]
}
```

---

## Module `s3/`

```hcl
# Bucket modeles ONNX (versioning ON, chiffrement AES256)
resource "aws_s3_bucket" "models" {
  bucket = "voiceflow-models-${var.environment}-${random_id.suffix.hex}"
}
resource "aws_s3_bucket_versioning" "models" {
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_public_access_block" "models" {
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket audio uploads (lifecycle: expire apres 7 jours)
resource "aws_s3_bucket_lifecycle_configuration" "audio" {
  rule {
    expiration { days = 7 }
    status = "Enabled"
  }
}
```

---

## Environnements

### `environments/dev/terraform.tfvars.example`

```hcl
aws_region     = "eu-west-1"
environment    = "dev"
rds_min_acu    = 0.5
rds_max_acu    = 2.0
ml_min_tasks   = 1
ml_max_tasks   = 2
inf_min_tasks  = 1
inf_max_tasks  = 4
alert_email    = "alerts@example.com"
```

### `environments/prod/terraform.tfvars.example`

```hcl
aws_region     = "eu-west-1"
environment    = "prod"
rds_min_acu    = 1.0
rds_max_acu    = 8.0
ml_min_tasks   = 2
ml_max_tasks   = 6
inf_min_tasks  = 2
inf_max_tasks  = 20
multi_az       = true
```

---

## Contraintes de securite (OWASP / AWS Well-Architected)

- Secrets Manager pour TOUS les credentials (zero en dur)
- S3 : block public access sur TOUS les buckets
- RDS : subnet prive, jamais accessible depuis internet
- IAM Task Roles : least privilege (uniquement ce dont le service a besoin)
- KMS : encryption at rest pour RDS et S3
- VPC Flow Logs : activer pour audit

---

## Verification finale

```bash
cd infra/environments/dev
terraform init
terraform validate           # Zero erreur
terraform plan               # Zero erreur

# Apres apply
aws ecs describe-services   --cluster voiceflow-dev   --services ml-service inference-engine   --query 'services[].{name:serviceName,status:status,running:runningCount}'
```

---

## Handoff pour Agents 6, 7

Creer `cursor-agents/handoff-agent5.md` :
- [x] ECR repositories crees : URL ml-service et inference-engine notes
- [x] ECS cluster names : voiceflow-dev, voiceflow-prod
- [x] ALB DNS names : notes (pour smoke tests CI/CD)
- [x] Secrets Manager ARNs : JWT, DB, Redis notes
- [x] S3 bucket models : URL notee
