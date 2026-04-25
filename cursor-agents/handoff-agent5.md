# Agent 5 Handoff — Terraform AWS Infrastructure

## Module tree

```
infra/
├── README.md
├── iam/
│   ├── ci-policy.json          (Agent 6)
│   └── ci-trust-policy.json    (Agent 6)
├── modules/
│   ├── networking/             VPC + 2 public/2 private/2 db subnets, single or multi-AZ NAT, ALB/ml/inference/RDS/Redis SGs, optional VPC flow logs
│   ├── secrets/                KMS CMK + Secrets Manager (jwt_secret_key, redis_auth_token); RDS-managed DB password
│   ├── rds/                    Aurora PostgreSQL Serverless v2 (15.x), KMS-encrypted, manage_master_user_password = true
│   ├── elasticache/            Redis 7.x replication group, transit + at-rest encryption, AUTH token
│   ├── s3/                     Models bucket (versioned, AES256, public-block) + audio bucket (7-day lifecycle)
│   ├── ecr/                    Two repos (ml-service, inference-engine) with scan_on_push, lifecycle keep-last-N
│   ├── ecs/                    Fargate cluster, task defs (secrets via Secrets Manager, never plain env), services, ALB w/ sticky-session TGs for WS, listener rules, autoscaling
│   └── monitoring/             SNS + email subscription, CloudWatch log groups, alarms (5xx>5%, P99>100ms, RunningTaskCount<min)
└── environments/
    ├── dev/                    main.tf · variables.tf · outputs.tf · backend.tf · versions.tf · terraform.tfvars.example
    └── prod/                   main.tf · variables.tf · outputs.tf · backend.tf · versions.tf · terraform.tfvars.example
```

## Key contracts (selected)

| Module | Required inputs | Selected outputs |
| --- | --- | --- |
| networking | `env`, `vpc_cidr`, `enable_multi_az_nat` | `vpc_id`, `public_subnet_ids`, `private_subnet_ids`, `db_subnet_ids`, `alb_security_group_id`, `ml_service_security_group_id`, `inference_security_group_id`, `rds_security_group_id`, `redis_security_group_id` |
| secrets | `env`, `create_kms_key` | `jwt_secret_arn`, `redis_auth_secret_arn`, `kms_key_arn`, `redis_auth_token` |
| rds | `env`, `db_subnet_ids`, `rds_security_group_id`, `min_capacity`, `max_capacity`, `kms_key_id` | `writer_endpoint`, `port`, `database_name`, `master_user_secret_arn` |
| elasticache | `env`, `db_subnet_ids`, `redis_security_group_id`, `auth_token`, `kms_key_id` | `primary_endpoint_address`, `port` |
| s3 | `env`, `audio_expiration_days` | `models_bucket_arn/name`, `audio_bucket_arn/name` |
| ecr | `env`, `image_tag_mutability` | `ml_repository_url`, `inference_repository_url` |
| ecs | networking ids, secret ARNs, image URIs, db/redis endpoints, ACM cert ARN | `cluster_name`, `ml_service_name`, `inference_service_name`, `alb_dns_name`, `alb_arn_suffix`, target group ARNs |
| monitoring | `env`, `alert_email`, `alb_arn_suffix`, ECS cluster/service names | `sns_topic_arn`, log group names |

Secret values are wired into ECS via `task_definition.containerDefinitions[*].secrets[*].valueFrom` (Secrets Manager ARN). They never appear in `environment`.

## Validation

```
$ terraform fmt -recursive               # clean
$ terraform -chdir=infra/environments/dev init -backend=false   # OK after retry
$ terraform -chdir=infra/environments/dev validate              # Success! The configuration is valid.
$ terraform -chdir=infra/environments/prod init -backend=false  # OK (used dev's plugin cache to bypass flaky network)
$ terraform -chdir=infra/environments/prod validate             # Success! The configuration is valid.
```

One issue was caught and fixed during validation:
* `aws_elasticache_parameter_group` only supports `name` (not `name_prefix`) — switched to a stable `name` and kept `lifecycle { create_before_destroy = true }`.

## Operator pre-`apply` checklist

Before running `terraform apply`, the operator must populate / create:

1. **`backend.tf`** placeholders:
   * S3 bucket name (already-provisioned remote-state bucket).
   * DynamoDB lock table name.
   * Region (defaults to `eu-west-1`).
2. **`terraform.tfvars`** values (copy from `terraform.tfvars.example`):
   * `aws_region`
   * `acm_certificate_arn` — ACM cert in the same region for the ALB HTTPS listener.
   * `alert_email` — recipient of the SNS alarm subscription (must confirm).
   * `ml_image`, `inference_image` — `:tag` references in ECR (created by Agent 6 CI).
3. AWS account-level prereqs:
   * Confirmed ACM certificate (DNS-validated).
   * SES out of sandbox (only if alarm e-mail must arrive promptly).
   * IAM permissions to assume `terraform` role.
4. **OIDC role for CI** (Agent 6): IAM role created from `infra/iam/ci-trust-policy.json` + `infra/iam/ci-policy.json`. The Terraform stack does not currently provision this role — see Agent 6 handoff.

## Run commands

```powershell
# dev
Set-Location infra/environments/dev
Copy-Item terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars, then:
terraform init
terraform plan  -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars

# prod (requires manual approval / off-hours window)
Set-Location ../prod
Copy-Item terraform.tfvars.example terraform.tfvars
terraform init
terraform plan  -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## Notes

* dev sizing: 1 NAT GW shared, Aurora 0.5–2.0 ACU, single Redis node, ECS `desired=1 max=2`, ECR `MUTABLE`, log retention 30 d.
* prod sizing: NAT per AZ, Aurora 1–8 ACU, Redis with replicas (multi-AZ), ECS `desired=2 max=10`, ECR `IMMUTABLE`, log retention 90 d, ALB / RDS deletion protection enabled.
* All resources tagged `Project = "voiceflow"`, `Environment = var.env`, `ManagedBy = "terraform"` via `default_tags` on the AWS provider.
