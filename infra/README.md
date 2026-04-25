# VoiceFlow Infrastructure (Terraform)

Terraform 1.5+ / AWS provider 5.x.

```
infra/
├── modules/
│   ├── networking/     # VPC, subnets, NAT, IGW, security groups
│   ├── secrets/        # Secrets Manager + customer-managed KMS key
│   ├── rds/            # Aurora PostgreSQL Serverless v2
│   ├── elasticache/    # Redis 7.x replication group
│   ├── s3/             # models + audio buckets (encryption + lifecycle)
│   ├── ecr/            # ml-service + inference-engine repos
│   ├── ecs/            # Fargate cluster, ALB, services, autoscaling
│   └── monitoring/     # SNS, CloudWatch log groups, alarms
├── environments/
│   ├── dev/  (single-NAT, 0.5–2 ACU, 1 task per service)
│   └── prod/ (multi-AZ NAT, 1–8 ACU, 2 desired tasks, IMMUTABLE images)
└── iam/                # CI OIDC role policy + trust policy (Agent 6)
```

## Prerequisites

- Terraform 1.5+
- AWS CLI 2.x configured (or `AWS_PROFILE` / OIDC role)
- An S3 bucket and DynamoDB table for the remote state backend
- An ACM certificate covering the public DNS name
- Two ECR images already pushed (Agent 6's CI workflow handles this)

## Run (dev)

```bash
cd infra/environments/dev
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: alert_email, ml_image, inference_image, acm_certificate_arn

terraform init \
  -backend-config="bucket=<state-bucket>" \
  -backend-config="key=voiceflow/dev/terraform.tfstate" \
  -backend-config="region=<aws-region>" \
  -backend-config="dynamodb_table=<lock-table>" \
  -backend-config="encrypt=true"

terraform validate
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

For a CI lint without remote state: `terraform init -backend=false && terraform validate`.

## Run (prod)

Same as dev but `cd infra/environments/prod` and use a different state key
(`voiceflow/prod/terraform.tfstate`).

## Outputs of interest

After `apply`, retrieve the most useful values:

```bash
terraform output -raw alb_dns_name      # set as Route53 alias
terraform output ml_repository_url
terraform output inference_repository_url
terraform output -raw jwt_secret_arn
terraform output -raw redis_auth_secret_arn
terraform output -raw master_user_secret_arn
```

> Note: many of the most-used outputs live on the `module.<name>` level
> (e.g. `module.ecs.alb_dns_name`). Add a top-level `outputs.tf` in each
> environment if you need them surfaced as bare `terraform output` keys.
