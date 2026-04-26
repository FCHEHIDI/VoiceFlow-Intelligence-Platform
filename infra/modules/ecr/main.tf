######################################################################
# VoiceFlow ECR module
#
# Two repositories with scan-on-push and a lifecycle policy that keeps
# only the last N images. Tag mutability is environment-driven.
######################################################################

locals {
  name_prefix = "voiceflow-${var.env}"
  repos = {
    ml-service       = "ml-service"
    inference-engine = "inference-engine"
  }

  lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only the last ${var.max_image_count} images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.max_image_count
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# nosemgrep: terraform.aws.security.aws-ecr-mutable-image-tags.aws-ecr-mutable-image-tags
# `image_tag_mutability` is wired through a variable so the dev environment can
# use MUTABLE (faster iteration) while prod hard-pins IMMUTABLE (see
# infra/environments/prod/main.tf). The semgrep rule cannot statically resolve
# this and would otherwise block CI for both environments.
resource "aws_ecr_repository" "this" {
  for_each = local.repos

  name                 = "${local.name_prefix}/${each.value}"
  image_tag_mutability = var.image_tag_mutability
  force_delete         = var.force_delete

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name      = "${local.name_prefix}-${each.value}"
    Component = each.value
  }
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each = local.repos

  repository = aws_ecr_repository.this[each.key].name
  policy     = local.lifecycle_policy
}
