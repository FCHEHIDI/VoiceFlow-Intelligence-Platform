output "ml_repository_url" {
  description = "URL of the ml-service ECR repository (e.g. 123.dkr.ecr.<region>.amazonaws.com/voiceflow-<env>/ml-service)."
  value       = aws_ecr_repository.this["ml-service"].repository_url
}

output "ml_repository_arn" {
  description = "ARN of the ml-service ECR repository."
  value       = aws_ecr_repository.this["ml-service"].arn
}

output "ml_repository_name" {
  description = "Name of the ml-service ECR repository."
  value       = aws_ecr_repository.this["ml-service"].name
}

output "inference_repository_url" {
  description = "URL of the inference-engine ECR repository."
  value       = aws_ecr_repository.this["inference-engine"].repository_url
}

output "inference_repository_arn" {
  description = "ARN of the inference-engine ECR repository."
  value       = aws_ecr_repository.this["inference-engine"].arn
}

output "inference_repository_name" {
  description = "Name of the inference-engine ECR repository."
  value       = aws_ecr_repository.this["inference-engine"].name
}

output "repository_arns" {
  description = "Map of ECR repository ARNs."
  value = {
    "ml-service"       = aws_ecr_repository.this["ml-service"].arn
    "inference-engine" = aws_ecr_repository.this["inference-engine"].arn
  }
}
