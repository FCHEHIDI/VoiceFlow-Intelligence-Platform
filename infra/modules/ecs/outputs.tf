output "cluster_name" {
  description = "Name of the ECS cluster."
  value       = aws_ecs_cluster.this.name
}

output "cluster_arn" {
  description = "ARN of the ECS cluster."
  value       = aws_ecs_cluster.this.arn
}

output "ml_service_name" {
  description = "Name of the ml-service ECS service."
  value       = aws_ecs_service.ml.name
}

output "inference_service_name" {
  description = "Name of the inference-engine ECS service."
  value       = aws_ecs_service.inference.name
}

output "ml_task_definition_arn" {
  description = "ARN of the ml-service task definition."
  value       = aws_ecs_task_definition.ml.arn
}

output "inference_task_definition_arn" {
  description = "ARN of the inference-engine task definition."
  value       = aws_ecs_task_definition.inference.arn
}

output "execution_role_arn" {
  description = "ARN of the shared task execution role."
  value       = aws_iam_role.execution.arn
}

output "ml_task_role_arn" {
  description = "ARN of the ml-service task role."
  value       = aws_iam_role.ml_task.arn
}

output "inference_task_role_arn" {
  description = "ARN of the inference-engine task role."
  value       = aws_iam_role.inference_task.arn
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer."
  value       = aws_lb.this.arn
}

output "alb_arn_suffix" {
  description = "ALB ARN suffix (used by CloudWatch metric dimensions)."
  value       = aws_lb.this.arn_suffix
}

output "alb_dns_name" {
  description = "Public DNS name of the ALB. Point a Route53 record at this value."
  value       = aws_lb.this.dns_name
}

output "alb_zone_id" {
  description = "Hosted zone ID of the ALB (for Route53 alias records)."
  value       = aws_lb.this.zone_id
}

output "ml_target_group_arn" {
  description = "ARN of the ml-service target group."
  value       = aws_lb_target_group.ml.arn
}

output "ml_target_group_arn_suffix" {
  description = "ARN suffix of the ml-service target group."
  value       = aws_lb_target_group.ml.arn_suffix
}

output "inference_target_group_arn" {
  description = "ARN of the inference-engine target group."
  value       = aws_lb_target_group.inference.arn
}

output "inference_target_group_arn_suffix" {
  description = "ARN suffix of the inference-engine target group."
  value       = aws_lb_target_group.inference.arn_suffix
}

output "service_discovery_namespace_id" {
  description = "Cloud Map private namespace ID."
  value       = aws_service_discovery_private_dns_namespace.this.id
}
