output "sns_topic_arn" {
  description = "ARN of the SNS alerts topic. Subscribe additional protocols (PagerDuty, Slack via Lambda, etc.) as needed."
  value       = aws_sns_topic.alerts.arn
}

output "ml_log_group_name" {
  description = "Name of the ml-service CloudWatch log group."
  value       = aws_cloudwatch_log_group.ml.name
}

output "ml_log_group_arn" {
  description = "ARN of the ml-service CloudWatch log group."
  value       = aws_cloudwatch_log_group.ml.arn
}

output "inference_log_group_name" {
  description = "Name of the inference-engine CloudWatch log group."
  value       = aws_cloudwatch_log_group.inference.name
}

output "inference_log_group_arn" {
  description = "ARN of the inference-engine CloudWatch log group."
  value       = aws_cloudwatch_log_group.inference.arn
}

output "alarm_names" {
  description = "Names of all alarms created by this module."
  value = [
    aws_cloudwatch_metric_alarm.high_error_rate.alarm_name,
    aws_cloudwatch_metric_alarm.latency_p99.alarm_name,
    aws_cloudwatch_metric_alarm.inference_tasks_low.alarm_name,
  ]
}
