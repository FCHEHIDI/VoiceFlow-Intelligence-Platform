######################################################################
# VoiceFlow monitoring module
#
# Provisions:
#   * SNS topic + email subscription for alerts
#   * CloudWatch log groups for ml-service and inference-engine ECS tasks
#   * Alarms:
#       - high error rate: ALB 5xx / total requests > N%
#       - p99 latency on the ALB > 100 ms
#       - inference RunningTaskCount < N
######################################################################

locals {
  name_prefix = "voiceflow-${var.env}"
  topic_name  = "${local.name_prefix}-alerts"
}

######################################################################
# SNS topic + email subscription
######################################################################

resource "aws_sns_topic" "alerts" {
  name = local.topic_name

  tags = {
    Name = local.topic_name
  }
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

######################################################################
# Log groups
######################################################################

resource "aws_cloudwatch_log_group" "ml" {
  name              = "/ecs/voiceflow-ml-${var.env}"
  retention_in_days = var.log_retention_days

  tags = {
    Name      = "/ecs/voiceflow-ml-${var.env}"
    Component = "ml-service"
  }
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/ecs/voiceflow-inference-${var.env}"
  retention_in_days = var.log_retention_days

  tags = {
    Name      = "/ecs/voiceflow-inference-${var.env}"
    Component = "inference-engine"
  }
}

######################################################################
# Alarms
######################################################################

# High error rate: 5xx / RequestCount > X% over 5 min
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${local.name_prefix}-high-error-rate"
  alarm_description   = "ALB 5xx responses exceed ${var.error_rate_threshold_percent}% of total requests over 5 minutes."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = var.error_rate_threshold_percent
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "error_rate"
    expression  = "IF(requests > 0, 100 * five_xx / requests, 0)"
    label       = "5xx error rate (%)"
    return_data = true
  }

  metric_query {
    id = "five_xx"
    metric {
      metric_name = "HTTPCode_Target_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
      }
    }
  }

  metric_query {
    id = "requests"
    metric {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
      }
    }
  }

  tags = {
    Name = "${local.name_prefix}-high-error-rate"
  }
}

# P99 latency on the ALB > 100 ms over 5 min
resource "aws_cloudwatch_metric_alarm" "latency_p99" {
  alarm_name          = "${local.name_prefix}-latency-p99"
  alarm_description   = "ALB target response time P99 > ${var.p99_latency_threshold_seconds * 1000} ms over 5 minutes."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = var.p99_latency_threshold_seconds
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]

  metric_name        = "TargetResponseTime"
  namespace          = "AWS/ApplicationELB"
  period             = 300
  extended_statistic = "p99"

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
  }

  tags = {
    Name = "${local.name_prefix}-latency-p99"
  }
}

# Inference running task count < N
resource "aws_cloudwatch_metric_alarm" "inference_tasks_low" {
  alarm_name          = "${local.name_prefix}-inference-tasks-low"
  alarm_description   = "Inference service running task count is below ${var.inference_min_running_tasks}."
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  threshold           = var.inference_min_running_tasks
  treat_missing_data  = "breaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]

  metric_name = "RunningTaskCount"
  namespace   = "ECS/ContainerInsights"
  period      = 60
  statistic   = "Average"

  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = var.inference_service_name
  }

  tags = {
    Name = "${local.name_prefix}-inference-tasks-low"
  }
}
