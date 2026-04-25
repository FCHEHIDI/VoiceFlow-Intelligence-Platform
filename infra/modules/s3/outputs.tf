output "models_bucket_name" {
  description = "Name of the models bucket."
  value       = aws_s3_bucket.models.id
}

output "models_bucket_arn" {
  description = "ARN of the models bucket."
  value       = aws_s3_bucket.models.arn
}

output "audio_bucket_name" {
  description = "Name of the audio uploads bucket."
  value       = aws_s3_bucket.audio.id
}

output "audio_bucket_arn" {
  description = "ARN of the audio uploads bucket."
  value       = aws_s3_bucket.audio.arn
}
