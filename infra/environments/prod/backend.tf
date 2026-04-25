######################################################################
# Remote state backend (prod).
#
# Operator must provide real values via -backend-config flags or by
# editing this file. Both prod and dev MUST use distinct keys (the
# `key` attribute below) within the same state bucket.
#
#   terraform init \
#     -backend-config="bucket=my-tf-state-bucket" \
#     -backend-config="key=voiceflow/prod/terraform.tfstate" \
#     -backend-config="region=eu-west-1" \
#     -backend-config="dynamodb_table=my-tf-locks" \
#     -backend-config="encrypt=true"
######################################################################

terraform {
  backend "s3" {
    bucket         = "REPLACE_WITH_YOUR_TF_STATE_BUCKET"
    key            = "voiceflow/prod/terraform.tfstate"
    region         = "REPLACE_WITH_YOUR_AWS_REGION"
    dynamodb_table = "REPLACE_WITH_YOUR_TF_LOCK_TABLE"
    encrypt        = true
  }
}
