######################################################################
# Remote state backend.
#
# Operator must:
#   1. Create the S3 bucket (versioned, AES256 SSE, public access blocked).
#   2. Create the DynamoDB table for state locking (PK = LockID:String).
#   3. Replace the placeholders below with real names, OR pass them on the
#      command line:
#
#        terraform init \
#          -backend-config="bucket=my-tf-state-bucket" \
#          -backend-config="key=voiceflow/dev/terraform.tfstate" \
#          -backend-config="region=eu-west-1" \
#          -backend-config="dynamodb_table=my-tf-locks" \
#          -backend-config="encrypt=true"
#
# To run plan/validate without a remote backend (e.g. for CI lint),
# call:    terraform init -backend=false
######################################################################

terraform {
  backend "s3" {
    bucket         = "REPLACE_WITH_YOUR_TF_STATE_BUCKET"
    key            = "voiceflow/dev/terraform.tfstate"
    region         = "REPLACE_WITH_YOUR_AWS_REGION"
    dynamodb_table = "REPLACE_WITH_YOUR_TF_LOCK_TABLE"
    encrypt        = true
  }
}
