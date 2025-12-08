#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Model path required${NC}"
    echo "Usage: $0 <model_path>"
    echo "Example: $0 models/diarization_transformer_optimized.onnx"
    exit 1
fi

MODEL_PATH="$1"
RESOURCE_GROUP="voiceflow-gpu-test"
WORKSPACE_NAME="voiceflow-ml"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 VoiceFlow Azure ML GPU Benchmark${NC}"
echo -e "${YELLOW}Model: $MODEL_PATH${NC}"
echo -e "${YELLOW}Workspace: $WORKSPACE_NAME${NC}"
echo ""

# Upload model as data asset
echo -e "${YELLOW}📤 Uploading model...${NC}"
MODEL_NAME=$(basename "$MODEL_PATH" .onnx)
MODEL_VERSION=$(date +%Y%m%d_%H%M%S)

az ml data create \
    --name "$MODEL_NAME" \
    --version "$MODEL_VERSION" \
    --type uri_file \
    --path "$MODEL_PATH" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --output none

echo -e "${GREEN}✅ Model uploaded: $MODEL_NAME:$MODEL_VERSION${NC}"

# Create results directory
mkdir -p benchmark_results

# Submit job
echo -e "${YELLOW}🎯 Submitting benchmark job...${NC}"
echo -e "${BLUE}This will:${NC}"
echo -e "${BLUE}  1. Spin up T4 GPU node (~60s)${NC}"
echo -e "${BLUE}  2. Install dependencies (~30s)${NC}"
echo -e "${BLUE}  3. Run 200 iterations (~5 min)${NC}"
echo -e "${BLUE}  4. Auto-scale to 0 (stop billing)${NC}"
echo ""

JOB_NAME=$(az ml job create \
    --file scripts/azure/benchmark_job.yml \
    --set inputs.model_path.path="azureml:$MODEL_NAME:$MODEL_VERSION" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query name \
    --output tsv)

echo -e "${GREEN}✅ Job submitted: $JOB_NAME${NC}"
echo ""
echo -e "${YELLOW}⏳ Monitoring job progress...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop monitoring (job will continue)${NC}"
echo ""

# Stream logs
az ml job stream \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" || true

# Wait for completion
echo ""
echo -e "${YELLOW}⏳ Waiting for job completion...${NC}"
az ml job show \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query "{Status:status,Duration:properties.duration}" \
    --output table

# Check if job succeeded
JOB_STATUS=$(az ml job show \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query status \
    --output tsv)

if [ "$JOB_STATUS" = "Completed" ]; then
    echo -e "${GREEN}✅ Job completed successfully${NC}"
    
    # Download results
    echo -e "${YELLOW}📥 Downloading results...${NC}"
    az ml job download \
        --name "$JOB_NAME" \
        --download-path "benchmark_results/azureml_$JOB_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --output none
    
    echo -e "${GREEN}✅ Results saved to: benchmark_results/azureml_$JOB_NAME${NC}"
    
    # Display results if JSON exists
    RESULT_FILE="benchmark_results/azureml_$JOB_NAME/outputs/benchmark_results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo ""
        echo -e "${BLUE}📊 Benchmark Results:${NC}"
        cat "$RESULT_FILE" | python -m json.tool
    fi
else
    echo -e "${RED}❌ Job failed with status: $JOB_STATUS${NC}"
    echo -e "${YELLOW}View logs at:${NC}"
    echo -e "${BLUE}https://ml.azure.com/runs/$JOB_NAME?wsid=/subscriptions/$(az account show --query id -o tsv)/resourcegroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE_NAME${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Benchmark complete!${NC}"
echo -e "${YELLOW}Cost: ~\$0.10 (compute auto-scaled to 0)${NC}"
