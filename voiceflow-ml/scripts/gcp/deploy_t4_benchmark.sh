#!/bin/bash
# Deploy and benchmark model on GCP T4 GPU instance
# Usage: ./scripts/gcp/deploy_t4_benchmark.sh <model-path> <project-id>

set -euo pipefail

# Configuration
MODEL_PATH="${1:-models/diarization_fast_cnn_optimized.onnx}"
PROJECT_ID="${2:-}"
INSTANCE_NAME="voiceflow-t4-benchmark"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 VoiceFlow T4 GPU Deployment & Benchmark${NC}"
echo "Model: $MODEL_PATH"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo ""

# Check if project ID provided
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: PROJECT_ID required${NC}"
    echo "Usage: $0 <model-path> <project-id>"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Model not found: $MODEL_PATH${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}📋 Setting project: $PROJECT_ID${NC}"
gcloud config set project "$PROJECT_ID"

# Create instance
echo -e "${YELLOW}🖥️  Creating T4 GPU instance...${NC}"
gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator=type="$GPU_TYPE",count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=50GB \
    --metadata="install-nvidia-driver=True" \
    || echo -e "${YELLOW}⚠️  Instance may already exist${NC}"

# Wait for instance to be ready
echo -e "${YELLOW}⏳ Waiting for instance to be ready...${NC}"
sleep 30

# Install dependencies on instance
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    set -e
    echo '🔧 Installing Python packages...'
    pip install -q onnxruntime-gpu numpy
    
    echo '✅ Verifying GPU...'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"

# Upload model
echo -e "${YELLOW}📤 Uploading model...${NC}"
gcloud compute scp "$MODEL_PATH" "$INSTANCE_NAME:~/model.onnx" --zone="$ZONE"

# Upload benchmark script
echo -e "${YELLOW}📤 Uploading benchmark script...${NC}"
gcloud compute scp scripts/gcp/benchmark_gpu.py "$INSTANCE_NAME:~/benchmark_gpu.py" --zone="$ZONE"

# Run benchmark
echo -e "${YELLOW}🏃 Running benchmark...${NC}"
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    python benchmark_gpu.py \
        --model model.onnx \
        --iterations 200 \
        --batch-sizes 1,4,8 \
        --providers cuda,cpu \
        --output-json benchmark_results.json
"

# Download results
echo -e "${YELLOW}📥 Downloading results...${NC}"
gcloud compute scp "$INSTANCE_NAME:~/benchmark_results.json" "./benchmark_t4_results.json" --zone="$ZONE"

# Display results
echo -e "${GREEN}📊 Benchmark Results:${NC}"
cat benchmark_t4_results.json | python -m json.tool

# Cleanup prompt
echo ""
echo -e "${YELLOW}🧹 Instance still running: $INSTANCE_NAME${NC}"
echo "To delete instance:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo -e "${GREEN}✅ Benchmark complete!${NC}"
echo "Results saved to: benchmark_t4_results.json"
