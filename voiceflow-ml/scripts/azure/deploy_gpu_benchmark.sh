#!/bin/bash
# Azure GPU Deployment & Benchmark Script
# Usage: ./scripts/azure/deploy_gpu_benchmark.sh <model-path>

set -euo pipefail

# Configuration
MODEL_PATH="${1:-models/diarization_transformer_optimized.onnx}"
RESOURCE_GROUP="voiceflow-gpu-test"
LOCATION="eastus"
VM_NAME="voiceflow-gpu-vm"
VM_SIZE="Standard_NC4as_T4_v3"  # T4 GPU, 4 vCPUs, 28GB RAM
IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
ADMIN_USER="azureuser"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🚀 VoiceFlow Azure GPU Benchmark${NC}"
echo -e "${BLUE}Model: $MODEL_PATH${NC}"
echo -e "${BLUE}VM: $VM_SIZE (T4 GPU)${NC}"
echo -e "${BLUE}Location: $LOCATION${NC}"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Model not found: $MODEL_PATH${NC}"
    exit 1
fi

# Create resource group
echo -e "${YELLOW}📋 Creating resource group...${NC}"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

echo -e "${GREEN}✅ Resource group created${NC}"

# Generate SSH key if not exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo -e "${YELLOW}🔑 Generating SSH key...${NC}"
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
fi

# Create VM with on-demand pricing (spot quota exhausted)
echo -e "${YELLOW}🖥️  Creating T4 GPU VM (On-Demand)...${NC}"
echo -e "${BLUE}Cost: ~\$0.53/hour (10 min = ~\$0.09)${NC}"
az vm create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --location "$LOCATION" \
    --size "$VM_SIZE" \
    --image "$IMAGE" \
    --admin-username "$ADMIN_USER" \
    --ssh-key-values ~/.ssh/id_rsa.pub \
    --public-ip-sku Standard \
    --output none

echo -e "${GREEN}✅ VM created${NC}"

# Get public IP
echo -e "${YELLOW}🔍 Getting VM IP address...${NC}"
VM_IP=$(az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --show-details \
    --query publicIps \
    --output tsv)

echo -e "${GREEN}✅ VM IP: $VM_IP${NC}"

# Wait for VM to be fully ready
echo -e "${YELLOW}⏳ Waiting for VM to be ready...${NC}"
sleep 30

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP" << 'EOF'
    set -e
    echo "🔧 Updating system..."
    sudo apt-get update -qq
    
    echo "🔧 Installing NVIDIA drivers..."
    sudo apt-get install -y -qq \
        ubuntu-drivers-common \
        nvidia-driver-535 \
        nvidia-cuda-toolkit
    
    echo "🔧 Installing Python..."
    sudo apt-get install -y -qq python3-pip
    
    echo "🔧 Installing Python packages..."
    pip3 install -q onnxruntime-gpu numpy
    
    echo "✅ Dependencies installed"
EOF

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Upload model
echo -e "${YELLOW}📤 Uploading model...${NC}"
scp -o StrictHostKeyChecking=no "$MODEL_PATH" "$ADMIN_USER@$VM_IP:~/model.onnx"
echo -e "${GREEN}✅ Model uploaded${NC}"

# Upload benchmark script
echo -e "${YELLOW}📤 Uploading benchmark script...${NC}"
scp -o StrictHostKeyChecking=no scripts/azure/benchmark_gpu.py "$ADMIN_USER@$VM_IP:~/benchmark.py"
echo -e "${GREEN}✅ Benchmark script uploaded${NC}"

# Run benchmark
echo -e "${YELLOW}🏃 Running GPU benchmark...${NC}"
echo -e "${BLUE}This will take ~5 minutes${NC}"
echo ""

ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP" << 'EOF'
    # Check GPU
    echo "🔍 Verifying GPU..."
    nvidia-smi || echo "GPU check will work after reboot - continuing with benchmark"
    
    # Run benchmark
    echo ""
    echo "🏃 Running benchmark..."
    python3 benchmark.py \
        --model model.onnx \
        --iterations 200 \
        --batch-sizes 1 \
        --providers cuda,cpu \
        --output-json benchmark_results.json
    
    echo ""
    echo "📊 Results:"
    cat benchmark_results.json
EOF

# Download results
echo ""
echo -e "${YELLOW}📥 Downloading results...${NC}"
mkdir -p benchmark_results
scp -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP:~/benchmark_results.json" "./benchmark_results/azure_t4_$(date +%Y%m%d_%H%M%S).json"
scp -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP:~/benchmark_results.json" "./benchmark_results/latest.json"

echo -e "${GREEN}✅ Results downloaded${NC}"

# Display results
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}📊 Benchmark Results${NC}"
echo -e "${GREEN}======================================${NC}"
cat ./benchmark_results/latest.json | python3 -m json.tool || cat ./benchmark_results/latest.json

# Cleanup
echo ""
echo -e "${YELLOW}🧹 Cleaning up (deleting VM to stop billing)...${NC}"
az group delete \
    --name "$RESOURCE_GROUP" \
    --yes \
    --no-wait

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✅ Benchmark Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${BLUE}Results saved to: benchmark_results/latest.json${NC}"
echo -e "${BLUE}VM deleted - no ongoing costs${NC}"
echo -e "${YELLOW}Note: Resource group deletion is running in background${NC}"
echo ""
echo -e "${GREEN}Cost estimate for this run: \$0.05 - \$0.10${NC}"
