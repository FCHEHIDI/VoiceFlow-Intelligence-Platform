#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

RESOURCE_GROUP="voiceflow-gpu-test"
WORKSPACE_NAME="voiceflow-ml"
LOCATION="eastus"
COMPUTE_NAME="gpu-cluster"

echo -e "${BLUE}🚀 Setting up Azure ML Workspace${NC}"
echo -e "${YELLOW}Resource Group: $RESOURCE_GROUP${NC}"
echo -e "${YELLOW}Workspace: $WORKSPACE_NAME${NC}"
echo -e "${YELLOW}Location: $LOCATION${NC}"
echo ""

# Create resource group if not exists
echo -e "${YELLOW}📋 Ensuring resource group exists...${NC}"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none 2>/dev/null || true
echo -e "${GREEN}✅ Resource group ready${NC}"

# Create Azure ML workspace
echo -e "${YELLOW}🏢 Creating Azure ML workspace...${NC}"
az ml workspace create \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
echo -e "${GREEN}✅ Workspace created${NC}"

# Create compute cluster (auto-scales 0-1)
echo -e "${YELLOW}💻 Creating GPU compute cluster...${NC}"
echo -e "${BLUE}VM: Standard_NC4as_T4_v3 (T4 GPU)${NC}"
echo -e "${BLUE}Auto-scale: 0-1 nodes (zero idle cost)${NC}"
az ml compute create \
    --name "$COMPUTE_NAME" \
    --type AmlCompute \
    --size Standard_NC4as_T4_v3 \
    --min-instances 0 \
    --max-instances 1 \
    --idle-time-before-scale-down 300 \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --output none
echo -e "${GREEN}✅ Compute cluster created${NC}"

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo -e "${YELLOW}Next: Run the benchmark with:${NC}"
echo -e "${BLUE}bash scripts/azure/run_azureml_benchmark.sh models/diarization_transformer_optimized.onnx${NC}"
