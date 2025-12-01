# GCP T4 GPU Deployment - Quick Reference

## Prerequisites

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
```

## Option 1: Automated Benchmark (Recommended)

```bash
# Run full deployment + benchmark pipeline
./scripts/gcp/deploy_t4_benchmark.sh \
    models/diarization_fast_cnn_optimized.onnx \
    your-project-id

# Results saved to: benchmark_t4_results.json
```

## Option 2: Manual Deployment

### Step 1: Create T4 Instance

```bash
gcloud compute instances create voiceflow-t4-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=50GB \
    --metadata="install-nvidia-driver=True"

# Cost: ~$0.70/hour (n1-standard-4 + T4)
```

### Step 2: Setup Instance

```bash
# SSH into instance
gcloud compute ssh voiceflow-t4-gpu --zone=us-central1-a

# Install dependencies
pip install onnxruntime-gpu numpy

# Verify GPU
nvidia-smi
```

### Step 3: Upload Model & Benchmark

```bash
# Upload model
gcloud compute scp models/diarization_fast_cnn_optimized.onnx \
    voiceflow-t4-gpu:~/model.onnx --zone=us-central1-a

# Upload benchmark script
gcloud compute scp scripts/gcp/benchmark_gpu.py \
    voiceflow-t4-gpu:~/benchmark_gpu.py --zone=us-central1-a

# SSH and run
gcloud compute ssh voiceflow-t4-gpu --zone=us-central1-a

# Run benchmark
python benchmark_gpu.py \
    --model model.onnx \
    --iterations 200 \
    --batch-sizes 1,4,8 \
    --providers cuda,tensorrt,cpu \
    --output-json results.json

# View results
cat results.json | python -m json.tool
```

## Option 3: Docker Deployment

```bash
# Upload docker-compose.gpu.yml and Dockerfile.gpu
gcloud compute scp docker-compose.gpu.yml Dockerfile.gpu \
    voiceflow-t4-gpu:~/ --zone=us-central1-a

# SSH into instance
gcloud compute ssh voiceflow-t4-gpu --zone=us-central1-a

# Install Docker & nvidia-docker2
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with Docker Compose
docker-compose -f docker-compose.gpu.yml up -d

# Check logs
docker-compose -f docker-compose.gpu.yml logs -f
```

## Expected Performance (GCP T4)

| Model | Hardware | P50 | P95 | P99 | Throughput |
|-------|----------|-----|-----|-----|------------|
| Fast CNN | T4 GPU | 1.5ms | 2.5ms | 3-5ms | 500-800 req/s |
| Fast CNN | CPU | 3.36ms | 4.2ms | 4.48ms | 297 req/s |

**Note**: These are **model inference only**. Add 15-48ms for end-to-end latency (network + Rust).

## Cost Analysis

| Instance Type | GPU | vCPUs | RAM | Cost/Hour | Cost/Month (24/7) |
|---------------|-----|-------|-----|-----------|-------------------|
| n1-standard-4 + T4 | T4 (16GB) | 4 | 15GB | $0.70 | $504 |
| n1-standard-8 + T4 | T4 (16GB) | 8 | 30GB | $0.96 | $691 |
| a2-highgpu-1g | A100 (40GB) | 12 | 85GB | $3.67 | $2,641 |

**Recommendation**: Use n1-standard-4 + T4 for production (best cost/performance ratio).

## Monitoring

### GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU metrics
nvidia-smi dmon -s pucvmet -o T -f gpu_metrics.log
```

### Application Metrics

```bash
# Prometheus metrics available at :9090/metrics
curl http://localhost:9090/metrics

# Grafana dashboard at :3000
# Default credentials: admin/admin
```

## Cleanup

```bash
# Delete instance
gcloud compute instances delete voiceflow-t4-gpu --zone=us-central1-a

# List all instances
gcloud compute instances list

# Stop instance (preserve data, cheaper than running)
gcloud compute instances stop voiceflow-t4-gpu --zone=us-central1-a

# Start stopped instance
gcloud compute instances start voiceflow-t4-gpu --zone=us-central1-a
```

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU status
nvidia-smi

# If not found, reinstall CUDA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-525 nvidia-cuda-toolkit

# Reboot
sudo reboot
```

### ONNX Runtime GPU Not Working

```bash
# Check CUDA version
nvcc --version

# Ensure onnxruntime-gpu matches CUDA version
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3  # CUDA 11.8

# Verify providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### High Latency

1. **Check GPU utilization**: Should be >80% during inference
2. **Verify provider**: Ensure using CUDAExecutionProvider, not CPU fallback
3. **Batch size**: Try batch_size=4 or 8 for better GPU utilization
4. **TensorRT**: Enable TensorrtExecutionProvider for additional 20-30% speedup

```python
# Enable TensorRT
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': True,
            'trt_max_workspace_size': 2147483648,  # 2GB
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
)
```

## Next Steps

1. ✅ Validate T4 performance (expected 2-4ms P99)
2. ⏳ Compare cost vs CPU deployment
3. ⏳ Load test with concurrent requests
4. ⏳ Implement auto-scaling (Kubernetes on GKE)
5. ⏳ Production deployment with CI/CD

See [docs/PERFORMANCE_ANALYSIS.md](../../voiceflow-ml/docs/PERFORMANCE_ANALYSIS.md) for detailed SLA definitions.
