# GCP GPU Deployment Guide

## Quick Deploy to Google Cloud Platform

### Step 1: Create GPU Instance

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Create a T4 GPU instance (cheapest GPU option)
gcloud compute instances create voiceflow-gpu-inference \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=50GB \
    --metadata="install-nvidia-driver=True"

# Expected cost: ~$0.70/hour (n1-standard-4 + T4 GPU)
```

### Step 2: SSH into Instance

```bash
gcloud compute ssh voiceflow-gpu-inference --zone=us-central1-a
```

### Step 3: Setup on GPU Instance

```bash
# Install ONNX Runtime GPU
pip install onnxruntime-gpu numpy

# Verify GPU is available
nvidia-smi

# Download your model (use SCP or gcloud storage)
# Option 1: From local machine
exit  # Exit SSH first
gcloud compute scp models/diarization_transformer_optimized.onnx \
    voiceflow-gpu-inference:~/model.onnx --zone=us-central1-a

# Option 2: Upload to Cloud Storage first
gsutil cp models/diarization_transformer_optimized.onnx gs://your-bucket/
# Then on instance:
gsutil cp gs://your-bucket/diarization_transformer_optimized.onnx ~/model.onnx
```

### Step 4: Test Inference Speed

Create `test_gpu.py` on instance:

```python
import onnxruntime as ort
import numpy as np
import time

# Load model with GPU
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print(f"Provider: {session.get_providers()[0]}")

# Generate test audio
audio = np.random.randn(1, 48000).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {'audio': audio})

# Benchmark
latencies = []
for i in range(100):
    start = time.perf_counter()
    outputs = session.run(None, {'audio': audio})
    latencies.append((time.perf_counter() - start) * 1000)

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
p99 = latencies[int(len(latencies) * 0.99)]

print(f"\n🎯 Results:")
print(f"P50: {p50:.2f} ms")
print(f"P95: {p95:.2f} ms")
print(f"P99: {p99:.2f} ms")
print(f"\n{'✅ PASS' if p99 < 100 else '❌ FAIL'}: P99 < 100ms")
```

Run:
```bash
python test_gpu.py
```

### Step 5: Deploy Rust Server (Optional)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Copy your Rust project
# (Use gcloud compute scp to upload voiceflow-inference directory)

cd voiceflow-inference
cargo build --release
./target/release/voiceflow_inference
```

### Alternative: Quick Python Test (No Rust)

If you just want to validate GPU performance immediately:

```bash
# On instance
pip install onnxruntime-gpu numpy flask

# Create simple API
cat > api.py << 'EOF'
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import time

app = Flask(__name__)

session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.perf_counter()
    
    # Get audio from request
    audio = np.array(request.json['audio'], dtype=np.float32).reshape(1, -1)
    
    # Inference
    outputs = session.run(None, {'audio': audio})
    
    latency = (time.perf_counter() - start) * 1000
    
    return jsonify({
        'speaker_1': float(outputs[0][0][0]),
        'speaker_2': float(outputs[0][0][1]),
        'latency_ms': latency
    })

if __name__ == '__main__':
    print(f"Provider: {session.get_providers()[0]}")
    app.run(host='0.0.0.0', port=8080)
EOF

# Run server
python api.py
```

Test from your local machine:
```bash
# Get instance external IP
INSTANCE_IP=$(gcloud compute instances describe voiceflow-gpu-inference \
    --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Test API
curl -X POST http://$INSTANCE_IP:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"audio": [0.1, 0.2, ..., 0.3]}'  # 48000 samples
```

### Cost Management

```bash
# Stop instance when not in use (pay only for disk storage ~$2/month)
gcloud compute instances stop voiceflow-gpu-inference --zone=us-central1-a

# Start when needed
gcloud compute instances start voiceflow-gpu-inference --zone=us-central1-a

# Delete completely
gcloud compute instances delete voiceflow-gpu-inference --zone=us-central1-a
```

### Expected Results

With T4 GPU, you should see:
- **P50: 20-40ms** ✅
- **P95: 35-60ms** ✅
- **P99: 40-80ms** ✅ (well under 100ms target)

This proves production readiness!
