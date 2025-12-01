# Containerized Model Development Workflow

## Overview

This document outlines the architecture for separating model training from the inference server, enabling independent development, versioning, and deployment of ML models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐       ┌────────────────────────┐  │
│  │  Model Training Repo    │       │  Inference Server Repo │  │
│  │  (voiceflow-models)     │       │  (voiceflow-inference) │  │
│  ├─────────────────────────┤       ├────────────────────────┤  │
│  │ - Train on GPU cluster  │       │ - Rust WebSocket API   │  │
│  │ - Export to ONNX        │──────>│ - ONNX Runtime         │  │
│  │ - Validate accuracy     │ ONNX  │ - Hot model reload     │  │
│  │ - Benchmark performance │       │ - Metrics (Prometheus) │  │
│  │ - CI/CD model registry  │       │ - Health checks        │  │
│  └─────────────────────────┘       └────────────────────────┘  │
│           │                                   │                 │
│           │ Publish                           │ Pull            │
│           ▼                                   ▼                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Model Registry (GCS/S3/Azure Blob)              │   │
│  │                                                          │   │
│  │  models/                                                │   │
│  │  ├── fast-cnn/                                          │   │
│  │  │   ├── v1.0.0/model.onnx (10MB, 4.48ms P99)          │   │
│  │  │   ├── v1.1.0/model.onnx (12MB, 3.2ms P99)           │   │
│  │  │   └── latest -> v1.1.0                              │   │
│  │  ├── sophisticated/                                     │   │
│  │  │   ├── v1.0.0/model.onnx (362MB, 40ms P99 GPU)       │   │
│  │  │   └── latest -> v1.0.0                              │   │
│  │  └── metadata.json (version history, benchmarks)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Model Contract Specification

### ONNX Input/Output Schema

All models MUST adhere to this contract:

```yaml
# model_contract.yaml
name: VoiceFlow Speaker Diarization Model
version: 1.0.0

input:
  name: audio
  type: float32
  shape: [batch_size, 48000]  # 1 second at 48kHz
  description: Raw audio waveform, mono, 48kHz sample rate

output:
  name: speaker_embeddings
  type: float32
  shape: [batch_size, num_speakers]
  description: Speaker activation probabilities (softmax)

metadata:
  sample_rate: 48000
  audio_duration: 1.0  # seconds
  num_speakers: 2  # or dynamic based on model
  
performance_requirements:
  cpu_p99_ms: 10.0
  gpu_p99_ms: 5.0
  model_size_mb: 50.0  # max
```

### Validation Script

```python
# scripts/validate_model_contract.py
import onnx
import numpy as np
import onnxruntime as ort

def validate_model_contract(model_path: str):
    """
    Validates that ONNX model adheres to VoiceFlow contract.
    
    Checks:
    - Input shape [batch, 48000]
    - Output shape [batch, num_speakers]
    - Input/output data types
    - Metadata attributes
    """
    model = onnx.load(model_path)
    
    # Check input
    input_tensor = model.graph.input[0]
    assert input_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert input_tensor.type.tensor_type.shape.dim[1].dim_value == 48000
    
    # Check output
    output_tensor = model.graph.output[0]
    assert output_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    
    # Runtime validation
    session = ort.InferenceSession(model_path)
    test_audio = np.random.randn(1, 48000).astype(np.float32)
    outputs = session.run(None, {input_tensor.name: test_audio})
    
    assert outputs[0].shape[0] == 1  # Batch size
    assert len(outputs[0].shape) == 2  # [batch, speakers]
    
    print("✅ Model contract validation passed!")
    return True
```

## Repository Structure

### Training Repository (`voiceflow-models`)

```
voiceflow-models/
├── README.md
├── pyproject.toml
├── .github/
│   └── workflows/
│       ├── train.yml          # Training CI
│       ├── benchmark.yml      # Performance validation
│       └── publish.yml        # Model registry upload
├── data/
│   ├── download_datasets.py  # AMI, LibriSpeech, VoxConverse
│   └── preprocessing/
├── models/
│   ├── fast_cnn.py           # Fast CNN architecture
│   ├── sophisticated.py       # Wav2Vec2-based model
│   └── base.py               # Base model interface
├── training/
│   ├── train.py              # Main training script
│   ├── augmentation.py       # Audio augmentation
│   └── metrics.py            # DER, accuracy metrics
├── export/
│   ├── to_onnx.py           # Export to ONNX
│   ├── optimize.py          # ONNX optimization
│   └── validate_contract.py # Contract validation
├── benchmarks/
│   ├── latency.py           # Latency benchmarks
│   ├── accuracy.py          # Accuracy validation
│   └── hardware_matrix.py   # CPU/GPU/TPU tests
└── scripts/
    ├── publish_model.py     # Upload to registry
    └── version_bump.py      # Semantic versioning
```

### Inference Repository (`voiceflow-inference`)

```
voiceflow-inference/
├── README.md
├── Cargo.toml
├── .github/
│   └── workflows/
│       ├── build.yml         # Build Rust binary
│       ├── test.yml          # Integration tests
│       └── deploy.yml        # GCP/AWS deployment
├── src/
│   ├── main.rs              # Main server
│   ├── websocket.rs         # WebSocket handler
│   ├── inference.rs         # ONNX Runtime wrapper
│   ├── model_loader.rs      # Hot reload logic
│   ├── metrics.rs           # Prometheus metrics
│   └── config.rs            # Configuration
├── models/
│   └── .gitignore           # Don't commit models
├── config/
│   ├── production.toml      # Production config
│   ├── staging.toml         # Staging config
│   └── development.toml     # Local dev config
├── scripts/
│   ├── download_model.sh    # Pull from registry
│   └── hot_reload.sh        # Trigger model reload
├── docker/
│   ├── Dockerfile.cpu       # CPU-optimized
│   ├── Dockerfile.gpu       # GPU-optimized
│   └── docker-compose.yml   # Local testing
└── tests/
    ├── integration/
    │   ├── test_websocket.rs
    │   └── test_inference.rs
    └── load/
        └── locust_test.py   # Load testing
```

## Model Versioning Strategy

### Semantic Versioning

```
MAJOR.MINOR.PATCH

MAJOR: Breaking changes to input/output contract
MINOR: New features, architecture changes (backward compatible)
PATCH: Bug fixes, optimization improvements
```

**Examples**:
- `v1.0.0`: Initial Fast CNN (4.48ms P99, 2.3M params)
- `v1.1.0`: Add speaker count detection (backward compatible)
- `v1.1.1`: Optimize ONNX graph (10% faster, no architecture change)
- `v2.0.0`: Change output to [batch, time, speakers] (BREAKING)

### Model Registry Structure

```
gs://voiceflow-models/  # Google Cloud Storage bucket

fast-cnn/
├── v1.0.0/
│   ├── model.onnx              # 10MB
│   ├── metadata.json           # Benchmark results, training config
│   ├── validation_report.json  # Accuracy metrics
│   └── checksum.sha256
├── v1.1.0/
│   ├── model.onnx
│   ├── metadata.json
│   ├── validation_report.json
│   └── checksum.sha256
└── latest -> v1.1.0            # Symlink to latest stable

sophisticated/
├── v1.0.0/
│   ├── model.onnx              # 362MB
│   ├── metadata.json
│   ├── validation_report.json
│   └── checksum.sha256
└── latest -> v1.0.0

manifest.json                    # All models catalog
```

### Metadata Example

```json
{
  "model_name": "fast-cnn",
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "git_commit": "a3f5c8d",
  "training": {
    "dataset": "AMI + LibriSpeech + VoxConverse",
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 0.001
  },
  "architecture": {
    "type": "cnn",
    "params": 2300000,
    "layers": 12
  },
  "performance": {
    "cpu": {
      "p50_ms": 3.36,
      "p99_ms": 4.48,
      "throughput_rps": 297
    },
    "gpu_t4": {
      "p50_ms": 1.5,
      "p99_ms": 2.8,
      "throughput_rps": 650
    }
  },
  "accuracy": {
    "der_test_set": 0.15,
    "precision": 0.92,
    "recall": 0.89
  },
  "contract": {
    "input_shape": [1, 48000],
    "output_shape": [1, 2],
    "sample_rate": 48000
  },
  "file": {
    "size_bytes": 10485760,
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "onnx_opset": 14
  }
}
```

## CI/CD Pipeline

### Training Pipeline (GitHub Actions)

```yaml
# .github/workflows/train_and_publish.yml
name: Train and Publish Model

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'training/**'

jobs:
  train:
    runs-on: ubuntu-latest-gpu  # Self-hosted with GPU
    steps:
      - uses: actions/checkout@v3
      
      - name: Train model
        run: |
          python training/train.py \
            --model fast-cnn \
            --epochs 50 \
            --output models/checkpoints/
      
      - name: Export to ONNX
        run: |
          python export/to_onnx.py \
            --checkpoint models/checkpoints/best.pth \
            --output models/fast_cnn.onnx
      
      - name: Optimize ONNX
        run: |
          python export/optimize.py \
            --input models/fast_cnn.onnx \
            --output models/fast_cnn_optimized.onnx
      
      - name: Validate contract
        run: |
          python export/validate_contract.py \
            --model models/fast_cnn_optimized.onnx
      
      - name: Benchmark performance
        run: |
          python benchmarks/latency.py \
            --model models/fast_cnn_optimized.onnx \
            --output benchmark_results.json
      
      - name: Version and publish
        env:
          GCS_BUCKET: gs://voiceflow-models
        run: |
          VERSION=$(python scripts/version_bump.py --auto)
          python scripts/publish_model.py \
            --model models/fast_cnn_optimized.onnx \
            --version $VERSION \
            --metadata benchmark_results.json \
            --bucket $GCS_BUCKET
```

### Inference Server Pipeline

```yaml
# .github/workflows/deploy_inference.yml
name: Deploy Inference Server

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
        default: 'latest'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model
        run: |
          ./scripts/download_model.sh \
            --model fast-cnn \
            --version ${{ github.event.inputs.model_version }} \
            --output models/
      
      - name: Build Docker image
        run: |
          docker build -f docker/Dockerfile.cpu -t voiceflow-inference:latest .
      
      - name: Deploy to GCP
        run: |
          gcloud run deploy voiceflow-inference \
            --image voiceflow-inference:latest \
            --region us-central1 \
            --allow-unauthenticated
```

## Hot Model Reload

### Rust Implementation

```rust
// src/model_loader.rs
use std::sync::{Arc, RwLock};
use std::path::Path;
use notify::{Watcher, RecursiveMode, watcher};
use ort::{Session, Environment};

pub struct ModelLoader {
    current_session: Arc<RwLock<Session>>,
    model_path: String,
}

impl ModelLoader {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let environment = Arc::new(Environment::builder().build()?);
        let session = Session::builder(&environment)?
            .with_model_from_file(model_path)?;
        
        Ok(Self {
            current_session: Arc::new(RwLock::new(session)),
            model_path: model_path.to_string(),
        })
    }
    
    pub fn watch_and_reload(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = watcher(tx, std::time::Duration::from_secs(2))?;
        watcher.watch(Path::new(&self.model_path), RecursiveMode::NonRecursive)?;
        
        loop {
            match rx.recv() {
                Ok(event) => {
                    println!("Model file changed, reloading...");
                    match self.reload() {
                        Ok(_) => println!("✅ Model reloaded successfully"),
                        Err(e) => eprintln!("❌ Failed to reload model: {}", e),
                    }
                }
                Err(e) => eprintln!("Watch error: {}", e),
            }
        }
    }
    
    pub fn reload(&self) -> Result<(), Box<dyn std::error::Error>> {
        let environment = Arc::new(Environment::builder().build()?);
        let new_session = Session::builder(&environment)?
            .with_model_from_file(&self.model_path)?;
        
        let mut session = self.current_session.write().unwrap();
        *session = new_session;
        
        Ok(())
    }
    
    pub fn get_session(&self) -> Arc<RwLock<Session>> {
        Arc::clone(&self.current_session)
    }
}
```

### Hot Reload Trigger Script

```bash
#!/bin/bash
# scripts/hot_reload.sh
# Trigger model reload without server restart

MODEL_VERSION="${1:-latest}"
MODEL_NAME="${2:-fast-cnn}"

echo "🔄 Hot reloading model: $MODEL_NAME@$MODEL_VERSION"

# Download new model
gsutil cp "gs://voiceflow-models/$MODEL_NAME/$MODEL_VERSION/model.onnx" \
    /tmp/new_model.onnx

# Validate contract
python scripts/validate_model_contract.py /tmp/new_model.onnx || exit 1

# Atomic swap (inotify will trigger reload)
mv /tmp/new_model.onnx models/current_model.onnx

echo "✅ Model reload triggered!"
echo "Check server logs for reload confirmation"
```

## Deployment Workflow

### Development Environment

```bash
# 1. Train new model
cd voiceflow-models
python training/train.py --model fast-cnn --epochs 50

# 2. Export and validate
python export/to_onnx.py --checkpoint checkpoints/best.pth
python export/validate_contract.py --model fast_cnn.onnx

# 3. Test locally
python benchmarks/latency.py --model fast_cnn.onnx

# 4. Publish to registry
python scripts/publish_model.py --model fast_cnn.onnx --version 1.2.0

# 5. Update inference server
cd ../voiceflow-inference
./scripts/download_model.sh --model fast-cnn --version 1.2.0
cargo run --release
```

### Production Deployment

```bash
# 1. Deploy model to staging
./scripts/hot_reload.sh 1.2.0 fast-cnn --env staging

# 2. Smoke test staging
curl -X POST https://staging.voiceflow.ai/health
curl -X POST https://staging.voiceflow.ai/infer -d @test_audio.json

# 3. Monitor metrics
# Check Grafana: latency P99, error rate, throughput

# 4. Promote to production (blue/green deployment)
./scripts/hot_reload.sh 1.2.0 fast-cnn --env production

# 5. Monitor production
# 15-minute bake time before declaring success
```

## Benefits of This Architecture

1. **Separation of Concerns**: ML team focuses on models, backend team on serving
2. **Independent Versioning**: Model updates don't require server rebuild
3. **Rollback Support**: Easy rollback to previous model version
4. **A/B Testing**: Run multiple model versions simultaneously
5. **Hot Reload**: Zero-downtime model updates
6. **CI/CD Ready**: Automated training, testing, and deployment
7. **Reproducibility**: Full model lineage (dataset, code, hyperparameters)

## Next Steps

1. ✅ Define ONNX contract specification
2. ⏳ Create `voiceflow-models` repository
3. ⏳ Implement model registry (GCS/S3)
4. ⏳ Build hot reload mechanism in Rust server
5. ⏳ Set up CI/CD pipelines
6. ⏳ Create model versioning automation
7. ⏳ Implement A/B testing infrastructure

See [docs/PERFORMANCE_ANALYSIS.md](./PERFORMANCE_ANALYSIS.md) for SLA requirements.
