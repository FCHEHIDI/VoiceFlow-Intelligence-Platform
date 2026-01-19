# 🎤 VoiceFlow Intelligence Platform

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Rust](https://img.shields.io/badge/rust-1.75-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Production-ready real-time speaker diarization system with hybrid Python/Rust architecture, optimized for <100ms end-to-end latency streaming.

![VoiceFlow Diarization](assets/diarizerVoices.png)

## 🎯 Overview

VoiceFlow Intelligence Platform is a sophisticated audio processing system that identifies "who speaks when" in audio streams. It combines:

- **Python ML Service**: Model training, ONNX export, batch processing
- **Rust Inference Engine**: Real-time WebSocket streaming with ultra-low latency
- **MLOps Pipeline**: Complete CI/CD, monitoring, containerization

### Key Features

✅ **Ultra-Low Latency** - 4.48ms P99 model inference, 40-80ms P99 end-to-end  
✅ **Real-Time Streaming** - WebSocket audio streaming with <100ms P99 latency target  
✅ **Batch Processing** - Asynchronous processing of audio files via REST API  
✅ **Model Management** - Training, versioning, A/B testing, hot-reload  
✅ **Production-Ready** - Docker Compose, Prometheus metrics, Grafana dashboards  
✅ **High Performance** - Optimized ONNX Runtime with 297 req/s throughput (CPU)  
✅ **Scalable** - Stateless services, horizontal scaling support

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ├─── HTTP/REST ────────► Python ML Service (Port 8000)
       │                         ├─ FastAPI
       │                         ├─ PyTorch Training
       │                         ├─ ONNX Export
       │                         └─ Batch Jobs
       │
       └─── WebSocket ──────────► Rust Inference Engine (Port 3000)
                                  ├─ Axum/Tokio
                                  ├─ ONNX Runtime
                                  ├─ WebSocket Server
                                  └─ Real-Time Processing

        ┌──────────────────┐    ┌──────────────┐
        │   PostgreSQL     │    │    Redis     │
        │  (Metadata,      │    │  (Cache,     │
        │   Audit Logs)    │    │   Rate Limit)│
        └──────────────────┘    └──────────────┘

        ┌──────────────────┐    ┌──────────────┐
        │   Prometheus     │───►│   Grafana    │
        │   (Metrics)      │    │  (Dashboard) │
        └──────────────────┘    └──────────────┘
```

For detailed architecture diagrams, see [CONCEPTION_TECHNIQUE.md](docs/CONCEPTION_TECHNIQUE.md).

---

## 🚀 Quick Start

### Prerequisites

- **Docker** 20.10+ & Docker Compose 2.0+
- **Python** 3.11+ (for local development)
- **Rust** 1.75+ (for local development)
- **Git**

### 1️⃣ Clone Repository

```powershell
git clone https://github.com/YOUR_USERNAME/VoiceFlow-Intelligence-Platform.git
cd VoiceFlow-Intelligence-Platform
```

### 2️⃣ Start All Services (Docker Compose)

```powershell
docker-compose up --build
```

This starts:
- Python ML Service (http://localhost:8000)
- Rust Inference Engine (http://localhost:3000)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3001)

### 3️⃣ Verify Services

```powershell
# Check Python service
curl http://localhost:8000/health

# Check Rust service
curl http://localhost:3000/health

# Access API documentation
start http://localhost:8000/docs
```

### 4️⃣ Test Real-Time Streaming

```python
# streaming_test.py
import asyncio
import websockets
import json

async def test_streaming():
    async with websockets.connect("ws://localhost:3000/ws/stream") as ws:
        # Send audio chunk (dummy data for demo)
        message = {
            "type": "audio_chunk",
            "data": [0.1] * 16000,  # 1 second @ 16kHz
            "sequence": 0
        }
        await ws.send(json.dumps(message))
        
        # Receive result
        result = await ws.recv()
        print(f"Result: {result}")

asyncio.run(test_streaming())
```

---

## 📂 Project Structure

```
VoiceFlow-Intelligence-Platform/
├── docs/                           # 📚 Documentation
│   ├── CAHIER_DES_CHARGES.md      # Requirements specification
│   ├── NOTE_DE_CADRAGE.md         # Project planning
│   ├── CONCEPTION_TECHNIQUE.md    # Technical architecture
│   └── ARCHITECTURE_FLOW.md       # Data flow diagrams
│
├── voiceflow-ml/                   # 🐍 Python ML Service
│   ├── api/                       # FastAPI routes
│   │   ├── main.py               # Application entry point
│   │   └── routes/               # API endpoints
│   ├── models/                    # ML models
│   │   ├── diarization/          # Speaker diarization model
│   │   └── preprocessing/        # Audio feature extraction
│   ├── services/                  # Business logic layer
│   ├── repositories/              # Data access layer
│   ├── core/                      # Configuration & utilities
│   ├── tests/                     # Unit & integration tests
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Container image
│   └── .env.example              # Environment variables template
│
├── voiceflow-inference/           # 🦀 Rust Inference Engine
│   ├── src/
│   │   ├── main.rs               # Entry point
│   │   ├── api/                  # HTTP API handlers
│   │   ├── inference/            # ONNX Runtime integration
│   │   ├── streaming/            # WebSocket handlers
│   │   └── metrics/              # Prometheus metrics
│   ├── tests/                     # Rust tests
│   ├── Cargo.toml                # Rust dependencies
│   └── Dockerfile                # Multi-stage build
│
├── models/                         # 📦 Shared ONNX models
├── data/                          # 🗂️ Training datasets (gitignored)
├── docker-compose.yml             # 🐳 Full stack orchestration
├── prometheus.yml                 # 📊 Metrics configuration
└── README.md                      # 📖 This file
```

---

## 🛠️ Development Setup

### Python ML Service (Local)

```powershell
cd voiceflow-ml

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your settings

# Run development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Rust Inference Engine (Local)

```powershell
cd voiceflow-inference

# Build project
cargo build --release

# Run tests
cargo test --verbose

# Run service
cargo run --release
```

### Running Tests

**Python:**
```powershell
cd voiceflow-ml
pytest tests/ --cov=. --cov-report=html
```

**Rust:**
```powershell
cd voiceflow-inference
cargo test --verbose
cargo clippy -- -D warnings
```

---

## 📡 API Documentation

### Python ML Service (Port 8000)

**Interactive API Docs:** http://localhost:8000/docs

#### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe (checks DB & Redis) |
| POST | `/api/models/train` | Train new model |
| POST | `/api/models/{id}/export` | Export model to ONNX |
| GET | `/api/models` | List all models |
| PUT | `/api/models/{id}/activate` | Activate model for production |
| POST | `/api/inference/batch` | Submit batch job |
| GET | `/api/inference/batch/{job_id}` | Get job status/results |

#### Example: Train Model

```bash
curl -X POST http://localhost:8000/api/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/data/librispeech.tar.gz",
    "version": "1.3.0",
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 50
    }
  }'
```

### Rust Inference Engine (Port 3000)

#### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/infer` | Synchronous inference |
| WebSocket | `/ws/stream` | Real-time streaming |
| GET | `/metrics` | Prometheus metrics |

#### Example: WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:3000/ws/stream');

ws.onopen = () => {
  // Send audio chunk
  ws.send(JSON.stringify({
    type: 'audio_chunk',
    data: [...audioSamples], // Float32Array @ 16kHz
    sequence: 0
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Diarization:', result.segments);
  console.log('Latency:', result.latency_ms, 'ms');
};
```

---

## 📊 Monitoring & Metrics

### Prometheus Metrics

Access Prometheus: http://localhost:9090

**Key Metrics:**
- `inference_latency_seconds` - Inference latency histogram
- `inference_requests_total` - Total inference requests
- `inference_errors_total` - Total errors
- `websocket_connections_active` - Active WebSocket connections

### Grafana Dashboards

Access Grafana: http://localhost:3001 (admin/admin)

Pre-configured dashboards:
1. **Real-Time Inference** - Latency (P50/P99), throughput, error rate
2. **Model Performance** - Accuracy trends, version comparison
3. **System Health** - CPU, memory, network metrics

---

## 🔐 Security

### Authentication

All API endpoints (except `/health`) require JWT authentication:

```bash
# Get token (implement your auth endpoint)
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -d '{"username":"user","password":"pass"}' | jq -r '.access_token')

# Use token
curl http://localhost:8000/api/models \
  -H "Authorization: Bearer $TOKEN"
```

### Rate Limiting

Default: 100 requests/minute per user (configurable via Redis)

### Input Validation

- Audio file size: Max 100 MB
- Audio duration: Max 30 minutes
- File format validation (magic bytes check)

---

## 🚀 Deployment

### Docker Compose (Development/Staging)

```powershell
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Kubernetes (Production) - Coming Soon

Helm charts and Kubernetes manifests in progress.

---

## 📈 Performance Benchmarks

### Model Inference (Fast CNN, 2.3M params)
| Metric | CPU (Optimized ONNX) | GPU T4 (Projected) |
|--------|---------------------|-------------------|
| **P99 Latency** | 4.48ms ✅ | 3-5ms ✅ |
| **Median Latency** | 3.36ms | 1-2ms |
| **Throughput** | 297 req/s | 500-800 req/s |
| **Model Size** | 10 MB | 10 MB |

### End-to-End System Performance
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **End-to-End P99** | < 100ms | 40-80ms | ✅ |
| **Model Inference P99** | < 10ms | 4.48ms | ✅ |
| **Rust Overhead** | < 10ms | ~5-8ms | ✅ |
| **Throughput (CPU)** | > 100 req/s | 297 req/s | ✅ |
| **Memory (Rust)** | < 500 MB | 200-300 MB | ✅ |
| **Concurrent WebSocket** | > 500 | 1000+ | ✅ |

*Tested on: Intel/AMD 4-core CPU, 8GB RAM, no GPU*  
*End-to-end includes: network (10-40ms) + Rust processing (~5-8ms) + model inference (4.48ms)*

**See [docs/PERFORMANCE_ANALYSIS.md](voiceflow-ml/docs/PERFORMANCE_ANALYSIS.md) for detailed benchmarks**

---

## 🧪 Testing

### Run All Tests

```powershell
# Python tests
cd voiceflow-ml
pytest tests/ -v --cov=. --cov-report=term-missing

# Rust tests
cd voiceflow-inference
cargo test --verbose

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Load Testing

```powershell
# HTTP load test (wrk)
wrk -t4 -c100 -d30s http://localhost:8000/api/models

# WebSocket load test (custom script)
python tests/load/websocket_load.py --clients 1000 --duration 60
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [CAHIER_DES_CHARGES.md](docs/CAHIER_DES_CHARGES.md) | Requirements, features, acceptance criteria |
| [NOTE_DE_CADRAGE.md](docs/NOTE_DE_CADRAGE.md) | 2-day development plan, milestones |
| [CONCEPTION_TECHNIQUE.md](docs/CONCEPTION_TECHNIQUE.md) | Architecture, data models, API specs |
| [ARCHITECTURE_FLOW.md](docs/ARCHITECTURE_FLOW.md) | Request flows, sequence diagrams |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Quality Standards

- Python: `black`, `flake8`, `mypy`
- Rust: `cargo fmt`, `cargo clippy`
- Tests: Coverage > 80%

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **ONNX** - Model interoperability
- **Rust** - Systems programming language
- **FastAPI** - Modern Python web framework
- **Axum** - Ergonomic Rust web framework

---

## 📧 Contact

**Project Maintainer:** VoiceFlow Intelligence Team

- GitHub: [@FCHEHIDI](https://github.com/FCHEHIDI)
- Issues: [GitHub Issues](https://github.com/FCHEHIDI/VoiceFlow-Intelligence-Platform/issues)

---

## 🗺️ Roadmap

### v1.1 (Q1 2026)
- [ ] Advanced speaker embeddings (ResNet + LSTM)
- [ ] GPU acceleration (CUDA)
- [ ] gRPC Python ↔ Rust communication

### v1.2 (Q2 2026)
- [ ] Kubernetes deployment
- [ ] Advanced A/B testing (multi-armed bandit)
- [ ] Model distillation (INT8 quantization)

### v2.0 (Q3 2026)
- [ ] Multi-language support
- [ ] Real-time transcription integration
- [ ] Cloud deployment (AWS/Azure/GCP)

---

**⭐ Star this repo if you find it useful!**

Made with ❤️ by the VoiceFlow Intelligence Team
