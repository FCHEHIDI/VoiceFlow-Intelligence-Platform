# 🏗️ CONCEPTION TECHNIQUE - VoiceFlow Intelligence Platform

## 1. ARCHITECTURE GLOBALE

### 1.1 Vue d'Ensemble - Architecture Hybride

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │  Web/Mobile App │    │  WebSocket Client (Real-Time)    │    │
│  │  (HTTP REST)    │    │  (Audio Streaming)               │    │
│  └────────┬────────┘    └──────────────┬───────────────────┘    │
└───────────┼────────────────────────────┼────────────────────────┘
            │                            │
            │ HTTP/REST                  │ WebSocket
            │                            │
┌───────────┼────────────────────────────┼───────────────────────┐
│           ▼                            ▼                       │
│  ┌────────────────────┐         ┌──────────────────────┐       │
│  │  PYTHON ML SERVICE │         │  RUST INFERENCE      │       │
│  │  (Port 8000)       │◄────────┤  ENGINE              │       │
│  │                    │  HTTP   │  (Port 3000)         │       │
│  │  - FastAPI         │         │                      │       │
│  │  - PyTorch         │         │  - Axum/Tokio        │       │
│  │  - Model Training  │         │  - ONNX Runtime      │       │
│  │  - ONNX Export     │         │  - WebSocket Server  │       │
│  │  - Batch Jobs      │         │  - Real-Time Infer.  │       │
│  └────────┬───────────┘         └──────────┬───────────┘       │
│           │                                │                   │
│           │                                │                   │
│  ┌────────┴────────┬──────────┐   ┌────────┴────────┐          │
│  │                 │          │   │                 │          │
│  ▼                 ▼          ▼   ▼                 ▼          │
│ ┌──────────┐  ┌────────┐  ┌─────────┐      ┌─────────────┐     │
│ │PostgreSQL│  │ Redis  │  │ ONNX    │      │ Prometheus  │     │
│ │(Metadata)│  │(Cache) │  │ Models  │      │ (Metrics)   │     │
│ │          │  │        │  │ Shared  │      │             │     │
│ └──────────┘  └────────┘  └─────────┘      └──────┬──────┘     │
│                                                   │            │
│                                             ┌─────┴────────┐   │
│                                             │   Grafana    │   │
│                                             │ (Dashboards) │   │
│                                             └──────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Principes Architecturaux

**Separation of Concerns:**
- **Python:** ML Training, Data Management, Complex Business Logic
- **Rust:** Real-Time Inference, High-Performance Streaming, Low-Latency Processing

**Design Patterns:**
- **Data Mapper Pattern** (Python): Séparation Repository ↔ Service ↔ Controller
- **Repository Pattern** (Rust): Abstraction accès données et modèles
- **Circuit Breaker:** Protection inter-service communication
- **Retry with Backoff:** Résilience appels externes

**Stateless Services:**
- Aucun état partagé en mémoire (scale horizontal)
- Session/State dans Redis ou PostgreSQL

---

## 2. ARCHITECTURE LAYERED - PYTHON ML SERVICE

### 2.1 Structure des Layers

```
voiceflow-ml/
│
├── api/                      # PRESENTATION LAYER
│   ├── main.py              # FastAPI App + CORS + Middleware
│   ├── dependencies.py      # Dependency Injection (DB session, etc.)
│   └── routes/
│       ├── models.py        # POST /api/models/train, export
│       ├── inference.py     # POST /api/inference/batch
│       └── health.py        # GET /health, /ready
│
├── services/                 # BUSINESS LOGIC LAYER
│   ├── training_service.py  # Orchestrate training workflow
│   ├── export_service.py    # PyTorch → ONNX conversion logic
│   └── inference_service.py # Batch inference coordination
│
├── repositories/             # DATA ACCESS LAYER
│   ├── model_repository.py  # CRUD models table
│   ├── job_repository.py    # CRUD training_jobs table
│   └── audit_repository.py  # Audit logs
│
├── models/                   # DOMAIN LAYER (ML Models)
│   ├── diarization/
│   │   ├── model.py         # PyTorch model architecture
│   │   ├── train.py         # Training loop
│   │   ├── evaluate.py      # Validation metrics
│   │   └── export_onnx.py   # ONNX export with optimizations
│   ├── embeddings/          # Speaker embeddings (future)
│   └── preprocessing/       # Audio feature extraction
│
├── core/                     # SHARED UTILITIES
│   ├── config.py            # Settings (pydantic BaseSettings)
│   ├── database.py          # SQLAlchemy engine, session
│   ├── redis_client.py      # Redis connection pool
│   └── logging_config.py    # Structured logging (structlog)
│
└── tests/                    # TEST LAYER
    ├── test_services.py
    ├── test_repositories.py
    └── test_api.py
```

### 2.2 Data Flow - Batch Processing

```
1. Client Request
   │
   ▼
2. API Controller (routes/inference.py)
   - Validate request (pydantic model)
   - Authentication check (JWT)
   - Rate limiting (Redis)
   │
   ▼
3. Service Layer (inference_service.py)
   - Business logic: queue job
   - Call Repository to persist job
   │
   ▼
4. Repository Layer (job_repository.py)
   - Insert into PostgreSQL training_jobs table
   - Return job_id
   │
   ▼
5. Background Worker (Celery/RQ)
   - Dequeue job
   - Load audio file
   - Preprocess (feature extraction)
   - Call Rust service /infer
   - Store results in DB
   │
   ▼
6. Response to Client
   - Polling: GET /api/jobs/{job_id}
   - Webhook: POST to callback_url
```

---

## 3. ARCHITECTURE LAYERED - RUST INFERENCE ENGINE

### 3.1 Structure des Modules

```
voiceflow-inference/
│
├── src/
│   ├── main.rs                    # Entry point, Axum server setup
│   │
│   ├── api/                       # HTTP LAYER
│   │   ├── mod.rs                # Router config
│   │   ├── handlers.rs           # POST /infer, GET /health
│   │   └── middleware.rs         # Logging, CORS, Auth
│   │
│   ├── inference/                 # INFERENCE LAYER
│   │   ├── mod.rs
│   │   ├── onnx_runtime.rs       # ONNX model loading + inference
│   │   ├── model_manager.rs      # Model versioning, hot-reload
│   │   └── preprocessing.rs      # Audio feature extraction
│   │
│   ├── streaming/                 # WEBSOCKET LAYER
│   │   ├── mod.rs
│   │   ├── websocket.rs          # WebSocket handler
│   │   ├── audio_buffer.rs       # Circular buffer for audio chunks
│   │   └── protocol.rs           # Message format (JSON/binary)
│   │
│   ├── metrics/                   # OBSERVABILITY LAYER
│   │   ├── mod.rs
│   │   └── prometheus.rs         # Metrics registry + exporters
│   │
│   └── core/                      # SHARED UTILITIES
│       ├── config.rs             # Settings (serde)
│       ├── error.rs              # Custom error types
│       └── state.rs              # AppState (Arc<ModelManager>)
│
└── tests/
    ├── inference_tests.rs
    └── integration_tests.rs
```

### 3.2 Data Flow - Real-Time Streaming

```
1. Client WebSocket Connection
   ws://localhost:3000/ws/stream
   │
   ▼
2. WebSocket Handler (streaming/websocket.rs)
   - Upgrade HTTP → WebSocket
   - Create audio buffer (1s chunks)
   │
   ▼
3. Audio Chunk Reception Loop
   while let Some(msg) = ws.recv().await {
       - Validate audio format (16kHz mono)
       - Append to buffer
       │
       ▼
4. Feature Extraction (inference/preprocessing.rs)
       - Compute MFCC/mel-spectrogram
       - Normalize features
       │
       ▼
5. ONNX Inference (inference/onnx_runtime.rs)
       - Load model (cached in Arc<Session>)
       - run_inference(features) → speaker_probs
       - Post-processing (argmax, confidence)
       │
       ▼
6. Stream Result Back
       - Format JSON: {start, end, speaker_id, confidence}
       - ws.send(result).await
   }
```

---

## 4. MODÈLE DE DONNÉES (PostgreSQL)

### 4.1 Schema Global

```sql
-- ============================================
-- TABLE: models
-- Description: Métadonnées des modèles ML
-- ============================================
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version VARCHAR(50) NOT NULL UNIQUE,
    architecture VARCHAR(100),  -- Ex: "ResNet18-LSTM"
    onnx_path TEXT NOT NULL,     -- /models/model-v1.2.3.onnx
    pytorch_checkpoint TEXT,     -- Optional backup
    
    -- Metrics
    accuracy FLOAT,
    f1_score FLOAT,
    inference_latency_ms INT,    -- Measured during export
    
    -- Status
    status VARCHAR(20) DEFAULT 'inactive', -- active|inactive|deprecated
    is_production BOOLEAN DEFAULT FALSE,   -- Currently used in Rust service
    
    -- Metadata
    training_dataset TEXT,
    hyperparameters JSONB,       -- {lr: 0.001, batch_size: 32, ...}
    trained_by VARCHAR(100),     -- User/system
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CHECK (status IN ('active', 'inactive', 'deprecated'))
);

CREATE INDEX idx_models_version ON models(version);
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_is_production ON models(is_production);

-- ============================================
-- TABLE: training_jobs
-- Description: Historique des entraînements
-- ============================================
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    
    -- Job info
    status VARCHAR(20) DEFAULT 'pending', -- pending|running|completed|failed
    progress INT DEFAULT 0,               -- 0-100%
    
    -- Config
    dataset_path TEXT,
    hyperparameters JSONB,
    
    -- Timings
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INT,
    
    -- Results
    final_loss FLOAT,
    validation_accuracy FLOAT,
    error_message TEXT,  -- If failed
    
    -- Metadata
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    
    CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE INDEX idx_jobs_status ON training_jobs(status);
CREATE INDEX idx_jobs_model_id ON training_jobs(model_id);

-- ============================================
-- TABLE: inference_logs
-- Description: Audit trail de toutes les inférences
-- ============================================
CREATE TABLE inference_logs (
    id BIGSERIAL PRIMARY KEY,
    
    -- Request info
    request_id UUID NOT NULL,
    user_id VARCHAR(100),
    model_version VARCHAR(50) NOT NULL,
    
    -- Input
    audio_duration_sec FLOAT,
    audio_format VARCHAR(20),  -- wav, mp3, pcm
    
    -- Output
    num_speakers INT,
    confidence_avg FLOAT,
    
    -- Performance
    latency_ms INT,
    processing_type VARCHAR(20), -- streaming|batch
    
    -- Metadata
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Privacy: No audio content stored
    -- Only metadata for analytics
    
    CHECK (processing_type IN ('streaming', 'batch'))
);

CREATE INDEX idx_inference_logs_timestamp ON inference_logs(timestamp DESC);
CREATE INDEX idx_inference_logs_model_version ON inference_logs(model_version);
CREATE INDEX idx_inference_logs_user_id ON inference_logs(user_id);

-- Partitioning by month for performance (optional)
-- CREATE TABLE inference_logs_2025_11 PARTITION OF inference_logs
--   FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- ============================================
-- TABLE: users (simplified JWT auth)
-- Description: Utilisateurs avec API keys
-- ============================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL,  -- bcrypt hash
    
    -- Rate limiting
    rate_limit_rpm INT DEFAULT 100,      -- Requests per minute
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- ============================================
-- TABLE: model_routing (A/B Testing)
-- Description: Traffic split configuration
-- ============================================
CREATE TABLE model_routing (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    traffic_percent INT NOT NULL,        -- 0-100
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CHECK (traffic_percent >= 0 AND traffic_percent <= 100)
);

-- Ensure total traffic = 100%
CREATE OR REPLACE FUNCTION check_total_traffic()
RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT SUM(traffic_percent) FROM model_routing WHERE is_active = TRUE) != 100 THEN
        RAISE EXCEPTION 'Total active traffic must equal 100%%';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_traffic_total
AFTER INSERT OR UPDATE ON model_routing
FOR EACH STATEMENT EXECUTE FUNCTION check_total_traffic();
```

### 4.2 Relations et Cardinalités

```
models (1) ──────< (N) training_jobs
  │
  │ (referenced by model_version)
  │
  └──────< (N) inference_logs

users (1) ──────< (N) inference_logs

model_routing (1:1) ────── models (via model_version)
```

---

## 5. API CONTRACTS

### 5.1 Python ML Service API

#### Endpoint: `POST /api/models/train`
**Description:** Déclenche un entraînement de modèle

**Request:**
```json
{
  "dataset_path": "/data/librispeech-train.tar.gz",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "architecture": "ResNet18-LSTM"
  },
  "version": "1.3.0"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2025-11-29T10:00:00Z"
}
```

---

#### Endpoint: `POST /api/models/{model_id}/export`
**Description:** Exporte un modèle PyTorch vers ONNX

**Response (200 OK):**
```json
{
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "onnx_path": "/models/model-v1.3.0.onnx",
  "optimizations": ["graph_optimization_level_3", "quantization_fp16"],
  "exported_at": "2025-11-29T11:30:00Z"
}
```

---

#### Endpoint: `POST /api/inference/batch`
**Description:** Traitement batch d'un fichier audio

**Request (multipart/form-data):**
```
file: audio.wav (binary)
callback_url: https://client.com/webhook (optional)
```

**Response (202 Accepted):**
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "pending",
  "estimated_duration_sec": 12
}
```

---

#### Endpoint: `GET /api/inference/batch/{job_id}`
**Description:** Récupère le statut et résultats

**Response (200 OK - Completed):**
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "results": {
    "audio_duration_sec": 30.5,
    "num_speakers": 2,
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "speaker_id": "SPEAKER_00",
        "confidence": 0.95
      },
      {
        "start": 5.2,
        "end": 10.8,
        "speaker_id": "SPEAKER_01",
        "confidence": 0.89
      }
    ]
  },
  "processing_time_ms": 4520,
  "model_version": "1.2.3"
}
```

---

### 5.2 Rust Inference Engine API

#### Endpoint: `POST /infer`
**Description:** Inférence synchrone sur audio segment

**Request:**
```json
{
  "audio": [0.123, -0.045, 0.678, ...],  // PCM float32 array
  "sample_rate": 16000,
  "model_version": "1.2.3"  // optional, uses production default
}
```

**Response (200 OK):**
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "speaker_id": "SPEAKER_00",
      "confidence": 0.92
    }
  ],
  "latency_ms": 78,
  "model_version": "1.2.3"
}
```

---

#### WebSocket: `ws://host:3000/ws/stream`
**Description:** Streaming temps-réel audio

**Message Protocol (Client → Server):**
```json
{
  "type": "audio_chunk",
  "data": [0.123, -0.045, ...],  // 1s of audio @ 16kHz = 16000 samples
  "timestamp": 1234567890.123,
  "sequence": 42
}
```

**Message Protocol (Server → Client):**
```json
{
  "type": "diarization_result",
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "speaker_id": "SPEAKER_00",
      "confidence": 0.94
    }
  ],
  "latency_ms": 82,
  "sequence": 42
}
```

**Control Messages:**
```json
// Client → Server: End stream
{"type": "end_stream"}

// Server → Client: Error
{
  "type": "error",
  "code": "INVALID_AUDIO_FORMAT",
  "message": "Expected 16kHz mono PCM"
}
```

---

### 5.3 OpenAPI Specification (excerpt)

```yaml
openapi: 3.0.0
info:
  title: VoiceFlow ML API
  version: 1.0.0
  description: Speaker Diarization ML Training & Management

servers:
  - url: http://localhost:8000
    description: Development

paths:
  /api/models/train:
    post:
      summary: Train new diarization model
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TrainingRequest'
      responses:
        '202':
          description: Training job created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobResponse'
        '401':
          description: Unauthorized
        '429':
          description: Rate limit exceeded

components:
  schemas:
    TrainingRequest:
      type: object
      required:
        - dataset_path
        - version
      properties:
        dataset_path:
          type: string
          example: "/data/train.tar.gz"
        hyperparameters:
          type: object
          additionalProperties: true
        version:
          type: string
          pattern: '^\d+\.\d+\.\d+$'
          example: "1.3.0"
    
    JobResponse:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [pending, running, completed, failed]
        created_at:
          type: string
          format: date-time

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

---

## 6. STRATÉGIE DE TEST

### 6.1 Pyramide de Tests

```
           /\
          /  \  E2E Tests (10%)
         /────\
        /      \  Integration Tests (30%)
       /────────\
      /          \  Unit Tests (60%)
     /────────────\
```

### 6.2 Tests Unitaires (Python)

**Exemple: `tests/test_services.py`**
```python
import pytest
from unittest.mock import Mock, patch
from services.training_service import TrainingService

@pytest.fixture
def mock_repository():
    return Mock()

def test_create_training_job_success(mock_repository):
    # Arrange
    service = TrainingService(mock_repository)
    mock_repository.create_job.return_value = "job-123"
    
    # Act
    job_id = service.create_training_job(
        dataset_path="/data/train.tar.gz",
        hyperparameters={"lr": 0.001}
    )
    
    # Assert
    assert job_id == "job-123"
    mock_repository.create_job.assert_called_once()

def test_export_onnx_invalid_model_id(mock_repository):
    service = TrainingService(mock_repository)
    mock_repository.get_model.return_value = None
    
    with pytest.raises(ValueError, match="Model not found"):
        service.export_to_onnx("invalid-id")
```

**Coverage Target:** 80%+ with pytest-cov
```bash
pytest tests/ --cov=services --cov=repositories --cov-report=html
```

---

### 6.3 Tests Unitaires (Rust)

**Exemple: `tests/inference_tests.rs`**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_model_inference_output_shape() {
        // Arrange
        let model = ModelRunner::new("tests/fixtures/model.onnx").unwrap();
        let input = vec![0.5; 16000]; // 1s @ 16kHz
        
        // Act
        let output = model.run_inference(&input).unwrap();
        
        // Assert
        assert_eq!(output.len(), 2); // 2 speakers
        assert!(output[0] >= 0.0 && output[0] <= 1.0); // Valid probability
    }

    #[tokio::test]
    async fn test_websocket_echo() {
        // Arrange
        let app = create_test_app();
        let client = TestClient::new(app);
        
        // Act
        let ws = client.websocket("/ws/stream").await.unwrap();
        ws.send(b"test_audio_data").await.unwrap();
        let response = ws.recv().await.unwrap();
        
        // Assert
        assert!(response.contains("diarization_result"));
    }
}
```

**Quality Checks:**
```bash
cargo test --verbose
cargo clippy -- -D warnings
cargo fmt --check
```

---

### 6.4 Tests d'Intégration

**Test Python → Rust Communication:**
```python
def test_python_calls_rust_inference():
    # Start both services (test containers)
    python_client = TestClient(app)
    
    # Upload model via Python
    response = python_client.post("/api/models/upload", files={"file": model_bytes})
    model_id = response.json()["model_id"]
    
    # Trigger inference via Rust
    audio_data = load_test_audio("sample.wav")
    rust_response = requests.post(
        "http://rust-service:3000/infer",
        json={"audio": audio_data, "model_version": model_id}
    )
    
    assert rust_response.status_code == 200
    assert "segments" in rust_response.json()
```

---

### 6.5 Tests End-to-End

**Scénario: Streaming Audio complet**
```python
async def test_e2e_streaming_workflow():
    # 1. Upload & activate model
    model_version = await upload_and_activate_model()
    
    # 2. Establish WebSocket connection
    async with websockets.connect("ws://localhost:3000/ws/stream") as ws:
        # 3. Stream 30s of audio
        audio = load_audio("tests/fixtures/conversation_30s.wav")
        chunks = split_audio_into_chunks(audio, chunk_size_sec=1)
        
        results = []
        for i, chunk in enumerate(chunks):
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "data": chunk.tolist(),
                "sequence": i
            }))
            
            response = await ws.recv()
            result = json.loads(response)
            results.append(result)
            
            # Assert latency < 100ms
            assert result["latency_ms"] < 100
        
        # 4. Verify diarization results
        assert len(results) == 30
        assert all(r["segments"] for r in results)
```

---

### 6.6 Load Testing

**Tool:** wrk (HTTP) + custom WebSocket script

**Batch API Load Test:**
```bash
wrk -t4 -c100 -d30s --latency \
  -s scripts/batch_inference.lua \
  http://localhost:8000/api/inference/batch
```

**Expected Results:**
- Throughput: > 1000 req/sec
- Latency P99: < 5s
- Error rate: < 0.1%

**Streaming Load Test:**
```python
# tests/load/websocket_load.py
import asyncio
import websockets

async def simulate_client(client_id):
    async with websockets.connect("ws://localhost:3000/ws/stream") as ws:
        for i in range(100):  # 100s of audio
            await ws.send(audio_chunk)
            await ws.recv()

# Simulate 1000 concurrent clients
await asyncio.gather(*[simulate_client(i) for i in range(1000)])
```

---

## 7. SÉCURITÉ

### 7.1 Authentification JWT

**Flow:**
```
1. Client → POST /auth/login (username, password)
2. Server validates → returns JWT token
   {
     "access_token": "eyJhbGc...",
     "token_type": "bearer",
     "expires_in": 3600
   }
3. Client includes in subsequent requests:
   Authorization: Bearer eyJhbGc...
4. Server validates JWT signature + expiration
```

**JWT Payload:**
```json
{
  "sub": "user-uuid",
  "username": "john_doe",
  "exp": 1701350400,
  "iat": 1701346800,
  "scopes": ["inference:read", "models:write"]
}
```

### 7.2 Rate Limiting (Redis)

**Algorithm:** Token Bucket avec sliding window

**Implementation:**
```python
# Python (FastAPI middleware)
async def rate_limit_middleware(request: Request, call_next):
    user_id = get_user_from_jwt(request)
    key = f"rate_limit:{user_id}"
    
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, 60)  # 1 minute window
    
    if current > 100:  # 100 req/min
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return await call_next(request)
```

### 7.3 Input Validation

**Audio File Validation:**
```rust
// Rust: Validate audio format
fn validate_audio_input(data: &[u8]) -> Result<(), ValidationError> {
    // 1. Check magic bytes (WAV: RIFF, MP3: ID3)
    if !is_valid_audio_format(data) {
        return Err(ValidationError::InvalidFormat);
    }
    
    // 2. Check file size (max 100 MB)
    if data.len() > 100 * 1024 * 1024 {
        return Err(ValidationError::FileTooLarge);
    }
    
    // 3. Check duration (max 30 min)
    let duration = parse_audio_duration(data)?;
    if duration > 30 * 60 {
        return Err(ValidationError::DurationTooLong);
    }
    
    Ok(())
}
```

---

## 8. PERFORMANCE OPTIMIZATIONS

### 8.1 ONNX Model Optimization

**Quantization FP16:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model-fp32.onnx",
    model_output="model-fp16.onnx",
    weight_type=QuantType.QUInt8,
    optimize_model=True
)
```

**Graph Optimization:**
```rust
use ort::{GraphOptimizationLevel, Session};

let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file("model.onnx")?;
```

### 8.2 Connection Pooling

**PostgreSQL (Python):**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/voiceflow",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30
)
```

### 8.3 Caching Strategy

**Redis Cache (Rust):**
```rust
// Cache model metadata
async fn get_model_metadata(version: &str) -> Result<ModelMeta> {
    let cache_key = format!("model:meta:{}", version);
    
    // Try cache first
    if let Some(cached) = redis.get(&cache_key).await? {
        return Ok(serde_json::from_str(&cached)?);
    }
    
    // Fetch from Python service
    let meta = fetch_from_python_service(version).await?;
    
    // Cache for 1 hour
    redis.setex(&cache_key, 3600, serde_json::to_string(&meta)?).await?;
    
    Ok(meta)
}
```

---

## 9. DÉPLOIEMENT

### 9.1 Docker Multi-Stage Build (Rust)

```dockerfile
# Stage 1: Build
FROM rust:1.75 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

# Stage 2: Runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/voiceflow_inference /usr/local/bin/
COPY models/ /models/

EXPOSE 3000
CMD ["voiceflow_inference"]
```

### 9.2 Health Checks

**Liveness Probe:**
```yaml
# docker-compose.yml
services:
  inference-engine:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Readiness Probe (Rust):**
```rust
async fn health_check(State(app_state): State<AppState>) -> impl IntoResponse {
    // Check model loaded
    if !app_state.model_manager.is_ready() {
        return (StatusCode::SERVICE_UNAVAILABLE, "Model not loaded").into_response();
    }
    
    // Check dependencies (optional)
    // ...
    
    (StatusCode::OK, "healthy").into_response()
}
```

---

## 10. ANNEXES

### 10.1 Technologies Version Matrix

| Composant | Version Minimale | Version Recommandée |
|-----------|------------------|---------------------|
| Python | 3.10 | 3.11 |
| Rust | 1.70 | 1.75 |
| FastAPI | 0.100 | 0.104 |
| PyTorch | 2.0 | 2.1 |
| ONNX | 1.14 | 1.15 |
| Axum | 0.6 | 0.7 |
| Tokio | 1.30 | 1.35 |
| PostgreSQL | 14 | 15 |
| Redis | 7.0 | 7.2 |

---

**Version:** 1.0  
**Date:** 29 Novembre 2025  
**Statut:** ✅ Validé
