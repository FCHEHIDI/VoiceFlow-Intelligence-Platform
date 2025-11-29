# 🔄 ARCHITECTURE FLOW - VoiceFlow Intelligence Platform

## 1. VUE D'ENSEMBLE DES FLUX

### 1.1 Types de Flux Principaux

```
┌────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                     │
└────────────┬──────────────────────────────┬────────────────┘
             │                              │
             │ BATCH PROCESSING             │ REAL-TIME STREAMING
             │ (HTTP REST)                  │ (WebSocket)
             ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │ Python ML       │            │ Rust Inference  │
    │ Service         │◄───────────┤ Engine          │
    │ (Port 8000)     │  Model     │ (Port 3000)     │
    └────────┬────────┘  Metadata  └────────┬────────┘
             │                              │
             ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │ PostgreSQL      │            │ Prometheus      │
    │ + Redis         │            │ + Grafana       │
    └─────────────────┘            └─────────────────┘
```

---

## 2. FLUX BATCH PROCESSING (Asynchrone)

### 2.1 Diagramme de Séquence Complet

```
Client          API Gateway      Python Service    Repository    Background Worker    Rust Service    PostgreSQL
  │                  │                 │                │                │                 │              │
  │ 1. POST /api/    │                 │                │                │                 │              │
  │    inference/    │                 │                │                │                 │              │
  │    batch         │                 │                │                │                 │              │
  ├─────────────────>│                 │                │                │                 │              │
  │                  │ 2. Validate     │                │                │                 │              │
  │                  │    JWT + Rate   │                │                │                 │              │
  │                  │    Limit        │                │                │                 │              │
  │                  ├────────────────>│                │                │                 │              │
  │                  │                 │ 3. Validate    │                │                 │              │
  │                  │                 │    Audio       │                │                 │              │
  │                  │                 │    (format,    │                │                 │              │
  │                  │                 │     size)      │                │                 │              │
  │                  │                 ├───────────────>│                │                 │              │
  │                  │                 │                │ 4. INSERT job  │                 │              │
  │                  │                 │                │    (status:    │                 │              │
  │                  │                 │                │    'pending')  │                 │              │
  │                  │                 │                ├───────────────>│                 │              │
  │                  │                 │                │<───────────────┤                 │              │
  │                  │                 │                │  job_id        │                 │              │
  │                  │                 │ 5. Enqueue job │                │                 │              │
  │                  │                 ├────────────────┼───────────────>│                 │              │
  │                  │                 │                │                │                 │              │
  │ <── 202 Accepted │<────────────────┤                │                │                 │              │
  │ {job_id: "..."}  │                 │                │                │                 │              │
  │                  │                 │                │                │                 │              │
  │                  │                 │                │ 6. Dequeue &   │                 │              │
  │                  │                 │                │    UPDATE      │                 │              │
  │                  │                 │                │    status:     │                 │              │
  │                  │                 │                │    'running'   │                 │              │
  │                  │                 │                ├───────────────>│                 │              │
  │                  │                 │                │                │<────────────────┤              │
  │                  │                 │                │                │                 │              │
  │                  │                 │                │                │ 7. Preprocess   │              │
  │                  │                 │                │                │    audio        │              │
  │                  │                 │                │                │    (MFCC)       │              │
  │                  │                 │                │                │                 │              │
  │                  │                 │                │                │ 8. POST /infer  │              │
  │                  │                 │                │                ├────────────────>│              │
  │                  │                 │                │                │                 │ 9. ONNX      │
  │                  │                 │                │                │                 │    Inference │
  │                  │                 │                │                │                 │              │
  │                  │                 │                │                │<────────────────┤              │
  │                  │                 │                │                │  segments       │              │
  │                  │                 │                │                │                 │              │
  │                  │                 │                │                │ 10. UPDATE job  │              │
  │                  │                 │                │                │     status:     │              │
  │                  │                 │                │                │     'completed' │              │
  │                  │                 │                │                │     + results   │              │
  │                  │                 │                ├────────────────┤                 │              │
  │                  │                 │                │<───────────────┤                 │              │
  │                  │                 │                │                │                 │              │
  │                  │                 │                │                │ 11. Optional:   │              │
  │                  │                 │                │                │     POST        │              │
  │ <── Webhook ─────┼─────────────────┼────────────────┼────────────────┤  callback_url   │              │
  │ (if configured)  │                 │                │                │                 │              │
  │                  │                 │                │                │                 │              │
  │ 12. GET /api/    │                 │                │                │                 │              │
  │     jobs/{id}    │                 │                │                │                 │              │
  ├─────────────────>│                 │                │                │                 │              │
  │                  ├────────────────>│                │                │                 │              │
  │                  │                 ├───────────────>│ SELECT job     │                 │              │
  │                  │                 │                ├───────────────>│                 │              │
  │                  │                 │                │<───────────────┤                 │              │
  │                  │                 │<───────────────┤  job + results │                 │              │
  │ <── 200 OK ──────┤<────────────────┤                │                │                 │              │
  │ {status:         │                 │                │                │                 │              │
  │  "completed",    │                 │                │                │                 │              │
  │  results: {...}} │                 │                │                │                 │              │
```

### 2.2 Étapes Détaillées

#### Étape 1-2: Authentification & Rate Limiting
```python
# FastAPI Middleware
@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    # Validate JWT
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = verify_jwt(token)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    # Check rate limit (Redis)
    key = f"rate_limit:{user.id}"
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, 60)
    if current > 100:
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
    
    # Continue
    response = await call_next(request)
    return response
```

#### Étape 3: Validation Audio
```python
def validate_audio_file(file: UploadFile) -> None:
    # 1. Check file size
    if file.size > 100 * 1024 * 1024:  # 100 MB
        raise HTTPException(status_code=413, detail="File too large")
    
    # 2. Check magic bytes
    header = file.file.read(12)
    file.file.seek(0)
    if not (header.startswith(b'RIFF') or header.startswith(b'ID3')):
        raise HTTPException(status_code=400, detail="Invalid audio format")
    
    # 3. Check duration (parse WAV header)
    duration = parse_audio_duration(file.file)
    if duration > 30 * 60:  # 30 minutes
        raise HTTPException(status_code=400, detail="Audio too long (max 30 min)")
```

#### Étape 4-5: Job Creation & Enqueue
```python
# Service Layer
async def create_batch_job(
    audio_file: UploadFile,
    user_id: str,
    callback_url: Optional[str] = None
) -> str:
    # Save audio to temp storage
    audio_path = await save_to_temp_storage(audio_file)
    
    # Create job record
    job = await job_repository.create({
        "status": "pending",
        "audio_path": audio_path,
        "user_id": user_id,
        "callback_url": callback_url,
        "created_at": datetime.utcnow()
    })
    
    # Enqueue for background processing
    await task_queue.enqueue("process_audio", job.id)
    
    return job.id
```

#### Étape 6-10: Background Worker Processing
```python
# Background Worker (Celery/RQ)
async def process_audio(job_id: str):
    try:
        # Update status
        await job_repository.update(job_id, {"status": "running"})
        
        # Load audio
        job = await job_repository.get(job_id)
        audio_data = load_audio(job.audio_path)
        
        # Preprocess
        features = extract_features(audio_data)  # MFCC, mel-spectrogram
        
        # Call Rust inference service
        response = await httpx.post(
            "http://rust-service:3000/infer",
            json={
                "audio": features.tolist(),
                "sample_rate": 16000
            },
            timeout=30.0
        )
        results = response.json()
        
        # Update job with results
        await job_repository.update(job_id, {
            "status": "completed",
            "results": results,
            "completed_at": datetime.utcnow(),
            "processing_time_ms": results["latency_ms"]
        })
        
        # Optional: Trigger webhook
        if job.callback_url:
            await httpx.post(job.callback_url, json=results)
        
        # Cleanup temp file
        os.remove(job.audio_path)
        
    except Exception as e:
        await job_repository.update(job_id, {
            "status": "failed",
            "error_message": str(e)
        })
        logger.error(f"Job {job_id} failed: {e}")
```

---

## 3. FLUX STREAMING TEMPS-RÉEL (WebSocket)

### 3.1 Diagramme de Séquence

```
Client                    Rust WebSocket Handler              ModelManager           ONNX Runtime
  │                              │                                 │                       │
  │ 1. WS Connect                │                                 │                       │
  │   ws://host:3000/ws/stream   │                                 │                       │
  ├─────────────────────────────>│                                 │                       │
  │                              │ 2. Upgrade HTTP → WebSocket     │                       │
  │                              │    + Create AudioBuffer         │                       │
  │                              │                                 │                       │
  │ <─────── 101 Switching ──────┤                                 │                       │
  │          Protocols           │                                 │                       │
  │                              │                                 │                       │
  │ 3. Send audio chunk #1       │                                 │                       │
  │    (1s @ 16kHz = 16000       │                                 │                       │
  │     samples)                 │                                 │                       │
  ├─────────────────────────────>│                                 │                       │
  │                              │ 4. Append to buffer             │                       │
  │                              │    + Extract features (MFCC)    │                       │
  │                              │                                 │                       │
  │                              │ 5. Get production model         │                       │
  │                              ├────────────────────────────────>│                       │
  │                              │<────────────────────────────────┤                       │
  │                              │  Arc<Session>                   │                       │
  │                              │                                 │                       │
  │                              │ 6. run_inference(features)      │                       │
  │                              │─────────────────────────────────┼──────────────────────>│
  │                              │                                 │  ONNX forward pass    │
  │                              │                                 │  (optimized FP16)     │
  │                              │<────────────────────────────────┼───────────────────────┤
  │                              │  speaker_probs: [0.92, 0.08]    │                       │
  │                              │                                 │                       │
  │                              │ 7. Post-process                 │                       │
  │                              │    - Argmax → speaker_id        │                       │
  │                              │    - Format JSON result         │                       │
  │                              │    - Measure latency            │                       │
  │                              │                                 │                       │
  │ <─────── WS Send ────────────┤                                 │                       │
  │ {                            │                                 │                       │
  │   "type": "diarization_      │                                 │                       │
  │            result",          │                                 │                       │
  │   "segments": [...],         │                                 │                       │
  │   "latency_ms": 78,          │                                 │                       │
  │   "sequence": 1              │                                 │                       │
  │ }                            │                                 │                       │
  │                              │                                 │                       │
  │ 4. Send chunk #2             │                                 │                       │
  ├─────────────────────────────>│                                 │                       │
  │ ...                          │ [Repeat steps 4-7]              │                       │
  │ ...                          │                                 │                       │
  │                              │                                 │                       │
  │ 5. Send end_stream           │                                 │                       │
  ├─────────────────────────────>│                                 │                       │
  │                              │ 6. Flush buffer                 │                       │
  │                              │    + Cleanup                    │                       │
  │                              │                                 │                       │
  │ <─── WS Close (1000) ────────┤                                 │                       │
  │                              │                                 │                       │
```

### 3.2 Implémentation WebSocket Handler (Rust)

```rust
use axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::Response,
};
use futures::{SinkExt, StreamExt};

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(app_state): State<AppState>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, app_state))
}

async fn handle_socket(socket: WebSocket, app_state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut audio_buffer = AudioBuffer::new(16000); // 1s @ 16kHz
    
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                // Parse JSON message
                let request: StreamRequest = serde_json::from_str(&text)?;
                
                match request.message_type.as_str() {
                    "audio_chunk" => {
                        let start = Instant::now();
                        
                        // 1. Append to buffer
                        audio_buffer.push(&request.data);
                        
                        // 2. Extract features
                        let features = extract_mfcc(&audio_buffer.get_chunk());
                        
                        // 3. Run inference
                        let model = app_state.model_manager.get_production_model().await?;
                        let probs = model.run_inference(&features)?;
                        
                        // 4. Post-process
                        let speaker_id = argmax(&probs);
                        let confidence = probs[speaker_id];
                        
                        let latency_ms = start.elapsed().as_millis() as u32;
                        
                        // 5. Send result back
                        let response = StreamResponse {
                            message_type: "diarization_result".to_string(),
                            segments: vec![Segment {
                                start: 0.0,
                                end: 1.0,
                                speaker_id: format!("SPEAKER_{:02}", speaker_id),
                                confidence,
                            }],
                            latency_ms,
                            sequence: request.sequence,
                        };
                        
                        sender.send(Message::Text(
                            serde_json::to_string(&response)?
                        )).await?;
                        
                        // 6. Log metrics
                        INFERENCE_LATENCY.observe(latency_ms as f64 / 1000.0);
                        INFERENCE_REQUESTS_TOTAL.inc();
                    }
                    
                    "end_stream" => {
                        // Flush buffer and close
                        break;
                    }
                    
                    _ => {
                        eprintln!("Unknown message type: {}", request.message_type);
                    }
                }
            }
            
            Message::Close(_) => {
                break;
            }
            
            _ => {}
        }
    }
    
    // Cleanup
    drop(audio_buffer);
}
```

### 3.3 Optimisations Performance

**1. Buffer Circulaire:**
```rust
struct AudioBuffer {
    buffer: VecDeque<f32>,
    capacity: usize,
}

impl AudioBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front(); // Remove oldest
            }
            self.buffer.push_back(sample);
        }
    }
    
    pub fn get_chunk(&self) -> Vec<f32> {
        self.buffer.iter().copied().collect()
    }
}
```

**2. Batch Inference (accumuler 5 chunks):**
```rust
// Au lieu de 1 inférence par chunk (latency ~80ms)
// → 5 inférences batched (latency ~120ms mais throughput x5)

if audio_buffer.len() >= 5 * 16000 {
    let batch_features = extract_features_batch(&audio_buffer.get_all());
    let batch_results = model.run_inference_batch(&batch_features)?;
    // Send 5 results at once
}
```

---

## 4. FLUX GESTION DES MODÈLES

### 4.1 Training → Export → Deployment

```
Data Scientist          Python Service        FileSystem         Rust Service      PostgreSQL
     │                       │                    │                   │                │
     │ 1. POST /api/models/  │                    │                   │                │
     │    train              │                    │                   │                │
     ├──────────────────────>│                    │                   │                │
     │                       │ 2. INSERT job      │                   │                │
     │                       │    (status:        │                   │                │
     │                       │    'running')      │                   │                │
     │                       ├───────────────────────────────────────>│                │
     │                       │                    │                   │<───────────────┤
     │                       │                    │                   │                │
     │                       │ 3. Training loop   │                   │                │
     │                       │    (PyTorch)       │                   │                │
     │                       │    - Load dataset  │                   │                │
     │                       │    - Forward/back  │                   │                │
     │                       │    - Optimize      │                   │                │
     │                       │    [50 epochs]     │                   │                │
     │                       │                    │                   │                │
     │                       │ 4. Save checkpoint │                   │                │
     │                       ├───────────────────>│                   │                │
     │                       │  model_v1.3.0.pth  │                   │                │
     │                       │                    │                   │                │
     │ <── 200 OK ───────────┤                    │                   │                │
     │ {model_id: "..."}     │                    │                   │                │
     │                       │                    │                   │                │
     │ 2. POST /api/models/  │                    │                   │                │
     │    {id}/export        │                    │                   │                │
     ├──────────────────────>│                    │                   │                │
     │                       │ 5. Load PyTorch    │                   │                │
     │                       │    checkpoint      │                   │                │
     │                       │<───────────────────┤                   │                │
     │                       │                    │                   │                │
     │                       │ 6. Export to ONNX  │                   │                │
     │                       │   torch.onnx.      │                   │                │
     │                       │   export(...)      │                   │                │
     │                       │                    │                   │                │
     │                       │ 7. Optimize ONNX   │                   │                │
     │                       │   - Graph opt L3   │                   │                │
     │                       │   - Quantize FP16  │                   │                │
     │                       │                    │                   │                │
     │                       │ 8. Save ONNX       │                   │                │
     │                       ├───────────────────>│                   │                │
     │                       │  model_v1.3.0.onnx │                   │                │
     │                       │                    │                   │                │
     │                       │ 9. UPDATE models   │                   │                │
     │                       │    SET onnx_path   │                   │                │
     │                       ├───────────────────────────────────────>│                │
     │                       │                    │                   │                │
     │ <── 200 OK ───────────┤                    │                   │                │
     │ {onnx_path: "..."}    │                    │                   │                │
     │                       │                    │                   │                │
     │ 3. PUT /api/models/   │                    │                   │                │
     │    {id}/activate      │                    │                   │                │
     ├──────────────────────>│                    │                   │                │
     │                       │ 10. UPDATE models  │                   │                │
     │                       │     SET            │                   │                │
     │                       │     is_production  │                   │                │
     │                       │     = TRUE         │                   │                │
     │                       ├───────────────────────────────────────>│                │
     │                       │                    │                   │                │
     │                       │ 11. Notify Rust    │                   │                │
     │                       │     (Webhook ou    │                   │                │
     │                       │      poll)         │                   │                │
     │                       ├───────────────────────────────────────>│                │
     │                       │                    │                   │ 12. Hot-reload │
     │                       │                    │                   │     model      │
     │                       │                    │<──────────────────┤                │
     │                       │                    │  Load new ONNX    │                │
     │                       │                    │                   │                │
     │                       │                    │                   │ 13. UPDATE     │
     │                       │                    │                   │     AppState   │
     │                       │                    │                   │     (Arc<      │
     │                       │                    │                   │     Session>)  │
     │                       │                    │                   │                │
     │ <── 200 OK ───────────┤                    │                   │                │
     │ {status: "active"}    │                    │                   │                │
```

### 4.2 Hot-Reload Implémentation (Rust)

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, Arc<Session>>>>,
    production_version: Arc<RwLock<String>>,
}

impl ModelManager {
    pub async fn reload_model(&self, version: &str, path: &str) -> Result<()> {
        // 1. Load new model
        let new_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;
        
        // 2. Update models registry (atomic)
        let mut models = self.models.write().await;
        models.insert(version.to_string(), Arc::new(new_session));
        
        // 3. Update production version (atomic)
        let mut prod_version = self.production_version.write().await;
        *prod_version = version.to_string();
        
        info!("Model {} hot-reloaded successfully", version);
        Ok(())
    }
    
    pub async fn get_production_model(&self) -> Result<Arc<Session>> {
        let version = self.production_version.read().await;
        let models = self.models.read().await;
        
        models.get(version.as_str())
            .cloned()
            .ok_or_else(|| anyhow!("Production model not found"))
    }
}
```

---

## 5. FLUX A/B TESTING

### 5.1 Traffic Routing Logic

```
Client Request        Load Balancer         Model Router         Model A (90%)      Model B (10%)
     │                      │                    │                     │                  │
     │ 1. POST /infer       │                    │                     │                  │
     ├─────────────────────>│                    │                     │                  │
     │  (user_id: "abc123") │                    │                     │                  │
     │                      │ 2. Hash user_id    │                     │                  │
     │                      │    hash("abc123")  │                     │                  │
     │                      │    % 100 = 42      │                     │                  │
     │                      │                    │                     │                  │
     │                      │ 3. Get routing     │                     │                  │
     │                      │    config          │                     │                  │
     │                      ├───────────────────>│                     │                  │
     │                      │                    │ SELECT * FROM       │                  │
     │                      │                    │ model_routing       │                  │
     │                      │<───────────────────┤ WHERE is_active     │                  │
     │                      │ [                  │                     │                  │
     │                      │   {v1.2.3: 90%},   │                     │                  │
     │                      │   {v1.3.0: 10%}    │                     │                  │
     │                      │ ]                  │                     │                  │
     │                      │                    │                     │                  │
     │                      │ 4. Route decision  │                     │                  │
     │                      │    42 < 90         │                     │                  │
     │                      │    → Use Model A   │                     │                  │
     │                      │                    │                     │                  │
     │                      │ 5. Forward to A    │                     │                  │
     │                      ├─────────────────────────────────────────>│                  │
     │                      │                    │                     │ 6. Inference     │
     │                      │                    │                     │                  │
     │                      │<─────────────────────────────────────────┤                  │
     │                      │  result + metadata │                     │                  │
     │                      │  (model_version:   │                     │                  │
     │                      │   "1.2.3")         │                     │                  │
     │                      │                    │                     │                  │
     │ <────────────────────┤                    │                     │                  │
     │ result + version     │                    │                     │                  │
     │                      │                    │                     │                  │
     │                      │ 7. Log to          │                     │                  │
     │                      │    Prometheus      │                     │                  │
     │                      │   (labeled by      │                     │                  │
     │                      │    model_version)  │                     │                  │
```

### 5.2 Implémentation Router

```rust
pub struct ModelRouter {
    routing_config: Arc<RwLock<Vec<RoutingRule>>>,
}

impl ModelRouter {
    pub async fn select_model(&self, user_id: &str) -> String {
        // 1. Hash user_id for consistent routing
        let mut hasher = DefaultHasher::new();
        hasher.write(user_id.as_bytes());
        let hash = hasher.finish();
        let bucket = (hash % 100) as u8; // 0-99
        
        // 2. Get routing config
        let config = self.routing_config.read().await;
        
        // 3. Select model based on traffic percentage
        let mut cumulative = 0;
        for rule in config.iter() {
            cumulative += rule.traffic_percent;
            if bucket < cumulative {
                return rule.model_version.clone();
            }
        }
        
        // Fallback to first model
        config[0].model_version.clone()
    }
}

// Usage in handler
let model_version = app_state.router.select_model(&user_id).await;
let model = app_state.model_manager.get_model(&model_version).await?;
let result = model.run_inference(&features)?;

// Log with label
INFERENCE_REQUESTS_TOTAL
    .with_label_values(&[&model_version])
    .inc();
```

---

## 6. FLUX DE MONITORING

### 6.1 Metrics Collection Flow

```
Rust Service          Prometheus Exporter       Prometheus Server      Grafana
     │                       │                         │                  │
     │ 1. Inference          │                         │                  │
     │    completed          │                         │                  │
     │    latency: 78ms      │                         │                  │
     │                       │                         │                  │
     │ 2. Record metric      │                         │                  │
     │   INFERENCE_LATENCY   │                         │                  │
     │   .observe(0.078)     │                         │                  │
     │                       │                         │                  │
     │                       │ 3. Expose /metrics      │                  │
     │                       │<────────────────────────┤                  │
     │                       │                         │ GET /metrics     │
     │                       │                         │ (scrape every    │
     │                       │                         │  15s)            │
     │                       │                         │                  │
     │                       ├────────────────────────>│                  │
     │                       │ # HELP inference_...    │                  │
     │                       │ # TYPE inference_...    │                  │
     │                       │ inference_latency_      │                  │
     │                       │ seconds_bucket{le=      │                  │
     │                       │ "0.1"} 1245             │                  │
     │                       │ ...                     │                  │
     │                       │                         │                  │
     │                       │                         │ 4. Store TSDB    │
     │                       │                         │                  │
     │                       │                         │                  │
     │                       │                         │<─────────────────┤
     │                       │                         │ 5. PromQL query  │
     │                       │                         │   histogram_     │
     │                       │                         │   quantile(0.99, │
     │                       │                         │   inference_     │
     │                       │                         │   latency_       │
     │                       │                         │   seconds)       │
     │                       │                         │                  │
     │                       │                         ├─────────────────>│
     │                       │                         │ Result: 0.095s   │
     │                       │                         │                  │
     │                       │                         │                  │ 6. Display
     │                       │                         │                  │    Dashboard
     │                       │                         │                  │    Graph
```

### 6.2 Exemple Dashboard Grafana (JSON)

```json
{
  "dashboard": {
    "title": "Real-Time Inference Metrics",
    "panels": [
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99"
          }
        ],
        "yaxes": [
          {"format": "s", "label": "Latency"}
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {"params": [0.1], "type": "gt"},
              "query": {"params": ["A", "5m", "now"]},
              "reducer": {"type": "avg"}
            }
          ],
          "message": "P99 latency > 100ms"
        }
      },
      {
        "title": "Throughput by Model Version",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(inference_requests_total[1m])) by (model_version)",
            "legendFormat": "{{model_version}}"
          }
        ]
      }
    ]
  }
}
```

---

## 7. FLUX D'ERREURS ET RECOVERY

### 7.1 Circuit Breaker Pattern

```
Python Service      Circuit Breaker       Rust Service
     │                    │                    │
     │ 1. Call Rust       │                    │
     ├───────────────────>│                    │
     │                    │ 2. Forward         │
     │                    ├───────────────────>│
     │                    │                    │ ERROR (timeout)
     │                    │<───────────────────┤
     │                    │ 3. Record failure  │
     │<───────────────────┤    (count: 1)      │
     │ Error              │                    │
     │                    │                    │
     │ ... (5 failures)   │                    │
     │                    │ State: OPEN        │
     │                    │ (reject requests)  │
     │                    │                    │
     │ 4. New call        │                    │
     ├───────────────────>│                    │
     │                    │ 5. Reject          │
     │<───────────────────┤    immediately     │
     │ Error (circuit     │    (no network     │
     │  open)             │     call)          │
     │                    │                    │
     │ ... (30s timeout)  │                    │
     │                    │ State: HALF_OPEN   │
     │                    │                    │
     │ 6. Test call       │                    │
     ├───────────────────>│                    │
     │                    ├───────────────────>│
     │                    │<───────────────────┤
     │<───────────────────┤ Success!           │
     │ OK                 │                    │
     │                    │ State: CLOSED      │
     │                    │ (normal operation) │
```

### 7.2 Retry with Exponential Backoff

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def call_rust_service(data):
    try:
        response = await httpx.post(
            "http://rust-service:3000/infer",
            json=data,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        logger.warning("Rust service timeout, retrying...")
        raise
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            logger.warning(f"Rust service error {e.response.status_code}, retrying...")
            raise
        else:
            # Don't retry client errors (4xx)
            logger.error(f"Client error: {e}")
            raise

# Retry schedule:
# Attempt 1: immediate
# Attempt 2: wait 2s
# Attempt 3: wait 4s
# Fail after 3 attempts
```

---

## 8. RÉSUMÉ DES FLUX CRITIQUES

### 8.1 Latency Budget

| Étape | Budget (ms) | Notes |
|-------|-------------|-------|
| WebSocket receive | 5 | Network + deserialization |
| Feature extraction | 10 | MFCC computation |
| ONNX inference | 70 | Model forward pass (FP16) |
| Post-processing | 5 | Argmax + formatting |
| WebSocket send | 10 | Serialization + network |
| **TOTAL** | **100** | **P99 target** |

### 8.2 Bottleneck Identification

**Si latence > 100ms:**
1. **Profile ONNX inference:**
   ```rust
   let start = Instant::now();
   let output = session.run(inputs)?;
   let duration = start.elapsed();
   if duration.as_millis() > 70 {
       warn!("Slow inference: {}ms", duration.as_millis());
   }
   ```

2. **Vérifier quantization:**
   - FP32 → FP16: gain ~40%
   - FP16 → INT8: gain ~60% (mais accuracy loss)

3. **Optimiser feature extraction:**
   - Utiliser SIMD (Rust: `std::simd`)
   - Pre-compute constants

---

**Version:** 1.0  
**Date:** 29 Novembre 2025  
**Statut:** ✅ Validé
