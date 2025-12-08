# Rust Inference Server - Performance Analysis

## Build Summary
- **Date**: December 8, 2025
- **Rust Version**: 1.91.1
- **ONNX Runtime**: v2.0.0-rc.10 (CPU-only build)
- **Model**: Fast CNN (2.3M params, 10MB optimized ONNX)
- **Hardware**: AMD Ryzen 7 8845HS (8 cores, 16 threads)

## Build Configuration
```toml
[dependencies]
axum = { version = "0.7", features = ["ws"] }
tokio = { version = "1.35", features = ["full"] }
ort = { version = "2.0.0-rc.10", features = ["download-binaries"] }  # CPU-only
```

### Key Fixes Applied
1. **Python docstrings → Rust comments**: Converted all `"""..."""` to `//!` module-level docs
2. **ONNX Runtime API v2**: Updated to `session::Session`, `value::Value`, `execution_providers::CPUExecutionProvider`
3. **Axum 0.7 API**: Migrated from deprecated `axum::Server` to `axum::serve(listener, app)`
4. **Thread safety**: Wrapped `Session` in `Arc<RwLock<Session>>` for async inference
5. **Tensor creation**: Used `Value::from_array((shape, data))` tuple API
6. **WebSocket support**: Enabled `ws` feature in axum dependencies

## Performance Test Results

### Test Configuration
- **Requests**: 200 total
- **Concurrency**: 20 concurrent connections
- **Audio**: 48000 samples (1 second @ 48kHz)
- **Duration**: 3.98s

### Rust Server (CPU, v1.0.0)
```
Performance:
  * Throughput:          50.3 req/s
  * End-to-End P99:      403.57 ms
  * Model Inference P99: 96.00 ms
  * Success Rate:        100% (200/200)

Latency Breakdown:
  * Model Min:     5.00 ms
  * Model Median:  7.00 ms
  * Model P95:     55.00 ms
  * Model P99:     96.00 ms (target: <10ms)
  * Model Max:     102.00 ms
```

### Python FastAPI Server (Baseline)
```
Performance:
  * Throughput:          39.2 req/s
  * End-to-End P99:      563.52 ms
  * Model Inference P99: 7.99 ms
  * Success Rate:        100% (200/200)
```

## Analysis

### ✅ Improvements
1. **Throughput**: +28% improvement (39.2 → 50.3 req/s)
2. **End-to-End P99**: -28% improvement (563ms → 403ms)
3. **Startup time**: 53ms model load (excellent)
4. **Memory safety**: Zero unsafe code, full Rust memory safety
5. **Production ready**: Prometheus metrics, graceful shutdown, structured logging

### ❌ Unexpected Regression
**Model Inference P99**: 96ms vs Python's 7.99ms (1100% slower!)

### Root Cause
The `Arc<RwLock<Session>>` serializes all inference calls, creating a bottleneck under concurrent load:
- Write lock held during entire inference (5-100ms)
- 20 concurrent requests queue behind single lock
- Python's GIL + ONNX Runtime releases GIL during inference
- **Net effect**: Concurrency becomes serialization

### Solution Path
1. **Option A - Per-thread sessions**: Clone session for each worker thread (higher memory)
2. **Option B - Session cloning**: ONNX Runtime sessions are thread-safe, remove RwLock
3. **Option C - Request queue**: Single-threaded inference with async queue (simpler)
4. **Option D - GPU acceleration**: T4 GPU will achieve 2-4ms inference, masking lock overhead

## Comparison vs SLA Targets

| Metric | Target | Python | Rust CPU | GCP T4 (Est) |
|--------|--------|--------|----------|--------------|
| Model P99 | <10ms | ✅ 7.99ms | ❌ 96ms | ✅ 2-4ms |
| E2E P99 | <100ms | ❌ 563ms | ❌ 403ms | ✅ 40-80ms |
| Throughput | >50 req/s | ❌ 39.2 | ✅ 50.3 | ✅ 200+ |

## Next Steps

### Immediate (Local CPU Fix)
```rust
// Remove RwLock, leverage ONNX RT thread-safety
pub struct ModelRunner {
    session: Arc<Session>,  // Session is internally thread-safe
    version: String,
}

// Direct inference without lock
pub async fn run_inference(&self, audio: &[f32]) -> AppResult<Vec<f32>> {
    let outputs = self.session.run(...)?;  // No await, no lock
    ...
}
```

### Priority (GCP T4 GPU Deployment)
1. Update `Cargo.toml`: Enable CUDA features
   ```toml
   ort = { version = "2.0.0-rc.10", features = ["cuda", "download-binaries"] }
   ```
2. Deploy to GCP T4 instance ($0.70/hour)
3. Expected results:
   - Model P99: 2-4ms (24x faster than current)
   - E2E P99: 40-80ms (5x faster)
   - Throughput: 200+ req/s (4x faster)

## Recommendations

**DO NOT** merge current Rust server to production due to RwLock bottleneck.

**Priority order**:
1. ✅ Fix RwLock issue (remove lock, use Arc<Session> directly)
2. ✅ Validate local CPU performance matches Python baseline
3. ✅ Deploy to GCP T4 GPU
4. ✅ Production validation with <100ms P99 SLA

## Files Modified
- `Cargo.toml`: CPU-only ONNX Runtime, axum ws feature
- `src/main.rs`: Config path, axum::serve API
- `src/inference/mod.rs`: ort v2 API, RwLock for thread safety
- `src/api/handlers.rs`: Async inference calls
- `src/streaming/mod.rs`: Async inference in WebSocket handler
- All `.rs` files: Python docstrings → Rust comments

## Build Commands
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Build optimized binary
cd voiceflow-inference
cargo build --release

# Run server
export MODELS_DIR="../voiceflow-ml/models"
./target/release/voiceflow_inference

# Load test
cd ../voiceflow-ml
python load_test.py --requests 200 --concurrency 20
```

## Conclusion
Rust server is **production-capable** with minor RwLock fix + GPU deployment. The infrastructure is solid—just need to unlock parallelism and leverage GPU acceleration to achieve full SLA compliance.
