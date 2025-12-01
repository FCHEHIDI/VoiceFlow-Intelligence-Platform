# VoiceFlow Test Suite

Comprehensive testing suite for VoiceFlow Intelligence Platform, covering unit, integration, and hardware performance tests.

## Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
└── hardware/       # Hardware-specific performance tests
```

## Hardware Performance Tests

### CPU Benchmark Test

Validates model inference performance on CPU against SLA targets.

```bash
# Run with default model
python tests/hardware/test_cpu_benchmark.py

# Run with custom model
python tests/hardware/test_cpu_benchmark.py \
    --model models/diarization_fast_cnn_optimized.onnx \
    --iterations 200 \
    --sla-p99 10.0

# Expected output:
# ✅ PASS: P99 (4.48ms) < 10.0ms (Model Inference SLA)
```

**SLA Target**: P99 < 10ms (model inference only)  
**Validated Performance**: 4.48ms P99 on Intel/AMD 4-core CPU

### End-to-End Latency Test

Validates complete latency chain including network, Rust processing, and model inference.

```bash
# Run with default configuration
python tests/hardware/test_end_to_end_latency.py

# Simulate different network conditions
python tests/hardware/test_end_to_end_latency.py \
    --network-latency-range 10,40 \
    --rust-overhead-range 5,8 \
    --sla-p99 100.0

# Test worst-case scenario (slow network)
python tests/hardware/test_end_to_end_latency.py \
    --network-latency-range 30,50 \
    --rust-overhead-range 8,12

# Expected output:
# ✅ PASS: End-to-end P99 (65.23ms) < 100.0ms (Production SLA)
```

**Latency Breakdown**:
- Network (round-trip): 10-40ms typical
- Rust processing: 5-8ms
- Model inference: 4.48ms P99 (CPU), 2-4ms (GPU T4)
- **Total**: 40-80ms P99 typical

**SLA Target**: P99 < 100ms (end-to-end)

### GPU Inference Test

Tests GPU-accelerated inference on NVIDIA T4, RTX, or A100 GPUs.

```bash
# Basic GPU test
python tests/hardware/test_gpu_inference.py

# Compare CPU vs GPU
python tests/hardware/test_gpu_inference.py --compare-cpu

# Test specific providers
python tests/hardware/test_gpu_inference.py --providers cuda,tensorrt
```

**Expected Performance (T4 GPU)**:
- P99 latency: 2-4ms (model inference only)
- Throughput: 500-800 req/s
- GPU utilization: 80-95%

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/performance-tests.yml
name: Performance Tests

on: [push, pull_request]

jobs:
  cpu-benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install onnxruntime numpy
      - name: Run CPU benchmark
        run: |
          python tests/hardware/test_cpu_benchmark.py \
            --model models/diarization_fast_cnn_optimized.onnx \
            --iterations 100 \
            --sla-p99 10.0
      - name: Run end-to-end test
        run: |
          python tests/hardware/test_end_to_end_latency.py \
            --iterations 100 \
            --sla-p99 100.0
```

### Local Testing Script

```bash
#!/bin/bash
# Run all performance tests

echo "🧪 Running VoiceFlow Performance Test Suite"

# CPU benchmark
echo -e "\n1️⃣ CPU Benchmark Test"
python tests/hardware/test_cpu_benchmark.py || exit 1

# End-to-end latency
echo -e "\n2️⃣ End-to-End Latency Test"
python tests/hardware/test_end_to_end_latency.py || exit 1

# GPU test (if available)
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n3️⃣ GPU Inference Test"
    python tests/hardware/test_gpu_inference.py || exit 1
fi

echo -e "\n✅ All tests passed!"
```

## Performance SLA Summary

| Test | Component | SLA Target | Validated Result | Status |
|------|-----------|------------|------------------|--------|
| CPU Benchmark | Model Inference | P99 < 10ms | 4.48ms | ✅ |
| End-to-End | Full Chain | P99 < 100ms | 40-80ms | ✅ |
| GPU T4 | Model Inference | P99 < 10ms | 2-4ms (projected) | ⏳ |
| Throughput | Requests/sec | > 100 req/s | 297 req/s (CPU) | ✅ |

## Troubleshooting

### Test Fails with "Model not found"

```bash
# Ensure you're running from project root
cd /path/to/VoiceFlow-Intelligence-Platform

# Or specify absolute path
python tests/hardware/test_cpu_benchmark.py \
    --model /absolute/path/to/model.onnx
```

### Inconsistent Latency Results

1. **Close background applications** (browsers, IDEs, etc.)
2. **Increase warmup iterations**: `--warmup 50`
3. **Increase test iterations**: `--iterations 500`
4. **Pin CPU affinity** (Linux):
   ```bash
   taskset -c 0-3 python tests/hardware/test_cpu_benchmark.py
   ```

### GPU Test Fails

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include: ['CUDAExecutionProvider', ...]

# Reinstall onnxruntime-gpu
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```

## Next Steps

1. ✅ Validate CPU performance (4.48ms P99 achieved)
2. ✅ Test end-to-end latency simulation
3. ⏳ Deploy to GCP T4 and run GPU benchmarks
4. ⏳ Load testing with concurrent requests (Apache Bench, Locust)
5. ⏳ Production monitoring integration (Prometheus + Grafana)

See [docs/PERFORMANCE_ANALYSIS.md](../docs/PERFORMANCE_ANALYSIS.md) for detailed performance specifications.
