"""
Benchmark ONNX model inference speed.

Tests both unoptimized and optimized models to measure:
- Cold start latency
- Warm inference latency (P50, P95, P99)
- Throughput
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
from statistics import mean, median


def benchmark_model(model_path: Path, num_warmup: int = 10, num_iterations: int = 100):
    """Benchmark ONNX model inference speed."""
    
    print(f"\n{'=' * 60}")
    print(f"📊 Benchmarking: {model_path.name}")
    print(f"{'=' * 60}")
    
    # 1. Load model
    print(f"\n1️⃣ Loading model...")
    start = time.perf_counter()
    
    session = ort.InferenceSession(
        str(model_path),
        providers=['CPUExecutionProvider']
    )
    
    load_time = (time.perf_counter() - start) * 1000
    print(f"   └─ Load time: {load_time:.1f} ms")
    
    # 2. Create test input
    dummy_input = np.random.randn(1, 48000).astype(np.float32)  # 3 seconds audio
    
    # 3. Warmup runs
    print(f"\n2️⃣ Warming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        session.run(None, {'audio': dummy_input})
    print(f"   ✓ Warmup complete")
    
    # 4. Benchmark runs
    print(f"\n3️⃣ Benchmarking ({num_iterations} runs)...")
    latencies = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        outputs = session.run(None, {'audio': dummy_input})
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # 5. Calculate statistics
    latencies.sort()
    
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = mean(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    print(f"\n4️⃣ Results:")
    print(f"   ├─ Min:     {min_lat:.2f} ms")
    print(f"   ├─ Average: {avg:.2f} ms")
    print(f"   ├─ Median:  {p50:.2f} ms")
    print(f"   ├─ P95:     {p95:.2f} ms")
    print(f"   ├─ P99:     {p99:.2f} ms")
    print(f"   └─ Max:     {max_lat:.2f} ms")
    
    # 6. Performance assessment
    print(f"\n5️⃣ Performance Assessment:")
    
    if p99 < 100:
        print(f"   ✅ EXCELLENT: P99 latency < 100ms target!")
        print(f"      └─ Production ready for real-time inference")
    elif p99 < 200:
        print(f"   ⚠  GOOD: P99 latency < 200ms (acceptable)")
        print(f"      └─ Consider quantization for further speedup")
    elif p99 < 500:
        print(f"   ⚠  MODERATE: P99 latency < 500ms")
        print(f"      └─ Needs optimization (quantization, GPU, model pruning)")
    else:
        print(f"   ❌ SLOW: P99 latency > 500ms")
        print(f"      └─ Requires significant optimization")
    
    # 7. Throughput
    throughput = 1000 / avg  # Requests per second
    print(f"\n6️⃣ Throughput:")
    print(f"   └─ {throughput:.1f} requests/sec (single thread)")
    
    return {
        'load_time': load_time,
        'min': min_lat,
        'avg': avg,
        'median': p50,
        'p95': p95,
        'p99': p99,
        'max': max_lat,
        'throughput': throughput
    }


def compare_models():
    """Compare unoptimized vs optimized models."""
    
    unopt_path = Path("../models/diarization_transformer.onnx")
    opt_path = Path("../models/diarization_transformer_optimized.onnx")
    
    if not unopt_path.exists():
        print(f"❌ Unoptimized model not found: {unopt_path}")
        return
    
    if not opt_path.exists():
        print(f"❌ Optimized model not found: {opt_path}")
        return
    
    print("=" * 60)
    print("🚀 ONNX Model Performance Benchmark")
    print("=" * 60)
    print(f"\nTest configuration:")
    print(f"├─ Input: 3 seconds audio (48000 samples @ 16kHz)")
    print(f"├─ Hardware: CPU")
    print(f"├─ Provider: CPUExecutionProvider")
    print(f"└─ Iterations: 100")
    
    # Benchmark unoptimized
    unopt_results = benchmark_model(unopt_path)
    
    # Benchmark optimized
    opt_results = benchmark_model(opt_path)
    
    # Compare
    print(f"\n{'=' * 60}")
    print(f"📈 Comparison Summary")
    print(f"{'=' * 60}")
    
    print(f"\nLoad Time:")
    print(f"├─ Unoptimized: {unopt_results['load_time']:.1f} ms")
    print(f"└─ Optimized:   {opt_results['load_time']:.1f} ms")
    
    print(f"\nP99 Latency (target: <100ms):")
    print(f"├─ Unoptimized: {unopt_results['p99']:.2f} ms")
    print(f"└─ Optimized:   {opt_results['p99']:.2f} ms")
    
    speedup = unopt_results['p99'] / opt_results['p99']
    print(f"\nSpeedup: {speedup:.2f}x faster")
    
    print(f"\nThroughput:")
    print(f"├─ Unoptimized: {unopt_results['throughput']:.1f} req/sec")
    print(f"└─ Optimized:   {opt_results['throughput']:.1f} req/sec")
    
    # Recommendation
    print(f"\n{'=' * 60}")
    print(f"💡 Recommendation")
    print(f"{'=' * 60}")
    
    if opt_results['p99'] < 100:
        print(f"\n✅ Use optimized model for production!")
        print(f"   ├─ Meets <100ms P99 latency requirement")
        print(f"   ├─ Ready for real-time inference")
        print(f"   └─ File: {opt_path.name}")
    elif unopt_results['p99'] < 100 and opt_results['p99'] < 200:
        print(f"\n✅ Both models perform well!")
        print(f"   ├─ Optimized model: {opt_path.name}")
        print(f"   └─ Trade-off: {opt_results['load_time']:.0f}ms load time vs {opt_results['p99']:.0f}ms inference")
    else:
        print(f"\n⚠  Consider further optimizations:")
        print(f"   ├─ INT8 quantization (2-4x faster)")
        print(f"   ├─ FP16 on GPU (5-10x faster)")
        print(f"   ├─ Model distillation (smaller Wav2Vec2)")
        print(f"   └─ TensorRT/OpenVINO compilation")


if __name__ == "__main__":
    compare_models()
