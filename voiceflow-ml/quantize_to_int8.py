"""
Quantize ONNX model to INT8 for 2-4x speedup.

Dynamic quantization converts FP32 weights to INT8 while keeping
activations in FP32. This reduces model size and speeds up inference
with minimal accuracy loss (~1-2%).
"""

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import numpy as np
import time


def quantize_model(input_path: Path, output_path: Path):
    """Quantize ONNX model to INT8."""
    
    print("=" * 60)
    print("🔧 Quantizing Model to INT8")
    print("=" * 60)
    
    print(f"\n1️⃣ Input model:")
    print(f"   └─ {input_path.name}")
    
    orig_size = input_path.stat().st_size / (1024 * 1024)
    print(f"   └─ Size: {orig_size:.1f} MB")
    
    # 2. Quantize
    print(f"\n2️⃣ Quantizing...")
    print(f"   ├─ Method: Dynamic quantization")
    print(f"   ├─ Precision: INT8 (weights)")
    print(f"   └─ This will take 1-2 minutes...")
    
    start = time.time()
    
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,  # Quantize weights to INT8
        per_channel=True,  # Per-channel quantization (better accuracy)
    )
    
    duration = time.time() - start
    
    print(f"   ✓ Quantization complete in {duration:.1f}s")
    
    # 3. Compare sizes
    if output_path.exists():
        quant_size = output_path.stat().st_size / (1024 * 1024)
        reduction = ((orig_size - quant_size) / orig_size) * 100
        
        print(f"\n3️⃣ Results:")
        print(f"   ├─ Original:   {orig_size:.1f} MB")
        print(f"   ├─ Quantized:  {quant_size:.1f} MB")
        print(f"   └─ Reduction:  {reduction:.1f}%")
        
        return output_path
    else:
        print(f"\n❌ Quantization failed")
        return None


def benchmark_quantized(fp32_path: Path, int8_path: Path):
    """Compare FP32 vs INT8 performance."""
    
    print("\n" + "=" * 60)
    print("📊 Benchmarking Quantized Model")
    print("=" * 60)
    
    # Test input
    dummy_input = np.random.randn(1, 48000).astype(np.float32)
    
    # Benchmark FP32
    print(f"\n1️⃣ FP32 Model:")
    fp32_session = ort.InferenceSession(
        str(fp32_path),
        providers=['CPUExecutionProvider']
    )
    
    # Warmup
    for _ in range(10):
        fp32_session.run(None, {'audio': dummy_input})
    
    # Benchmark
    fp32_times = []
    for _ in range(50):
        start = time.perf_counter()
        fp32_session.run(None, {'audio': dummy_input})
        fp32_times.append((time.perf_counter() - start) * 1000)
    
    fp32_times.sort()
    fp32_median = fp32_times[len(fp32_times) // 2]
    fp32_p99 = fp32_times[int(len(fp32_times) * 0.99)]
    
    print(f"   ├─ Median: {fp32_median:.2f} ms")
    print(f"   └─ P99:    {fp32_p99:.2f} ms")
    
    # Benchmark INT8
    print(f"\n2️⃣ INT8 Model:")
    int8_session = ort.InferenceSession(
        str(int8_path),
        providers=['CPUExecutionProvider']
    )
    
    # Warmup
    for _ in range(10):
        int8_session.run(None, {'audio': dummy_input})
    
    # Benchmark
    int8_times = []
    for _ in range(50):
        start = time.perf_counter()
        int8_session.run(None, {'audio': dummy_input})
        int8_times.append((time.perf_counter() - start) * 1000)
    
    int8_times.sort()
    int8_median = int8_times[len(int8_times) // 2]
    int8_p99 = int8_times[int(len(int8_times) * 0.99)]
    
    print(f"   ├─ Median: {int8_median:.2f} ms")
    print(f"   └─ P99:    {int8_p99:.2f} ms")
    
    # Compare
    median_speedup = fp32_median / int8_median
    p99_speedup = fp32_p99 / int8_p99
    
    print(f"\n3️⃣ Speedup:")
    print(f"   ├─ Median: {median_speedup:.2f}x faster")
    print(f"   └─ P99:    {p99_speedup:.2f}x faster")
    
    # Assessment
    print(f"\n4️⃣ Performance Assessment:")
    if int8_p99 < 100:
        print(f"   ✅ EXCELLENT: INT8 P99 < 100ms!")
        print(f"      └─ Production ready")
    elif int8_p99 < 200:
        print(f"   ⚠  GOOD: INT8 P99 < 200ms")
        print(f"      └─ Acceptable for production")
    else:
        print(f"   ❌ NEEDS MORE: INT8 P99 = {int8_p99:.0f}ms")
        print(f"      └─ Consider GPU or model distillation")
    
    return {
        'fp32_median': fp32_median,
        'fp32_p99': fp32_p99,
        'int8_median': int8_median,
        'int8_p99': int8_p99,
        'median_speedup': median_speedup,
        'p99_speedup': p99_speedup
    }


def main():
    # Use the optimized model as base
    input_path = Path("../models/diarization_transformer_optimized.onnx")
    output_path = Path("../models/diarization_transformer_int8.onnx")
    
    if not input_path.exists():
        print(f"❌ Input model not found: {input_path}")
        return
    
    # Quantize
    quantized_path = quantize_model(input_path, output_path)
    
    if not quantized_path:
        print(f"\n❌ Quantization failed")
        return
    
    # Benchmark
    results = benchmark_quantized(input_path, quantized_path)
    
    print("\n" + "=" * 60)
    print("✅ Quantization Complete!")
    print("=" * 60)
    print(f"\nUse {quantized_path.name} for production deployment.")
    print(f"Expected accuracy loss: ~1-2% (acceptable for diarization)")


if __name__ == "__main__":
    main()
