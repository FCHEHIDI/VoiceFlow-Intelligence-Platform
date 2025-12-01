#!/usr/bin/env python3
"""
CPU benchmark test for Fast CNN model.

Validates that model achieves < 10ms P99 latency on CPU (SLA requirement).
This test uses the actual trained Fast CNN model and verifies performance
against production SLA targets.

Expected Results:
    - P99 < 10ms (model inference SLA)
    - Median ~3-4ms
    - Actual validated: 4.48ms P99 on Intel/AMD 4-core CPU

Usage:
    # Run with default model
    python tests/hardware/test_cpu_benchmark.py
    
    # Run with custom model
    python tests/hardware/test_cpu_benchmark.py --model path/to/model.onnx
    
    # Custom iterations
    python tests/hardware/test_cpu_benchmark.py --iterations 500
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort


def benchmark_cpu_inference(
    model_path: str,
    iterations: int = 200,
    warmup: int = 20
) -> dict:
    """
    Benchmark CPU inference performance.
    
    Args:
        model_path: Path to ONNX model
        iterations: Number of inference iterations
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with latency statistics
    """
    print(f"🔍 Benchmarking CPU inference: {model_path}")
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    
    # Create CPU-only session
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    
    provider = session.get_providers()[0]
    print(f"Provider: {provider}")
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name} {input_shape}")
    
    # Generate test audio (1 second at 48kHz)
    audio = np.random.randn(1, 48000).astype(np.float32)
    
    # Warmup
    print(f"\nWarming up...")
    for _ in range(warmup):
        session.run(None, {input_name: audio})
    
    # Benchmark
    print(f"Running benchmark...")
    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: audio})
        latencies.append((time.perf_counter() - start) * 1000)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    # Calculate statistics
    latencies.sort()
    stats = {
        'min': latencies[0],
        'median': latencies[len(latencies) // 2],
        'mean': sum(latencies) / len(latencies),
        'p95': latencies[int(len(latencies) * 0.95)],
        'p99': latencies[int(len(latencies) * 0.99)],
        'max': latencies[-1],
        'iterations': iterations
    }
    
    return stats


def print_results(stats: dict, sla_p99_ms: float = 10.0):
    """Print formatted benchmark results."""
    print("\n" + "="*60)
    print("📊 CPU BENCHMARK RESULTS")
    print("="*60)
    print(f"Min:        {stats['min']:.2f} ms")
    print(f"Median:     {stats['median']:.2f} ms")
    print(f"Mean:       {stats['mean']:.2f} ms")
    print(f"P95:        {stats['p95']:.2f} ms")
    print(f"P99:        {stats['p99']:.2f} ms")
    print(f"Max:        {stats['max']:.2f} ms")
    print(f"Iterations: {stats['iterations']}")
    print("="*60)
    
    # SLA validation
    if stats['p99'] < sla_p99_ms:
        print(f"✅ PASS: P99 ({stats['p99']:.2f}ms) < {sla_p99_ms}ms (Model Inference SLA)")
        return True
    else:
        print(f"❌ FAIL: P99 ({stats['p99']:.2f}ms) >= {sla_p99_ms}ms (Model Inference SLA)")
        return False


def main():
    parser = argparse.ArgumentParser(description='CPU benchmark test for speaker diarization model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/diarization_fast_cnn_optimized.onnx',
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=200,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=20,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--sla-p99',
        type=float,
        default=10.0,
        help='P99 latency SLA threshold in milliseconds'
    )
    
    args = parser.parse_args()
    
    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Expected path: {model_path.absolute()}")
        sys.exit(1)
    
    print("🚀 VoiceFlow CPU Benchmark Test")
    print(f"Model: {model_path}")
    print(f"SLA Target: P99 < {args.sla_p99}ms\n")
    
    # Run benchmark
    try:
        stats = benchmark_cpu_inference(
            str(model_path),
            args.iterations,
            args.warmup
        )
        
        # Print and validate results
        passed = print_results(stats, args.sla_p99)
        
        # Exit with appropriate code
        sys.exit(0 if passed else 1)
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
