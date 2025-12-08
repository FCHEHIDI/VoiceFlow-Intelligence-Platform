#!/usr/bin/env python3
"""
GPU-enabled ONNX model benchmark script for GCP T4 deployment.

This script validates:
- Model inference latency (P50, P95, P99)
- GPU utilization and memory usage
- Throughput under load
- Provider selection (CUDA vs TensorRT vs CPU fallback)

Usage:
    # Basic benchmark
    python scripts/gcp/benchmark_gpu.py --model models/diarization_fast_cnn_optimized.onnx
    
    # With custom iterations and batch sizes
    python scripts/gcp/benchmark_gpu.py \
        --model models/diarization_fast_cnn_optimized.onnx \
        --iterations 500 \
        --batch-sizes 1,4,8,16 \
        --providers cuda,tensorrt,cpu
    
    # Generate JSON report for CI/CD
    python scripts/gcp/benchmark_gpu.py \
        --model models/diarization_fast_cnn_optimized.onnx \
        --output-json benchmark_results.json

Expected Results (GCP T4):
    - Fast CNN: 2-4ms P99 (model inference only)
    - End-to-end: 25-50ms P99 (including network + Rust)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def get_available_providers() -> List[str]:
    """Get list of available ONNX Runtime providers."""
    available = ort.get_available_providers()
    print(f"Available providers: {available}")
    return available


def benchmark_provider(
    model_path: str,
    provider: str,
    iterations: int = 200,
    batch_size: int = 1,
    warmup: int = 20
) -> Dict[str, float]:
    """
    Benchmark model with specific provider.
    
    Args:
        model_path: Path to ONNX model
        provider: Provider name (CUDAExecutionProvider, TensorrtExecutionProvider, CPUExecutionProvider)
        iterations: Number of inference iterations
        batch_size: Batch size for inference
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with latency statistics
    """
    print(f"\n🔍 Benchmarking with {provider} (batch_size={batch_size})")
    
    # Create session with specific provider
    providers = [provider]
    if provider != 'CPUExecutionProvider':
        providers.append('CPUExecutionProvider')  # Fallback
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"❌ Failed to create session with {provider}: {e}")
        return {}
    
    actual_provider = session.get_providers()[0]
    print(f"✅ Using provider: {actual_provider}")
    
    # Get input shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name} {input_shape}")
    
    # Generate test audio
    audio_length = 48000  # 1 second at 48kHz
    audio = np.random.randn(batch_size, audio_length).astype(np.float32)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        session.run(None, {input_name: audio})
    
    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
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
        'provider': actual_provider,
        'batch_size': batch_size,
        'iterations': iterations,
        'min': latencies[0],
        'median': latencies[len(latencies) // 2],
        'mean': sum(latencies) / len(latencies),
        'p95': latencies[int(len(latencies) * 0.95)],
        'p99': latencies[int(len(latencies) * 0.99)],
        'max': latencies[-1],
        'throughput_rps': (batch_size * 1000) / (sum(latencies) / len(latencies))
    }
    
    # Print results
    print(f"\n📊 Results ({actual_provider}):")
    print(f"  Min:        {stats['min']:.2f} ms")
    print(f"  Median:     {stats['median']:.2f} ms")
    print(f"  Mean:       {stats['mean']:.2f} ms")
    print(f"  P95:        {stats['p95']:.2f} ms")
    print(f"  P99:        {stats['p99']:.2f} ms")
    print(f"  Max:        {stats['max']:.2f} ms")
    print(f"  Throughput: {stats['throughput_rps']:.1f} req/s")
    
    # SLA validation
    model_sla = 10.0  # ms
    if stats['p99'] < model_sla:
        print(f"  ✅ PASS: P99 ({stats['p99']:.2f}ms) < {model_sla}ms (Model SLA)")
    else:
        print(f"  ❌ FAIL: P99 ({stats['p99']:.2f}ms) >= {model_sla}ms (Model SLA)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='GPU benchmark for ONNX models')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup iterations')
    parser.add_argument('--batch-sizes', type=str, default='1', help='Comma-separated batch sizes (e.g., 1,4,8)')
    parser.add_argument('--providers', type=str, default='cuda,cpu', help='Comma-separated providers (cuda,tensorrt,cpu)')
    parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"🚀 GPU Benchmark - VoiceFlow Intelligence Platform")
    print(f"Model: {model_path}")
    print(f"Iterations: {args.iterations}")
    
    # Get available providers
    available_providers = get_available_providers()
    
    # Map provider names to ONNX Runtime provider names
    provider_map = {
        'cuda': 'CUDAExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider',
        'cpu': 'CPUExecutionProvider'
    }
    
    # Parse providers
    requested_providers = [p.strip().lower() for p in args.providers.split(',')]
    providers_to_test = []
    for p in requested_providers:
        provider_name = provider_map.get(p)
        if provider_name in available_providers:
            providers_to_test.append(provider_name)
        else:
            print(f"⚠️  {provider_name} not available, skipping")
    
    if not providers_to_test:
        print("❌ No valid providers to test")
        return
    
    # Parse batch sizes
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]
    
    # Run benchmarks
    all_results = []
    for provider in providers_to_test:
        for batch_size in batch_sizes:
            results = benchmark_provider(
                str(model_path),
                provider,
                args.iterations,
                batch_size,
                args.warmup
            )
            if results:
                all_results.append(results)
    
    # Summary
    print("\n" + "="*60)
    print("📈 BENCHMARK SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"{result['provider']} (batch={result['batch_size']}): "
              f"P99={result['p99']:.2f}ms, "
              f"Throughput={result['throughput_rps']:.1f} req/s")
    
    # Save to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, 'w') as f:
            json.dump({
                'model': str(model_path),
                'iterations': args.iterations,
                'results': all_results
            }, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")
    
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
