#!/usr/bin/env python3
"""
End-to-end latency test for VoiceFlow Intelligence Platform.

This test validates the complete latency chain:
    1. Network latency (simulated: 10-40ms typical)
    2. Rust processing overhead (simulated: 5-8ms)
    3. Model inference (actual: 4.48ms P99 CPU, 2-4ms GPU)
    
Expected Results:
    - End-to-end P99 < 100ms (production SLA)
    - Typical: 40-80ms P99
    - Components: Network (10-40ms) + Rust (5-8ms) + Model (4.48ms)

Usage:
    # Run with default model
    python tests/hardware/test_end_to_end_latency.py
    
    # Run with custom model and network simulation
    python tests/hardware/test_end_to_end_latency.py \
        --model models/diarization_fast_cnn_optimized.onnx \
        --network-latency-range 10,40 \
        --rust-overhead-range 5,8
    
    # Test worst-case scenario
    python tests/hardware/test_end_to_end_latency.py \
        --network-latency-range 30,50 \
        --rust-overhead-range 8,12
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort


def simulate_network_latency(latency_range: Tuple[float, float]) -> float:
    """
    Simulate network latency (one-way).
    
    In production:
        - Local network: 1-5ms
        - Same region: 5-15ms
        - Cross-region: 20-100ms
        - Global: 100-300ms
    
    Args:
        latency_range: (min_ms, max_ms) for random latency
    
    Returns:
        Simulated latency in milliseconds
    """
    min_ms, max_ms = latency_range
    latency = np.random.uniform(min_ms, max_ms)
    time.sleep(latency / 1000.0)  # Convert to seconds
    return latency


def simulate_rust_overhead(overhead_range: Tuple[float, float]) -> float:
    """
    Simulate Rust processing overhead.
    
    Includes:
        - WebSocket parsing
        - Audio decoding
        - Input validation
        - Response serialization
        - Memory allocation
    
    Typical overhead: 5-8ms
    
    Args:
        overhead_range: (min_ms, max_ms) for random overhead
    
    Returns:
        Simulated overhead in milliseconds
    """
    min_ms, max_ms = overhead_range
    overhead = np.random.uniform(min_ms, max_ms)
    time.sleep(overhead / 1000.0)
    return overhead


def benchmark_end_to_end(
    model_path: str,
    iterations: int,
    network_range: Tuple[float, float],
    rust_range: Tuple[float, float],
    warmup: int = 20
) -> dict:
    """
    Benchmark end-to-end latency including all components.
    
    Args:
        model_path: Path to ONNX model
        iterations: Number of iterations
        network_range: (min, max) network latency in ms
        rust_range: (min, max) Rust overhead in ms
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with latency breakdowns
    """
    print(f"🔍 End-to-End Latency Benchmark")
    print(f"Model: {model_path}")
    print(f"Network latency: {network_range[0]}-{network_range[1]}ms")
    print(f"Rust overhead: {rust_range[0]}-{rust_range[1]}ms")
    print(f"Iterations: {iterations}")
    
    # Create session
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    audio = np.random.randn(1, 48000).astype(np.float32)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        session.run(None, {input_name: audio})
    
    # Benchmark with full latency simulation
    print(f"Running end-to-end benchmark...")
    results = {
        'network_latencies': [],
        'rust_overheads': [],
        'model_latencies': [],
        'total_latencies': []
    }
    
    for i in range(iterations):
        # Simulate request network latency
        network_in = simulate_network_latency(network_range)
        
        # Simulate Rust preprocessing
        rust_pre = simulate_rust_overhead(rust_range)
        
        # Actual model inference
        start = time.perf_counter()
        session.run(None, {input_name: audio})
        model_latency = (time.perf_counter() - start) * 1000
        
        # Simulate Rust postprocessing (typically faster than preprocessing)
        rust_post = simulate_rust_overhead((rust_range[0] * 0.5, rust_range[1] * 0.5))
        
        # Simulate response network latency
        network_out = simulate_network_latency(network_range)
        
        # Calculate totals
        total_network = network_in + network_out
        total_rust = rust_pre + rust_post
        total = total_network + total_rust + model_latency
        
        results['network_latencies'].append(total_network)
        results['rust_overheads'].append(total_rust)
        results['model_latencies'].append(model_latency)
        results['total_latencies'].append(total)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{iterations}")
    
    # Calculate statistics for each component
    def calc_stats(latencies):
        latencies.sort()
        return {
            'min': latencies[0],
            'median': latencies[len(latencies) // 2],
            'mean': sum(latencies) / len(latencies),
            'p95': latencies[int(len(latencies) * 0.95)],
            'p99': latencies[int(len(latencies) * 0.99)],
            'max': latencies[-1]
        }
    
    return {
        'network': calc_stats(results['network_latencies']),
        'rust': calc_stats(results['rust_overheads']),
        'model': calc_stats(results['model_latencies']),
        'total': calc_stats(results['total_latencies']),
        'iterations': iterations
    }


def print_results(stats: dict, sla_p99_ms: float = 100.0):
    """Print formatted end-to-end benchmark results."""
    print("\n" + "="*70)
    print("📊 END-TO-END LATENCY BREAKDOWN")
    print("="*70)
    
    components = ['network', 'rust', 'model', 'total']
    labels = {
        'network': 'Network (round-trip)',
        'rust': 'Rust Processing',
        'model': 'Model Inference',
        'total': 'TOTAL END-TO-END'
    }
    
    for comp in components:
        s = stats[comp]
        print(f"\n{labels[comp]}:")
        print(f"  Min:    {s['min']:.2f} ms")
        print(f"  Median: {s['median']:.2f} ms")
        print(f"  Mean:   {s['mean']:.2f} ms")
        print(f"  P95:    {s['p95']:.2f} ms")
        print(f"  P99:    {s['p99']:.2f} ms")
        print(f"  Max:    {s['max']:.2f} ms")
    
    print("\n" + "="*70)
    print(f"Iterations: {stats['iterations']}")
    print("="*70)
    
    # SLA validation
    total_p99 = stats['total']['p99']
    if total_p99 < sla_p99_ms:
        print(f"\n✅ PASS: End-to-end P99 ({total_p99:.2f}ms) < {sla_p99_ms}ms (Production SLA)")
        return True
    else:
        print(f"\n❌ FAIL: End-to-end P99 ({total_p99:.2f}ms) >= {sla_p99_ms}ms (Production SLA)")
        return False


def main():
    parser = argparse.ArgumentParser(description='End-to-end latency test')
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
        '--network-latency-range',
        type=str,
        default='10,40',
        help='Network latency range in ms (min,max)'
    )
    parser.add_argument(
        '--rust-overhead-range',
        type=str,
        default='5,8',
        help='Rust processing overhead range in ms (min,max)'
    )
    parser.add_argument(
        '--sla-p99',
        type=float,
        default=100.0,
        help='End-to-end P99 SLA threshold in milliseconds'
    )
    
    args = parser.parse_args()
    
    # Parse ranges
    network_range = tuple(map(float, args.network_latency_range.split(',')))
    rust_range = tuple(map(float, args.rust_overhead_range.split(',')))
    
    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        sys.exit(1)
    
    print("🚀 VoiceFlow End-to-End Latency Test")
    print(f"SLA Target: P99 < {args.sla_p99}ms\n")
    
    # Run benchmark
    try:
        stats = benchmark_end_to_end(
            str(model_path),
            args.iterations,
            network_range,
            rust_range
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
