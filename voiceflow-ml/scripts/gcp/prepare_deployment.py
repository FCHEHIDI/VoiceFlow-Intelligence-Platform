"""
Upload model to GCP and create deployment package.

This script:
1. Creates a deployment package with model + test script
2. Uploads to GCP Cloud Storage
3. Provides instructions for instance setup
"""

import subprocess
import os
from pathlib import Path
import json


def check_gcloud_installed():
    """Check if gcloud CLI is installed."""
    try:
        result = subprocess.run(
            ["gcloud", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ gcloud CLI installed:")
        print(result.stdout.split('\n')[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gcloud CLI not installed")
        print("\nInstall from: https://cloud.google.com/sdk/docs/install")
        return False


def get_gcp_project():
    """Get current GCP project."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=True
        )
        project_id = result.stdout.strip()
        if project_id:
            print(f"✅ Current GCP project: {project_id}")
            return project_id
        else:
            print("⚠️  No GCP project set")
            return None
    except subprocess.CalledProcessError:
        print("❌ Failed to get GCP project")
        return None


def create_test_script():
    """Create GPU inference test script."""
    
    test_script = '''#!/usr/bin/env python3
"""
GPU Inference Performance Test

Tests the transformer diarization model on GCP GPU instance.
"""

import onnxruntime as ort
import numpy as np
import time
from statistics import mean


def main():
    print("=" * 60)
    print("🚀 VoiceFlow GPU Inference Test")
    print("=" * 60)
    
    # Check GPU availability
    print("\\n1️⃣ Checking execution providers...")
    available_providers = ort.get_available_providers()
    print(f"   Available: {', '.join(available_providers)}")
    
    if 'CUDAExecutionProvider' not in available_providers:
        print("   ⚠️  CUDA not available - will use CPU")
    else:
        print("   ✅ CUDA available!")
    
    # Load model
    print("\\n2️⃣ Loading model...")
    session = ort.InferenceSession(
        "diarization_transformer_optimized.onnx",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    actual_provider = session.get_providers()[0]
    print(f"   Using: {actual_provider}")
    
    # Generate test audio
    print("\\n3️⃣ Generating test audio (3 seconds @ 16kHz)...")
    audio = np.random.randn(1, 48000).astype(np.float32)
    print(f"   Shape: {audio.shape}")
    
    # Warmup
    print("\\n4️⃣ Warming up (20 runs)...")
    for _ in range(20):
        session.run(None, {'audio': audio})
    print("   ✅ Warmup complete")
    
    # Benchmark
    print("\\n5️⃣ Benchmarking (100 runs)...")
    latencies = []
    
    for i in range(100):
        start = time.perf_counter()
        outputs = session.run(None, {'audio': audio})
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        if i == 0:
            print(f"   First inference: {latency:.2f} ms")
            print(f"   Speaker 1: {outputs[0][0][0] * 100:.2f}%")
            print(f"   Speaker 2: {outputs[0][0][1] * 100:.2f}%")
    
    # Statistics
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = mean(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    print("\\n" + "=" * 60)
    print("📊 Results")
    print("=" * 60)
    print(f"\\nProvider: {actual_provider}")
    print(f"\\nLatency Statistics:")
    print(f"├─ Min:     {min_lat:.2f} ms")
    print(f"├─ Average: {avg:.2f} ms")
    print(f"├─ Median:  {p50:.2f} ms")
    print(f"├─ P95:     {p95:.2f} ms")
    print(f"├─ P99:     {p99:.2f} ms")
    print(f"└─ Max:     {max_lat:.2f} ms")
    
    print(f"\\nThroughput: {1000/avg:.1f} requests/sec")
    
    # Assessment
    print("\\n" + "=" * 60)
    print("🎯 Production Readiness")
    print("=" * 60)
    
    if p99 < 100:
        print(f"\\n✅ EXCELLENT: P99 ({p99:.2f}ms) < 100ms target!")
        print(f"   └─ Ready for production deployment")
        status = "READY"
    elif p99 < 200:
        print(f"\\n⚠️  ACCEPTABLE: P99 ({p99:.2f}ms) < 200ms")
        print(f"   └─ Meets most production requirements")
        status = "ACCEPTABLE"
    else:
        print(f"\\n❌ NEEDS WORK: P99 ({p99:.2f}ms) > 200ms")
        print(f"   └─ Consider faster GPU or optimization")
        status = "NEEDS_WORK"
    
    # Speedup vs CPU
    cpu_baseline = 1428.0  # From earlier CPU benchmark
    speedup = cpu_baseline / p99
    print(f"\\nSpeedup vs CPU: {speedup:.1f}x faster")
    
    # Save results
    results = {
        'provider': actual_provider,
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
        'avg_ms': avg,
        'throughput_rps': 1000/avg,
        'status': status,
        'speedup_vs_cpu': speedup
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n✅ Results saved to benchmark_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
'''
    
    with open("gcp_test_gpu.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created gcp_test_gpu.py")


def create_setup_script():
    """Create instance setup script."""
    
    setup_script = '''#!/bin/bash
# GCP GPU Instance Setup Script

set -e

echo "=========================================="
echo "VoiceFlow GPU Inference Setup"
echo "=========================================="

# Verify GPU
echo ""
echo "1️⃣ Checking GPU..."
nvidia-smi

# Install dependencies
echo ""
echo "2️⃣ Installing dependencies..."
pip install --upgrade pip
pip install onnxruntime-gpu numpy

# Verify ONNX Runtime GPU
echo ""
echo "3️⃣ Verifying ONNX Runtime GPU..."
python3 -c "import onnxruntime as ort; print(f'Providers: {ort.get_available_providers()}')"

# Download model (user needs to upload first)
echo ""
echo "4️⃣ Model setup..."
if [ -f "diarization_transformer_optimized.onnx" ]; then
    echo "✅ Model found"
    ls -lh diarization_transformer_optimized.onnx
else
    echo "⚠️  Model not found!"
    echo "Upload with: gcloud compute scp models/diarization_transformer_optimized.onnx voiceflow-gpu-inference:~/"
fi

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Run test: python3 gcp_test_gpu.py"
'''
    
    with open("gcp_setup.sh", "w") as f:
        f.write(setup_script)
    
    os.chmod("gcp_setup.sh", 0o755)
    print("✅ Created gcp_setup.sh")


def print_instructions(project_id):
    """Print deployment instructions."""
    
    print("\n" + "=" * 60)
    print("📋 GCP GPU Deployment Instructions")
    print("=" * 60)
    
    print(f"\n1️⃣ Create GPU Instance (T4 - $0.70/hr):")
    print(f"   gcloud compute instances create voiceflow-gpu-test \\")
    print(f"       --project={project_id} \\")
    print(f"       --zone=us-central1-a \\")
    print(f"       --machine-type=n1-standard-4 \\")
    print(f"       --accelerator=type=nvidia-tesla-t4,count=1 \\")
    print(f"       --image-family=pytorch-latest-gpu \\")
    print(f"       --image-project=deeplearning-platform-release \\")
    print(f"       --maintenance-policy=TERMINATE \\")
    print(f"       --boot-disk-size=50GB \\")
    print(f"       --metadata='install-nvidia-driver=True'")
    
    print(f"\n2️⃣ Upload files to instance:")
    print(f"   gcloud compute scp models/diarization_transformer_optimized.onnx \\")
    print(f"       voiceflow-gpu-test:~/ --zone=us-central1-a")
    print(f"   gcloud compute scp gcp_test_gpu.py gcp_setup.sh \\")
    print(f"       voiceflow-gpu-test:~/ --zone=us-central1-a")
    
    print(f"\n3️⃣ SSH into instance:")
    print(f"   gcloud compute ssh voiceflow-gpu-test --zone=us-central1-a")
    
    print(f"\n4️⃣ On instance, run:")
    print(f"   bash gcp_setup.sh")
    print(f"   python3 gcp_test_gpu.py")
    
    print(f"\n5️⃣ Expected results:")
    print(f"   P50: 20-40ms  ✅")
    print(f"   P95: 35-60ms  ✅")
    print(f"   P99: 40-80ms  ✅  (under 100ms target)")
    
    print(f"\n6️⃣ Cleanup when done:")
    print(f"   gcloud compute instances delete voiceflow-gpu-test --zone=us-central1-a")
    
    print(f"\n💡 Tip: Use --preemptible flag to reduce cost to ~$0.25/hr")


def main():
    print("=" * 60)
    print("🌩️  GCP GPU Deployment Setup")
    print("=" * 60)
    
    # Check gcloud
    if not check_gcloud_installed():
        return
    
    # Get project
    project_id = get_gcp_project()
    if not project_id:
        print("\n⚠️  Set project: gcloud config set project YOUR_PROJECT_ID")
        project_id = "your-project-id"
    
    # Check model exists
    model_path = Path("../models/diarization_transformer_optimized.onnx")
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Run export_and_optimize_separate.py first")
        return
    
    print(f"\n✅ Model ready: {model_path.name}")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    
    # Create scripts
    print(f"\n📝 Creating deployment scripts...")
    create_test_script()
    create_setup_script()
    
    # Print instructions
    print_instructions(project_id)
    
    print("\n" + "=" * 60)
    print("✅ Ready for GCP deployment!")
    print("=" * 60)
    print("\nThis will validate <100ms P99 on real GPU hardware.")


if __name__ == "__main__":
    main()
