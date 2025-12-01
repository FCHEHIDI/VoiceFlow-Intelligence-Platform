"""
One-command GCP GPU deployment and testing.

After gcloud is installed, run this to:
1. Create GPU instance
2. Upload model
3. Run benchmark
4. Get results
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run command and show output."""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True,
        check=check
    )
    
    return result.returncode == 0


def main():
    print("=" * 60)
    print("🚀 GCP GPU Quick Deploy & Test")
    print("=" * 60)
    
    # Configuration
    INSTANCE_NAME = "voiceflow-gpu-test"
    ZONE = "us-central1-a"
    MODEL_PATH = "../models/diarization_transformer_optimized.onnx"
    
    # Check model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    print(f"\n✅ Model ready: {Path(MODEL_PATH).stat().st_size / (1024*1024):.1f} MB")
    
    # Get project ID
    print("\n📋 Checking GCP project...")
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True
    )
    project_id = result.stdout.strip()
    
    if not project_id:
        print("❌ No GCP project configured")
        print("Run: gcloud init")
        sys.exit(1)
    
    print(f"✅ Project: {project_id}")
    
    # Confirm deployment
    print(f"\n{'='*60}")
    print("📊 Deployment Plan:")
    print(f"{'='*60}")
    print(f"Instance: {INSTANCE_NAME}")
    print(f"Zone: {ZONE}")
    print(f"GPU: NVIDIA T4 (1x)")
    print(f"Cost: ~$0.70/hour")
    print(f"Model: {MODEL_PATH}")
    
    response = input("\n🔵 Deploy? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        sys.exit(0)
    
    # Step 1: Create instance
    if not run_command(
        f'gcloud compute instances create {INSTANCE_NAME} '
        f'--zone={ZONE} '
        f'--machine-type=n1-standard-4 '
        f'--accelerator=type=nvidia-tesla-t4,count=1 '
        f'--image-family=pytorch-latest-gpu '
        f'--image-project=deeplearning-platform-release '
        f'--maintenance-policy=TERMINATE '
        f'--boot-disk-size=50GB '
        f'--metadata="install-nvidia-driver=True"',
        "Creating GPU instance (this takes 3-5 minutes)"
    ):
        print("❌ Failed to create instance")
        sys.exit(1)
    
    # Wait for instance to be ready
    print("\n⏳ Waiting for instance to fully boot (60 seconds)...")
    time.sleep(60)
    
    # Step 2: Upload files
    print("\n📤 Uploading model and test scripts...")
    
    files_to_upload = [
        (MODEL_PATH, "diarization_transformer_optimized.onnx"),
        ("gcp_test_gpu.py", "gcp_test_gpu.py"),
        ("gcp_setup.sh", "gcp_setup.sh"),
    ]
    
    for local_path, remote_name in files_to_upload:
        if Path(local_path).exists():
            run_command(
                f'gcloud compute scp {local_path} {INSTANCE_NAME}:~/{remote_name} --zone={ZONE}',
                f"Uploading {remote_name}",
                check=False
            )
    
    # Step 3: Setup and test
    print("\n🔧 Running setup and benchmark on GPU...")
    
    test_commands = f'''
        chmod +x gcp_setup.sh && 
        bash gcp_setup.sh && 
        python3 gcp_test_gpu.py
    '''
    
    run_command(
        f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="{test_commands}"',
        "Running GPU benchmark",
        check=False
    )
    
    # Step 4: Download results
    print("\n📥 Downloading results...")
    run_command(
        f'gcloud compute scp {INSTANCE_NAME}:~/benchmark_results.json benchmark_results_gpu.json --zone={ZONE}',
        "Getting benchmark results",
        check=False
    )
    
    if Path("benchmark_results_gpu.json").exists():
        import json
        with open("benchmark_results_gpu.json") as f:
            results = json.load(f)
        
        print("\n" + "="*60)
        print("🎯 GPU Performance Results")
        print("="*60)
        print(f"Provider: {results['provider']}")
        print(f"P50: {results['p50_ms']:.2f} ms")
        print(f"P95: {results['p95_ms']:.2f} ms")
        print(f"P99: {results['p99_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_rps']:.1f} req/sec")
        print(f"Speedup vs CPU: {results['speedup_vs_cpu']:.1f}x")
        print(f"\nStatus: {results['status']}")
    
    # Cleanup prompt
    print("\n" + "="*60)
    print("🧹 Cleanup")
    print("="*60)
    response = input("\n🔵 Delete instance now? (yes/no): ")
    if response.lower() == 'yes':
        run_command(
            f'gcloud compute instances delete {INSTANCE_NAME} --zone={ZONE} --quiet',
            "Deleting instance",
            check=False
        )
        print("\n✅ Instance deleted")
    else:
        print(f"\n⚠️  Instance still running (costing money)")
        print(f"Delete later: gcloud compute instances delete {INSTANCE_NAME} --zone={ZONE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
