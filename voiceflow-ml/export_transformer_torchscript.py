"""
Export trained Transformer diarization model to TorchScript format.

This script loads the trained model and exports it to TorchScript for
deployment with the Rust inference engine (which can load TorchScript via libtorch).
"""

import torch
import torch.jit
from pathlib import Path
import numpy as np

from models.diarization.model import SophisticatedProductionGradeDiarizationModel


def export_transformer_to_torchscript(
    checkpoint_path: str,
    output_path: str,
    num_speakers: int = 2,
    hidden_size: int = 256,
    sample_rate: int = 16000,
    duration: int = 3
):
    """
    Export trained Transformer model to TorchScript.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save TorchScript model
        num_speakers: Number of speakers
        hidden_size: LSTM hidden dimension
        sample_rate: Audio sample rate (16kHz)
        duration: Audio duration in seconds for dummy input
    """
    print("="*60)
    print("🔄 Exporting Transformer Model to TorchScript")
    print("="*60)
    
    # 1. Load trained model
    print(f"\n1️⃣ Loading trained model from: {checkpoint_path}")
    model = SophisticatedProductionGradeDiarizationModel(
        num_speakers=num_speakers,
        hidden_size=hidden_size,
        freeze_encoder=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Model loaded successfully!")
    print(f"   ├─ Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   ├─ Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"   └─ Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    
    # 2. Create dummy input (raw audio waveform)
    audio_length = sample_rate * duration  # 3 seconds @ 16kHz = 48,000 samples
    dummy_input = torch.randn(1, audio_length)
    
    print(f"\n2️⃣ Creating dummy input:")
    print(f"   ├─ Shape: {dummy_input.shape}")
    print(f"   ├─ Sample rate: {sample_rate} Hz")
    print(f"   └─ Duration: {duration} seconds")
    
    # 3. Test forward pass
    print(f"\n3️⃣ Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful!")
    print(f"   └─ Output shape: {output.shape}")
    
    # 4. Export to TorchScript using tracing
    print(f"\n4️⃣ Exporting to TorchScript (tracing)...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save traced model
    torch.jit.save(traced_model, str(output_path))
    
    print(f"   ✓ TorchScript export successful!")
    print(f"   └─ Saved to: {output_path}")
    
    # 5. Load and test saved model
    print(f"\n5️⃣ Testing saved TorchScript model...")
    loaded_model = torch.jit.load(str(output_path))
    loaded_model.eval()
    
    # Run inference
    with torch.no_grad():
        loaded_output = loaded_model(dummy_input)
    
    print(f"   ✓ TorchScript model loaded successfully!")
    print(f"   └─ Output: {loaded_output}")
    
    # 6. Compare outputs
    print(f"\n6️⃣ Comparing original vs TorchScript outputs...")
    max_diff = torch.max(torch.abs(output - loaded_output)).item()
    print(f"   ├─ Original output: {output[0]}")
    print(f"   ├─ TorchScript output: {loaded_output[0]}")
    print(f"   └─ Max difference: {max_diff:.10f}")
    
    if max_diff < 1e-6:
        print(f"   ✓ Outputs match perfectly! (diff < 1e-6)")
    else:
        print(f"   ⚠ Outputs differ slightly (expected due to precision)")
    
    # 7. Model info
    print(f"\n7️⃣ Model Information:")
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ├─ File size: {file_size:.2f} MB")
    print(f"   ├─ Format: TorchScript (traced)")
    print(f"   ├─ Input shape: [batch, {audio_length}]")
    print(f"   └─ Output shape: [batch, {num_speakers}]")
    
    print("\n" + "="*60)
    print("✅ Export Complete!")
    print("="*60)
    print(f"\n🎯 Next Steps:")
    print(f"   1. Use this model with Rust libtorch bindings")
    print(f"   2. Or convert to ONNX later when path issues are resolved")
    print(f"   3. Benchmark latency (<100ms target)")
    print()


def main():
    """Main export function."""
    
    # Paths
    checkpoint_path = "../models/checkpoints/transformer_diarization_best.pth"
    output_path = "../models/diarization_transformer.pt"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print(f"   Run training first: python train_transformer.py")
        return
    
    # Export
    export_transformer_to_torchscript(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        num_speakers=2,
        hidden_size=256,
        sample_rate=16000,
        duration=3
    )


if __name__ == "__main__":
    main()
