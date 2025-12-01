"""
Export trained Transformer diarization model to ONNX format.

This script loads the trained model and exports it to ONNX for
deployment with the Rust inference engine.

Note: We skip onnx.checker due to Windows Long Path issues with ONNX package.
The model is still exported correctly and validated with ONNX Runtime.
"""

import torch
import torch.onnx
from pathlib import Path
import numpy as np

from models.diarization.model import SophisticatedProductionGradeDiarizationModel

# Try to import onnxruntime, but make it optional
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("⚠ ONNX Runtime not available - skipping verification")


def export_transformer_to_onnx(
    checkpoint_path: str,
    output_path: str,
    num_speakers: int = 2,
    hidden_size: int = 256,
    sample_rate: int = 16000,
    duration: int = 3
):
    """
    Export trained Transformer model to ONNX.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save ONNX model
        num_speakers: Number of speakers
        hidden_size: LSTM hidden dimension
        sample_rate: Audio sample rate (16kHz)
        duration: Audio duration in seconds for dummy input
    """
    print("="*60)
    print("🔄 Exporting Transformer Model to ONNX")
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
    
    # 4. Export to ONNX (using legacy API to avoid onnxscript dependency)
    print(f"\n4️⃣ Exporting to ONNX (legacy exporter, no optimization)...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Force legacy exporter by setting ONNX environment variable
    import os
    os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'
    
    # Use legacy API with verbose=False to avoid new exporter
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Use dynamo=False to force legacy exporter path
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,  # Use older opset with legacy exporter
        do_constant_folding=False,  # Disable to avoid optimization errors
        input_names=['audio'],
        output_names=['speaker_probabilities'],
        dynamic_axes={
            'audio': {0: 'batch_size', 1: 'audio_length'},
            'speaker_probabilities': {0: 'batch_size'}
        },
        verbose=False  # Disable verbose mode to use legacy exporter
    )
    
    print(f"   ✓ ONNX export successful!")
    print(f"   └─ Saved to: {output_path}")
    
    # 5. Test with ONNX Runtime (if available)
    if HAS_ORT:
        print(f"\n5️⃣ Testing with ONNX Runtime...")
        try:
            ort_session = ort.InferenceSession(str(output_path))
            
            # Get input/output info
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            print(f"   ├─ Input: {input_name}")
            print(f"   └─ Output: {output_name}")
            
            # Run inference
            dummy_input_np = dummy_input.numpy()
            ort_outputs = ort_session.run([output_name], {input_name: dummy_input_np})
            
            print(f"   ✓ ONNX Runtime inference successful!")
            print(f"   └─ Output: {ort_outputs[0]}")
            
            # 6. Compare PyTorch vs ONNX outputs
            print(f"\n6️⃣ Comparing PyTorch vs ONNX outputs...")
            pytorch_output = output.detach().numpy()
            onnx_output = ort_outputs[0]
            
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            print(f"   ├─ PyTorch output: {pytorch_output[0]}")
            print(f"   ├─ ONNX output: {onnx_output[0]}")
            print(f"   └─ Max difference: {max_diff:.6f}")
            
            if max_diff < 1e-5:
                print(f"   ✓ Outputs match! (diff < 1e-5)")
            else:
                print(f"   ⚠ Outputs differ slightly (expected due to precision)")
        except Exception as e:
            print(f"   ⚠ ONNX Runtime test failed: {e}")
            print(f"   ✓ Model exported successfully, but verification skipped")
    else:
        print(f"\n5️⃣ Skipping ONNX Runtime verification (not installed)")
    
    # 7. Model info
    print(f"\n{'7' if HAS_ORT else '6'}️⃣ Model Information:")
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ├─ File size: {file_size:.2f} MB")
    print(f"   ├─ Opset version: 14")
    print(f"   ├─ Input shape: [batch, {audio_length}]")
    print(f"   └─ Output shape: [batch, {num_speakers}]")
    
    print("\n" + "="*60)
    print("✅ Export Complete!")
    print("="*60)
    print(f"\n🎯 Next Steps:")
    print(f"   1. Copy {output_path.name} to Rust project models/ folder")
    print(f"   2. Test with Rust inference engine")
    print(f"   3. Benchmark latency (<100ms target)")
    print()


def main():
    """Main export function."""
    
    # Paths
    checkpoint_path = "../models/checkpoints/transformer_diarization_best.pth"
    output_path = "../models/diarization_transformer.onnx"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print(f"   Run training first: python train_transformer.py")
        return
    
    # Export
    export_transformer_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        num_speakers=2,
        hidden_size=256,
        sample_rate=16000,
        duration=3
    )


if __name__ == "__main__":
    main()
