"""
Export to ONNX by skipping the buggy optimization step.

This patches the ONNXProgram.optimize() method to skip the broken
onnxscript optimizer that's causing the constant folding errors.
"""

import torch
import torch.onnx
from pathlib import Path
import sys

from models.diarization.model import SophisticatedProductionGradeDiarizationModel


def export_skip_optimizer(
    checkpoint_path: str,
    output_path: str,
    num_speakers: int = 2,
    hidden_size: int = 256,
):
    """Export model by patching ONNXProgram to skip optimization."""
    
    print("=" * 60)
    print("🔄 Exporting with Optimization Patch")
    print("=" * 60)
    
    # 1. Load model
    print(f"\n1️⃣ Loading model from: {checkpoint_path}")
    model = SophisticatedProductionGradeDiarizationModel(
        num_speakers=num_speakers,
        hidden_size=hidden_size
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Model loaded successfully!")
    print(f"   ├─ Epoch: {checkpoint['epoch']}")
    print(f"   ├─ Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   └─ Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # 2. Create dummy input
    print(f"\n2️⃣ Creating dummy input...")
    dummy_input = torch.randn(1, 48000)  # 3 seconds at 16kHz
    print(f"   └─ Shape: {dummy_input.shape}")
    
    # 3. Test model
    print(f"\n3️⃣ Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful!")
    print(f"   └─ Output shape: {output.shape}")
    
    # 4. Patch ONNXProgram.optimize to skip the buggy optimizer
    print(f"\n4️⃣ Patching ONNX exporter...")
    try:
        from torch.onnx._internal.exporter import _onnx_program
        
        original_optimize = _onnx_program.ONNXProgram.optimize
        
        def patched_optimize(self):
            """Skip optimization to avoid onnxscript bugs."""
            print(f"   ├─ Skipping buggy optimization pass...")
            # Just return without calling the optimizer
            return
        
        _onnx_program.ONNXProgram.optimize = patched_optimize
        print(f"   ✓ Patch applied successfully!")
    except Exception as e:
        print(f"   ⚠ Could not patch: {e}")
        print(f"   └─ Will try without patch...")
    
    # 5. Export to ONNX
    print(f"\n5️⃣ Exporting to ONNX...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,  # Use opset 18 to avoid version conversion
        do_constant_folding=False,
        input_names=['audio'],
        output_names=['speaker_probabilities'],
        verbose=False
    )
    
    print(f"   ✓ ONNX export successful!")
    print(f"   └─ Saved to: {output_path}")
    
    # 6. Verify file exists
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Export complete!")
        print(f"   ├─ File: {output_path.name}")
        print(f"   ├─ Size: {size_mb:.1f} MB")
        print(f"   └─ Ready for Rust deployment!")
    else:
        print(f"\n❌ Export failed - file not created")
        sys.exit(1)


def main():
    checkpoint_path = "../models/checkpoints/transformer_diarization_best.pth"
    output_path = "../models/diarization_transformer.onnx"
    
    try:
        export_skip_optimizer(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            num_speakers=2,
            hidden_size=256
        )
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
