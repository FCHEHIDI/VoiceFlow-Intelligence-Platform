"""
Export trained Transformer model to ONNX using ABSOLUTELY the legacy path.

This forces the old torch.onnx.export path by setting environment variables
and using the raw PyTorch model (not traced).
"""

import os
# MUST be set before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'

import torch
import torch.onnx
from pathlib import Path
import sys

from models.diarization.model import SophisticatedProductionGradeDiarizationModel


def export_legacy_force(
    checkpoint_path: str,
    output_path: str,
    num_speakers: int = 2,
    hidden_size: int = 256,
):
    """Force use of legacy ONNX exporter."""
    
    print("=" * 60)
    print("🔄 Exporting with FORCED Legacy Exporter")
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
    print(f"\n3️⃣ Testing model forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful!")
    print(f"   └─ Output shape: {output.shape}")
    
    # 4. Export using DIRECT torch.jit.script -> onnx save
    print(f"\n4️⃣ Using torch.jit.save -> manual ONNX...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as TorchScript first
    jit_path = output_path.with_suffix('.pt')
    print(f"   ├─ Saving TorchScript to: {jit_path.name}")
    
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced, str(jit_path))
    print(f"   ✓ TorchScript saved")
    
    # Now load and export to ONNX
    print(f"   ├─ Loading TorchScript...")
    loaded_model = torch.jit.load(str(jit_path))
    loaded_model.eval()
    
    print(f"   ├─ Exporting to ONNX...")
    
    # Use torch.onnx._export (internal legacy API)
    try:
        # Try using the internal _export function
        from torch.onnx import utils as onnx_utils
        
        # This should use the old code path
        torch.onnx._export(
            loaded_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            input_names=['audio'],
            output_names=['speaker_probabilities'],
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        print(f"   ✓ ONNX export successful!")
    except Exception as e:
        print(f"   ⚠ Internal _export failed: {e}")
        print(f"   ├─ Trying public torch.onnx.export...")
        
        # Fall back to public API
        torch.onnx.export(
            loaded_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            input_names=['audio'],
            output_names=['speaker_probabilities'],
            verbose=False
        )
        print(f"   ✓ ONNX export successful (public API)!")
    
    print(f"   └─ Saved to: {output_path}")
    
    # 5. Verify file exists
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Export complete!")
        print(f"   ├─ File: {output_path.name}")
        print(f"   ├─ Size: {size_mb:.1f} MB")
        print(f"   └─ Ready for Rust deployment!")
        
        # Clean up TorchScript file
        if jit_path.exists():
            jit_path.unlink()
            print(f"   └─ Cleaned up {jit_path.name}")
    else:
        print(f"\n❌ Export failed - file not created")
        sys.exit(1)


def main():
    checkpoint_path = "../models/checkpoints/transformer_diarization_best.pth"
    output_path = "../models/diarization_transformer.onnx"
    
    try:
        export_legacy_force(
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
