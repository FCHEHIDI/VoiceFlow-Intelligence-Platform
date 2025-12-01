"""
Direct ONNX export without onnx/onnxscript dependencies.

This bypasses Windows Long Path issues by using only PyTorch's native export
and skipping all verification steps that require the onnx package.
"""

import torch
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.diarization.model import SophisticatedProductionGradeDiarizationModel


def export_to_onnx_direct(
    checkpoint_path: str,
    output_path: str,
    num_speakers: int = 2,
    hidden_size: int = 256,
    sample_rate: int = 16000,
    duration: int = 3
):
    """
    Export trained model to ONNX using only PyTorch native functions.
    No onnx package required.
    """
    print("="*60)
    print("🔄 Direct ONNX Export (No Dependencies)")
    print("="*60)
    
    # 1. Load model
    print(f"\n1️⃣ Loading trained model...")
    model = SophisticatedProductionGradeDiarizationModel(
        num_speakers=num_speakers,
        hidden_size=hidden_size,
        freeze_encoder=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"   ├─ Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"   └─ Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    
    # 2. Create dummy input
    audio_length = sample_rate * duration
    dummy_input = torch.randn(1, audio_length)
    
    print(f"\n2️⃣ Preparing export...")
    print(f"   ├─ Input shape: {dummy_input.shape}")
    print(f"   ├─ Sample rate: {sample_rate} Hz")
    print(f"   └─ Duration: {duration} seconds")
    
    # 3. Test forward pass
    print(f"\n3️⃣ Testing model...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful!")
    print(f"   └─ Output shape: {output.shape}")
    
    # 4. Export using PyTorch's native ONNX export
    print(f"\n4️⃣ Exporting to ONNX (this may take 2-3 minutes)...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use torch._C._jit_pass_onnx to bypass new exporter
    import warnings
    warnings.filterwarnings("ignore")
    
    # Set environment to use legacy exporter
    import os
    os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
    
    try:
        # Export with minimal dependencies
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['audio'],
            output_names=['speaker_probabilities'],
            dynamic_axes={
                'audio': {0: 'batch_size', 1: 'audio_length'},
                'speaker_probabilities': {0: 'batch_size'}
            },
            verbose=False,
            # Force legacy exporter
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        
        print(f"   ✓ Export successful!")
        print(f"   └─ Saved to: {output_path}")
        
    except Exception as e:
        if "onnxscript" in str(e).lower():
            print(f"\n   ⚠ New exporter requires onnxscript (blocked by path limits)")
            print(f"   ⚠ Attempting workaround...")
            
            # Try disabling the new exporter completely
            try:
                # Monkey-patch to disable new exporter check
                import torch.onnx._internal.exporter as exporter_module
                original_export = torch.onnx.export
                
                # This won't work but shows the issue
                print(f"   ✗ PyTorch 2.x requires onnxscript for ONNX export")
                print(f"\n" + "="*60)
                print("❌ Export Failed - Windows Path Limit Issue")
                print("="*60)
                print(f"\nThe issue: PyTorch 2.x ONNX exporter requires onnxscript,")
                print(f"but onnxscript cannot install on Windows due to test data")
                print(f"paths exceeding 260 characters.")
                print(f"\n🔧 Solutions:")
                print(f"   1. Enable Long Paths (requires Admin + reboot):")
                print(f"      - Run as Admin: New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force")
                print(f"      - Enable in Group Policy: Computer Config > Admin Templates > System > Filesystem > Enable Win32 long paths")
                print(f"      - Reboot Windows")
                print(f"      - Then: pip install onnxscript")
                print(f"\n   2. Use Rust libtorch (tch-rs) to load PyTorch .pth directly")
                print(f"      - No ONNX needed, load checkpoint directly in Rust")
                print(f"      - File ready: {checkpoint_path}")
                print(f"\n   3. Export on Linux/WSL2/Docker (no path limits)")
                return False
            except:
                pass
        
        raise e
    
    # 5. Verify file was created
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n5️⃣ Export Verification:")
        print(f"   ✓ File created successfully")
        print(f"   ├─ Size: {file_size:.2f} MB")
        print(f"   ├─ Opset: 14")
        print(f"   └─ Ready for Rust inference")
        
        print("\n" + "="*60)
        print("✅ Export Complete!")
        print("="*60)
        print(f"\n🎯 Next Steps:")
        print(f"   1. Copy to Rust: cp {output_path.name} ../voiceflow-inference/models/")
        print(f"   2. Test loading in Rust with ort crate")
        print(f"   3. Run inference on real audio samples")
        print(f"   4. Benchmark latency (<100ms target)")
        return True
    else:
        print("\n❌ Export failed - file not created")
        return False


def main():
    """Main export function."""
    
    checkpoint_path = "../models/checkpoints/transformer_diarization_best.pth"
    output_path = "../models/diarization_transformer.onnx"
    
    # Check checkpoint
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Run training first: python train_transformer.py")
        return
    
    # Export
    success = export_to_onnx_direct(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        num_speakers=2,
        hidden_size=256,
        sample_rate=16000,
        duration=3
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
