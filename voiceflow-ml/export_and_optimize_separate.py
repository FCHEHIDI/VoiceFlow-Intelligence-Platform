"""
Export to ONNX without optimization, then optimize with ONNX Runtime.

This bypasses the buggy onnxscript optimizer by:
1. Exporting to ONNX with optimization disabled
2. Using ONNX Runtime's SessionOptions graph optimization
3. Saving the optimized model separately

ONNX Runtime has its own graph optimizer that's more stable.
"""

import torch
import torch.onnx
from pathlib import Path
import sys
import os

# Disable onnxscript optimizer completely
os.environ['ONNX_DISABLE_OPTIMIZATION'] = '1'

from models.diarization.model import SophisticatedProductionGradeDiarizationModel

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("❌ ONNX Runtime required for optimization!")
    sys.exit(1)


def export_unoptimized(
    checkpoint_path: str,
    output_path: str,
    num_speakers: int = 2,
    hidden_size: int = 256,
):
    """Export model to ONNX without optimization."""
    
    print("=" * 60)
    print("🔄 Step 1: Export ONNX (No Optimization)")
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
    dummy_input = torch.randn(1, 48000)
    print(f"   └─ Shape: {dummy_input.shape}")
    
    # 3. Test forward pass
    print(f"\n3️⃣ Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful!")
    print(f"   └─ Output shape: {output.shape}")
    
    # 4. Export to ONNX (bypass optimizer by patching)
    print(f"\n4️⃣ Exporting to ONNX (bypassing onnxscript optimizer)...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Monkey-patch onnxscript optimizer to skip optimization
    try:
        import onnxscript
        original_optimize = onnxscript.optimizer.optimize_ir
        
        def dummy_optimize(model):
            print("   ├─ Skipping onnxscript optimization (buggy)")
            return model
        
        onnxscript.optimizer.optimize_ir = dummy_optimize
        
        print("   ├─ Patched onnxscript.optimizer.optimize_ir")
    except Exception as e:
        print(f"   ⚠ Could not patch onnxscript: {e}")
    
    # Export with patched optimizer
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=False,  # Disabled
        input_names=['audio'],
        output_names=['speaker_probabilities'],
        verbose=False
    )
    
    print(f"   ✓ ONNX export successful (unoptimized)!")
    print(f"   └─ Saved to: {output_path}")
    
    return output_path


def optimize_with_ort(onnx_path: Path):
    """Optimize ONNX model using ONNX Runtime's graph optimizer."""
    
    print("\n" + "=" * 60)
    print("🔄 Step 2: Optimize with ONNX Runtime")
    print("=" * 60)
    
    optimized_path = onnx_path.with_name(onnx_path.stem + "_optimized.onnx")
    
    print(f"\n1️⃣ Loading unoptimized model...")
    print(f"   └─ {onnx_path.name}")
    
    # 2. Create session with optimization enabled
    print(f"\n2️⃣ Creating session with graph optimization...")
    
    sess_options = ort.SessionOptions()
    
    # Enable all optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Save optimized model
    sess_options.optimized_model_filepath = str(optimized_path)
    
    print(f"   ├─ Optimization level: ORT_ENABLE_ALL")
    print(f"   ├─ Includes:")
    print(f"   │  ├─ Constant folding")
    print(f"   │  ├─ Redundant node elimination")
    print(f"   │  ├─ Semantics preserving optimizations")
    print(f"   │  └─ Layout transformations")
    
    # 3. Create session (this triggers optimization)
    print(f"\n3️⃣ Running optimization...")
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"   ✓ Optimization complete!")
    except Exception as e:
        print(f"   ❌ Optimization failed: {e}")
        return None
    
    # 4. Verify optimized model
    if optimized_path.exists():
        orig_size = onnx_path.stat().st_size / (1024 * 1024)
        opt_size = optimized_path.stat().st_size / (1024 * 1024)
        reduction = ((orig_size - opt_size) / orig_size) * 100
        
        print(f"\n4️⃣ Optimization results:")
        print(f"   ├─ Original size: {orig_size:.1f} MB")
        print(f"   ├─ Optimized size: {opt_size:.1f} MB")
        print(f"   └─ Reduction: {reduction:.1f}%")
        
        # 5. Test optimized model
        print(f"\n5️⃣ Testing optimized model...")
        try:
            test_session = ort.InferenceSession(
                str(optimized_path),
                providers=['CPUExecutionProvider']
            )
            
            import numpy as np
            dummy_input = np.random.randn(1, 48000).astype(np.float32)
            
            outputs = test_session.run(
                None,
                {'audio': dummy_input}
            )
            
            print(f"   ✓ Optimized model works!")
            print(f"   └─ Output shape: {outputs[0].shape}")
            
        except Exception as e:
            print(f"   ❌ Optimized model test failed: {e}")
            return None
        
        return optimized_path
    else:
        print(f"\n❌ Optimized model not created")
        return None


def main():
    checkpoint_path = "../models/checkpoints/transformer_diarization_best.pth"
    output_path = "../models/diarization_transformer.onnx"
    
    try:
        # Step 1: Export without optimization
        onnx_path = export_unoptimized(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            num_speakers=2,
            hidden_size=256
        )
        
        if not onnx_path.exists():
            print("\n❌ Export failed")
            sys.exit(1)
        
        # Step 2: Optimize with ONNX Runtime
        optimized_path = optimize_with_ort(onnx_path)
        
        if optimized_path and optimized_path.exists():
            print("\n" + "=" * 60)
            print("✅ Export & Optimization Complete!")
            print("=" * 60)
            print(f"\nFiles:")
            print(f"├─ Unoptimized: {onnx_path.name}")
            print(f"└─ Optimized:   {optimized_path.name}")
            print(f"\n🚀 Use {optimized_path.name} for production deployment!")
            print(f"   This has ORT's full graph optimization applied.")
        else:
            print("\n⚠ Could not optimize - use unoptimized version")
            print(f"   File: {onnx_path.name}")
            
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
