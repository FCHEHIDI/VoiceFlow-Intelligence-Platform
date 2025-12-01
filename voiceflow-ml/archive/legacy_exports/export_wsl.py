#!/usr/bin/env python3
"""
Export trained Transformer model to ONNX - WSL2/Linux version.
Run this in WSL2/Linux where there are no path length limits.
"""

import torch
import torch.onnx
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from models.diarization.model import SophisticatedProductionGradeDiarizationModel


def export_model(checkpoint_path, output_path):
    print("="*60)
    print("ONNX Export - WSL2/Linux")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    model = SophisticatedProductionGradeDiarizationModel(
        num_speakers=2,
        hidden_size=256,
        freeze_encoder=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Checkpoint loaded (epoch {checkpoint.get('epoch', 0)})")
    
    # Create dummy input
    dummy_input = torch.randn(1, 48000)  # 3 seconds @ 16kHz
    
    # Test forward
    print("\n2. Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    
    # Export
    print("\n3. Exporting to ONNX...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        verbose=False
    )
    
    print(f"   Exported to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)


if __name__ == "__main__":
    checkpoint = "../models/checkpoints/transformer_diarization_best.pth"
    output = "../models/diarization_transformer.onnx"
    
    if not Path(checkpoint).exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        sys.exit(1)
    
    export_model(checkpoint, output)
