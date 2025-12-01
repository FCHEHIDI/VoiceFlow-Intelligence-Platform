"""
Fast CNN Model Training Script

Train the lightweight FastDiarizationModel for CPU-optimized inference.
This model has 2.3M parameters (42x smaller than Wav2Vec2) and is designed
to achieve <100ms P99 latency on CPU.

Usage:
    python train_fast_cnn.py --epochs 10 --batch-size 16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
import argparse
from typing import Optional

from models.diarization.model import FastDiarizationModel, ModelConfig


class QuickAudioDataset(Dataset):
    """
    Quick synthetic dataset for demonstrating training.
    Replace with your real audio data for production.
    """
    
    def __init__(self, num_samples: int = 500, sample_rate: int = 16000, duration: int = 3):
        self.num_samples = num_samples
        self.audio_length = sample_rate * duration
        
        print(f"Generating {num_samples} synthetic audio samples...")
        # Generate synthetic audio
        self.audio_data = torch.randn(num_samples, self.audio_length)
        self.labels = torch.randint(0, 2, (num_samples,))
        print(f"✓ Dataset ready")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.audio_data[idx], self.labels[idx]


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        
        # Forward + backward
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(train_loader) - 1:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            print(f"  [{epoch+1}/{total_epochs}] Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%", end='\r')
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    print(f"\n  Epoch {epoch+1}/{total_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.1f}s")
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            
            outputs = model(audio)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print(f"  Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Fast CNN diarization model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=500, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=100, help="Validation samples")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", 
                       help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("\n" + "="*70)
    print("🚀 Training Fast CNN Diarization Model")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    print("="*70 + "\n")
    
    # Create model
    print("Creating FastDiarizationModel (Lightweight CNN)...")
    model = FastDiarizationModel(
        num_speakers=2,
        hidden_size=256,
        encoder_type="lightweight-cnn",
        freeze_encoder=False,  # Train entire model
        dropout=0.1,
    )
    model = model.to(device)
    
    print(f"✓ Model created: {model.count_parameters() / 1e6:.1f}M parameters")
    print(f"  └─ All trainable (no frozen encoder)\n")
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = QuickAudioDataset(num_samples=args.train_samples)
    val_dataset = QuickAudioDataset(num_samples=args.val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("="*70)
    print("Starting Training...")
    print("="*70 + "\n")
    
    best_val_loss = float('inf')
    training_start = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "fast_cnn_diarization_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'model_config': {
                    'num_speakers': 2,
                    'hidden_size': 256,
                    'encoder_type': 'lightweight-cnn',
                },
            }, checkpoint_path)
            print(f"  💾 Best model saved: {checkpoint_path}\n")
        else:
            print()
    
    total_time = time.time() - training_start
    
    # Final summary
    print("="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {checkpoint_dir / 'fast_cnn_diarization_best.pth'}")
    print("="*70 + "\n")
    
    print("🎯 Next Steps:")
    print("\n1️⃣ Export to ONNX:")
    print("   python -m models.diarization.export_onnx \\")
    print("       --checkpoint models/checkpoints/fast_cnn_diarization_best.pth \\")
    print("       --model-type fast-cnn \\")
    print("       --output-dir models \\")
    print("       --output-name fast_cnn_diarization \\")
    print("       --optimization-level all")
    
    print("\n2️⃣ Benchmark on CPU:")
    print("   python -m models.diarization.benchmark \\")
    print("       --model models/fast_cnn_diarization_optimized.onnx \\")
    print("       --provider CPUExecutionProvider \\")
    print("       --iterations 100")
    
    print("\n3️⃣ Compare with sophisticated model:")
    print("   python -m models.diarization.benchmark \\")
    print("       --compare models/sophisticated.onnx models/fast_cnn_diarization.onnx")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
