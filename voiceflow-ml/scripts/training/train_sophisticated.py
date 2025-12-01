"""
Quick training script for Transformer-based diarization model.

This script trains the SophisticatedProductionGradeDiarizationModel
using synthetic or real audio data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
import time
from typing import Optional

from models.diarization.model import SophisticatedProductionGradeDiarizationModel


class AudioDiarizationDataset(Dataset):
    """
    Dataset for audio diarization with raw waveforms.
    
    For real training, replace this with actual audio loading from files.
    """
    
    def __init__(self, num_samples: int = 1000, sample_rate: int = 16000, duration: int = 3):
        """
        Args:
            num_samples: Number of training samples
            sample_rate: Audio sample rate (16kHz for Wav2Vec2)
            duration: Audio duration in seconds
        """
        self.num_samples = num_samples
        self.audio_length = sample_rate * duration
        
        # Generate synthetic audio (replace with real data loading)
        print(f"Generating {num_samples} synthetic audio samples...")
        self.audio_data = torch.randn(num_samples, self.audio_length)
        self.labels = torch.randint(0, 2, (num_samples,))  # Binary: speaker 0 or 1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.audio_data[idx], self.labels[idx]


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\nEpoch {epoch + 1}")
    print("-" * 60)
    
    start_time = time.time()
    
    for batch_idx, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 5 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] | "
                  f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    print(f"\n  Epoch {epoch + 1} Summary:")
    print(f"  ├─ Loss: {avg_loss:.4f}")
    print(f"  ├─ Accuracy: {accuracy:.2f}%")
    print(f"  └─ Time: {epoch_time:.2f}s")
    
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
    
    print(f"\n  Validation Results:")
    print(f"  ├─ Loss: {avg_loss:.4f}")
    print(f"  └─ Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    
    # Configuration
    NUM_SPEAKERS = 2
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 0.001
    NUM_TRAIN_SAMPLES = 200  # Small for quick training
    NUM_VAL_SAMPLES = 40
    CHECKPOINT_DIR = Path("../models/checkpoints")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("🚀 Training Sophisticated Production-Grade Diarization Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training samples: {NUM_TRAIN_SAMPLES}")
    print(f"Validation samples: {NUM_VAL_SAMPLES}")
    print(f"{'='*60}\n")
    
    # Create model
    print("Loading model...")
    model = SophisticatedProductionGradeDiarizationModel(
        num_speakers=NUM_SPEAKERS,
        hidden_size=256,
        freeze_encoder=True  # ✅ Frozen encoder for fast training
    )
    model = model.to(device)
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = AudioDiarizationDataset(num_samples=NUM_TRAIN_SAMPLES)
    val_dataset = AudioDiarizationDataset(num_samples=NUM_VAL_SAMPLES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # Loss and optimizer (only train unfrozen parameters!)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    training_start = time.time()
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / "transformer_diarization_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, checkpoint_path)
            print(f"\n  💾 Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - training_start
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR / 'transformer_diarization_best.pth'}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    trained_model = main()
    
    print("\n🎯 Next steps:")
    print("  1. Export to ONNX: python models/diarization/export_onnx.py")
    print("  2. Test with Rust inference engine")
    print("  3. Benchmark latency (<100ms target)")
