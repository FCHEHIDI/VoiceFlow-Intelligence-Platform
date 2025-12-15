# Cloud-Native Training Strategy: Zero Local Storage

## 🚨 Problem: VoxCeleb2 = 150GB Download

**Challenge**: We need real audio data for training, but:
- VoxCeleb2: ~150GB compressed, ~200GB uncompressed
- AMI Corpus: ~50GB
- LibriSpeech: ~60GB

**Local storage limitations**:
❌ Laptop/desktop: May not have 200GB free  
❌ Download time: 6-12 hours on typical internet  
❌ Preprocessing: Needs additional 100GB for processed files  
❌ Total impact: ~300GB local storage required  

---

## ✅ Solution: Cloud-Native Streaming Training

**Zero local storage** - Stream audio directly from cloud sources during training!

---

## 🎯 Recommended Approach: HuggingFace Datasets + Google Colab

### Strategy Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   HuggingFace Hub (Cloud)                   │
│                                                             │
│  VoxCeleb Dataset (Streaming) ─────────────────────┐       │
│  • No download required                            │       │
│  • Stream audio on-demand                          │       │
│  • Automatic caching of recent samples             │       │
└────────────────────────────────────────────────────┼───────┘
                                                     │
                                                     │ Stream
                                                     ↓
┌─────────────────────────────────────────────────────────────┐
│             Google Colab (Free T4 GPU)                      │
│                                                             │
│  • 15GB RAM (sufficient for batches)                       │
│  • 100GB disk (temporary cache only)                       │
│  • Free T4 GPU (12 hours/session)                          │
│  • No local storage used!                                  │
│                                                             │
│  Training Loop:                                             │
│  1. Fetch batch from HuggingFace (streaming)               │
│  2. Preprocess audio in-memory                             │
│  3. Train model on batch                                   │
│  4. Save checkpoint to Google Drive                        │
│  5. Repeat (no local data stored)                          │
└─────────────────────────────────────────────────────────────┘
                                                     │
                                                     │ Save checkpoints
                                                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Google Drive (Cloud Storage)                   │
│                                                             │
│  • Model checkpoints: ~20MB each                           │
│  • Training logs: <1MB                                     │
│  • Total storage: <500MB for entire training               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset Options (All Streaming-Compatible)

### Option 1: VoxCeleb via HuggingFace 🏆 RECOMMENDED

**Dataset**: `jonatasgrosman/voxceleb_test` or similar on HuggingFace

**Pros**:
✅ Streaming API - **zero download**  
✅ Automatic audio loading (librosa/torchaudio)  
✅ Pre-split train/val/test  
✅ Well-documented  
✅ Free access (no signup required)  

**Cons**:
⚠️ Requires stable internet during training  
⚠️ First epoch slower (caching builds up)  

**Implementation**:
```python
from datasets import load_dataset

# Streaming mode - NO download!
dataset = load_dataset(
    "jonatasgrosman/voxceleb_test",  # or custom VoxCeleb subset
    split="train",
    streaming=True  # 🔑 Key parameter
)

# Iterate without storing locally
for sample in dataset:
    audio = sample['audio']['array']  # NumPy array, in-memory only
    label = sample['label']
    # Train immediately, then discard
```

**Storage Impact**: ~2-5GB cache (Colab temp storage), auto-managed

---

### Option 2: Google Cloud Storage + Colab

**Dataset**: VoxCeleb2 uploaded to GCS bucket

**Setup** (one-time):
```bash
# On a machine with storage, download VoxCeleb2
wget http://www.robots.ox.ac.uk/~vg/data/voxceleb/vox2.zip

# Upload to Google Cloud Storage
gsutil -m cp -r voxceleb2/* gs://your-bucket/voxceleb2/
```

**Colab Training**:
```python
from google.colab import auth
import gcsfs

# Authenticate
auth.authenticate_user()

# Access GCS directly (streaming)
fs = gcsfs.GCSFileSystem(project='your-project')

# Stream audio files
for file_path in fs.ls('your-bucket/voxceleb2/train'):
    with fs.open(file_path, 'rb') as f:
        audio, sr = torchaudio.load(f)
        # Train immediately
```

**Storage Impact**: 
- GCS: ~150GB (one-time cost: $3/month)
- Colab: ~2GB cache
- Local: **0GB** ✅

---

### Option 3: LibriSpeech (Smaller, Good for Prototyping)

**Dataset**: `librispeech_asr` on HuggingFace

**Specs**:
- Size: 60GB (more manageable)
- Streaming: ✅ Fully supported
- Quality: Clean audiobooks

**Ideal for**:
- Quick prototyping
- Testing training pipeline
- Lower internet bandwidth requirements

**Implementation**:
```python
from datasets import load_dataset

# Streaming LibriSpeech
dataset = load_dataset(
    "librispeech_asr",
    "clean",  # or "other" for more variety
    split="train.360",  # 360 hours
    streaming=True
)
```

---

### Option 4: CommonVoice (Multi-Language, Free)

**Dataset**: Mozilla CommonVoice on HuggingFace

**Specs**:
- Size: Varies by language (English ~30GB)
- Speakers: 50k+ contributors
- Streaming: ✅ Supported
- License: CC0 (fully open)

**Implementation**:
```python
dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="train",
    streaming=True,
    use_auth_token=True  # Requires HF account (free)
)
```

---

## 🚀 Recommended Training Environment: Google Colab

### Why Google Colab?

| Feature | Colab Free | Colab Pro | Local Machine |
|---------|-----------|-----------|---------------|
| **GPU** | T4 (12h/session) | A100/V100 (24h) | None (CPU only) |
| **RAM** | 12-15GB | 25-50GB | Varies |
| **Storage** | 100GB temp | 200GB temp | Must download |
| **Cost** | $0 ✅ | $10/month | $0 (but slow) |
| **Streaming** | ✅ Perfect | ✅ Perfect | ❌ (need local data) |
| **Setup Time** | 2 minutes | 2 minutes | Hours (download) |

**Verdict**: Colab Free is **perfect** for this project!

---

## 📝 Complete Training Setup (Zero Local Storage)

### Step 1: Create Colab Training Notebook

```python
# Cell 1: Mount Google Drive (for checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory
!mkdir -p /content/drive/MyDrive/voiceflow_checkpoints

# Cell 2: Install dependencies
!pip install -q datasets torch torchaudio transformers accelerate

# Cell 3: Load streaming dataset
from datasets import load_dataset

print("Loading VoxCeleb (streaming mode - no download)...")
train_dataset = load_dataset(
    "jonatasgrosman/voxceleb_test",
    split="train",
    streaming=True
)

val_dataset = load_dataset(
    "jonatasgrosman/voxceleb_test",
    split="validation",
    streaming=True
)

print("✅ Datasets ready (zero local storage used)")

# Cell 4: Clone your model code
!git clone https://github.com/FCHEHIDI/VoiceFlow-Intelligence-Platform.git
%cd VoiceFlow-Intelligence-Platform/voiceflow-ml

# Cell 5: Import model
import sys
sys.path.append('/content/VoiceFlow-Intelligence-Platform/voiceflow-ml')

from models.diarization.model import FastDiarizationModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FastDiarizationModel(
    num_speakers=2,
    hidden_size=256,
    encoder_type="lightweight-cnn"
).to(device)

print(f"Model loaded on {device}")
print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

# Cell 6: Training loop with streaming
from torch.utils.data import DataLoader
import numpy as np

def collate_streaming_batch(batch):
    """Process streaming batch."""
    audios = []
    labels = []
    
    for sample in batch:
        # Extract audio (already loaded by HuggingFace)
        audio = torch.FloatTensor(sample['audio']['array'])
        
        # Resample to 16kHz if needed
        if sample['audio']['sampling_rate'] != 16000:
            resampler = torchaudio.transforms.Resample(
                sample['audio']['sampling_rate'], 16000
            )
            audio = resampler(audio)
        
        # Pad/crop to fixed length (3 seconds)
        target_length = 16000 * 3
        if audio.shape[0] > target_length:
            audio = audio[:target_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[0]))
        
        audios.append(audio)
        labels.append(sample['label'])
    
    return torch.stack(audios), torch.LongTensor(labels)

# Create streaming dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=collate_streaming_batch,
    num_workers=2  # Parallel audio loading
)

# Cell 7: Training loop
import torch.optim as optim
from tqdm import tqdm

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 30
checkpoint_dir = "/content/drive/MyDrive/voiceflow_checkpoints"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (audio, labels) in pbar:
        # Move to GPU
        audio = audio.to(device)
        labels = labels.to(device)
        
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
        
        # Update progress
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100 * correct / total
        })
        
        # Save checkpoint every 100 batches
        if (batch_idx + 1) % 100 == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth"
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
    
    # Epoch summary
    print(f"\nEpoch {epoch+1} | Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100*correct/total:.2f}%")
    
    # Save epoch checkpoint
    epoch_checkpoint = f"{checkpoint_dir}/checkpoint_epoch{epoch+1}_final.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, epoch_checkpoint)
    print(f"✅ Checkpoint saved: {epoch_checkpoint}")

print("\n🎉 Training complete!")
```

---

## 💾 Storage Requirements Comparison

| Approach | Local Storage | Cloud Storage | Cost | Setup Time |
|----------|--------------|---------------|------|------------|
| **Download Full Dataset** | 300GB | 0GB | $0 | 12+ hours |
| **Streaming (Colab)** | 0GB ✅ | 2-5GB cache | $0 | 5 minutes |
| **GCS + Colab** | 0GB ✅ | 150GB GCS | $3/mo | 1 hour |
| **Colab Pro + Streaming** | 0GB ✅ | 5GB cache | $10/mo | 5 minutes |

**Winner**: Streaming with Colab Free (zero local storage, zero cost)

---

## 🎯 Recommended Workflow

### Phase 1: Quick Validation (This Week)
```bash
# Objective: Validate training pipeline works
# Dataset: LibriSpeech (smaller, faster iteration)
# Environment: Google Colab Free
# Time: 2-3 hours for 5 epochs
# Cost: $0
```

**Steps**:
1. Create Colab notebook (provided above)
2. Stream LibriSpeech dataset (60GB, no download)
3. Train for 5 epochs (~1-2 hours on T4)
4. Validate accuracy >80% on validation set
5. Save checkpoint to Google Drive

**Deliverable**: Proof that streaming training works

---

### Phase 2: Full Training (Next Week)
```bash
# Objective: Train production model
# Dataset: VoxCeleb2 via HuggingFace streaming
# Environment: Google Colab Free (multiple sessions if needed)
# Time: 24-48 hours total (split across sessions)
# Cost: $0 (or $10 for Colab Pro for uninterrupted training)
```

**Steps**:
1. Switch to VoxCeleb streaming dataset
2. Train for 30 epochs
3. Implement checkpoint resume (for session timeouts)
4. Validate DER <15% on test set
5. Export to ONNX

**Deliverable**: Production-ready model

---

## 🔧 Checkpoint Resume Strategy

**Problem**: Colab Free disconnects after 12 hours

**Solution**: Automatic checkpoint resume

```python
# At start of training
checkpoint_dir = "/content/drive/MyDrive/voiceflow_checkpoints"
latest_checkpoint = None

# Find latest checkpoint
import glob
checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_epoch*.pth")
if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Resuming from: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("Starting from scratch")
    start_epoch = 0

# Continue training from start_epoch
for epoch in range(start_epoch, num_epochs):
    # ... training loop
```

---

## 📊 Expected Training Timeline

### Colab Free (T4 GPU)

| Phase | Duration | Epochs | Storage | Cost |
|-------|----------|--------|---------|------|
| **Setup** | 5 min | - | 0GB | $0 |
| **Epoch 1-5** | 2 hours | 5 | 50MB checkpoints | $0 |
| **Epoch 6-15** | 4 hours | 10 | 150MB checkpoints | $0 |
| **Epoch 16-30** | 6 hours | 15 | 300MB checkpoints | $0 |
| **Total** | ~12 hours | 30 | 500MB | $0 |

**Note**: May need 2-3 Colab sessions due to 12h limit

---

### Colab Pro (A100 GPU) - Optional

| Phase | Duration | Epochs | Storage | Cost |
|-------|----------|--------|---------|------|
| **Full Training** | 6-8 hours | 30 | 500MB | $10 |

**Advantage**: Single uninterrupted session

---

## ✅ Action Plan: Zero Local Storage Training

### This Week (Setup + Validation)
- [ ] Create Google Colab notebook
- [ ] Mount Google Drive for checkpoints
- [ ] Test streaming LibriSpeech dataset
- [ ] Train 5 epochs (validation run)
- [ ] Verify accuracy >80%

**Time**: 3-4 hours  
**Storage**: 0GB local ✅

---

### Next Week (Full Training)
- [ ] Switch to VoxCeleb streaming
- [ ] Train 30 epochs (may span 2-3 Colab sessions)
- [ ] Implement checkpoint resume
- [ ] Evaluate DER on test set
- [ ] Export ONNX model

**Time**: 24-48 hours (mostly automated)  
**Storage**: 0GB local ✅

---

## 🎯 Dataset Decision Matrix

| Dataset | Streaming? | Size | Quality | Setup Time | Recommended For |
|---------|-----------|------|---------|------------|-----------------|
| **VoxCeleb2** | ✅ | 150GB | Excellent | 5 min | Production training |
| **LibriSpeech** | ✅ | 60GB | Good | 5 min | Quick validation |
| **CommonVoice** | ✅ | 30GB | Good | 10 min (HF token) | Multi-language |
| **AMI Corpus** | ⚠️ Manual | 50GB | Excellent (meetings) | 30 min | Fine-tuning |

**Recommendation**:
1. **Start with LibriSpeech** (validate pipeline, 5 epochs)
2. **Train on VoxCeleb2** (production model, 30 epochs)
3. **Fine-tune on AMI** (optional, meeting-specific)

---

## 🚀 Next Steps

I'll now create:
1. **Colab training notebook** with streaming dataset
2. **Enhanced model architecture** (attention + GRU)
3. **Training script** with checkpoint management

**Zero local storage required!** ✅

Ready to proceed with creating the streaming training notebook?
