# 🎓 The Complete Guide to Embedding-Based Speaker Diarization

**A Mentor's Guide: From Theory to Production**

---

## Table of Contents
1. [The Fundamental Problem](#1-the-fundamental-problem)
2. [Why Classification Fails](#2-why-classification-fails)
3. [The Embedding Approach](#3-the-embedding-approach)
4. [Training with Contrastive Learning](#4-training-with-contrastive-learning)
5. [Inference and Clustering](#5-inference-and-clustering)
6. [Production Architecture](#6-production-architecture)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. The Fundamental Problem

### What is Speaker Diarization?

Speaker diarization answers the question: **"Who spoke when?"** in an audio recording.

```
Input Audio (Meeting Recording):
┌────────────────────────────────────────────────────┐
│ "Hello everyone..." "Thanks John..." "I agree..."  │
│  [Mixed voices, overlapping, different speakers]   │
└────────────────────────────────────────────────────┘
                        ↓
              DIARIZATION SYSTEM
                        ↓
Output (Time-stamped Segments):
┌────────────────────────────────────────────────────┐
│ Speaker_A: 00:00-00:05 "Hello everyone..."         │
│ Speaker_B: 00:05-00:08 "Thanks John..."            │
│ Speaker_A: 00:08-00:12 "I agree..."                │
└────────────────────────────────────────────────────┘
```

### The Core Challenge

**You don't know who the speakers are in advance!**

- Meeting #1: Alice, Bob, Charlie
- Meeting #2: David, Emma
- Meeting #3: Frank, George, Helen, Ian

Each recording has **different people** that the model has **never seen before**.

---

## 2. Why Classification Fails

### ❌ The Classification Approach (What You Were Doing)

```
┌─────────────────────────────────────────────────────┐
│              CLASSIFICATION MODEL                   │
│                                                     │
│  Audio Input                                        │
│      ↓                                              │
│  [Neural Network]                                   │
│      ↓                                              │
│  [Output Layer: 479 neurons]                        │
│      ↓                                              │
│  Softmax → Speaker ID (0-478)                       │
│                                                     │
│  Training: Learn to map audio → speaker class      │
│  Problem: Only works for those 479 speakers!       │
└─────────────────────────────────────────────────────┘
```

**Why This Fails for Diarization:**

1. **Closed World Assumption**: Model only knows 479 speakers from LibriSpeech
2. **No Generalization**: Can't recognize new speakers in production
3. **Wrong Task**: Classification ≠ Clustering/Grouping
4. **Accuracy Ceiling**: ~60% because you're forcing wrong problem formulation

**Example of Failure:**
```python
# Training on LibriSpeech (Speaker IDs: 1-479)
model.predict(audio_from_meeting)  # Returns: Speaker_247

# But in your meeting:
# - None of these people are in LibriSpeech!
# - Model is confused, picks random training speaker
# - Result is meaningless
```

---

## 3. The Embedding Approach

### ✅ The Correct Solution: Learn a Similarity Space

Instead of predicting speaker IDs, learn to map audio to **embedding vectors** where:
- **Similar speakers** → **Close together** in embedding space
- **Different speakers** → **Far apart** in embedding space

```
┌────────────────────────────────────────────────────────┐
│            EMBEDDING EXTRACTION MODEL                  │
│                                                        │
│  Audio Input (3 seconds)                               │
│      ↓                                                 │
│  [Feature Extraction: MFCC/Mel-Spectrogram]           │
│      ↓                                                 │
│  [Neural Network: FastCNN]                             │
│      ↓                                                 │
│  [Embedding Layer: 512 dimensions]                     │
│      ↓                                                 │
│  Embedding Vector: [0.23, -0.45, 0.67, ..., 0.12]    │
│                                                        │
│  Training: Learn to make same speaker → similar        │
│           embeddings, different speakers → different   │
└────────────────────────────────────────────────────────┘
```

### Visual Intuition: Embedding Space

```
2D Visualization (actual is 512-dim):

    Speaker Embeddings in Vector Space
    
    ▲
    │                    ⬤ Speaker C (voice 3)
    │                 ⬤ ⬤ ⬤
    │               ⬤       ⬤
    │
    │    ⬤ ⬤           
    │  ⬤ ⬤ ⬤           ⬤  ⬤
    │    ⬤            ⬤ ⬤ ⬤  Speaker B (voice 2)
    │
    │         ⬤ ⬤ ⬤
    │      ⬤ ⬤ ⬤ ⬤ ⬤  Speaker A (voice 1)
    │        ⬤ ⬤ ⬤
    │
    └────────────────────────────────────►

Key Insight:
- Multiple samples from Speaker A cluster together
- Different speakers form distinct clusters
- New speakers will form their own clusters
- No need to know speaker identities beforehand!
```

### Why Embeddings Work

1. **Open World**: Works with speakers never seen in training
2. **Generalization**: Learns voice characteristics, not memorize IDs
3. **Flexible**: Can handle any number of speakers
4. **Scalable**: Add new speakers without retraining

---

## 4. Training with Contrastive Learning

### The Core Idea: Learn by Comparison

Instead of labels (Speaker_1, Speaker_2, ...), we use **pairs**:

```
Training Sample:
┌──────────────────────────────────────────────────┐
│  Audio_A (Speaker 1, segment 1)                  │
│  Audio_B (Speaker 1, segment 2)                  │
│  Label: SAME SPEAKER ✓                           │
└──────────────────────────────────────────────────┘

Training Sample:
┌──────────────────────────────────────────────────┐
│  Audio_C (Speaker 1, segment 1)                  │
│  Audio_D (Speaker 2, segment 1)                  │
│  Label: DIFFERENT SPEAKERS ✗                     │
└──────────────────────────────────────────────────┘
```

### Contrastive Loss Function

**Goal**: Pull same-speaker embeddings together, push different-speaker embeddings apart.

```
Contrastive Loss Visualization:

Before Training:
    ⬤A₁         ⬤A₂               Random positions
         ⬤B₁
                    ⬤B₂

After Training with Contrastive Loss:
    
    ⬤A₁⬤A₂                        Same speaker: Close
    
              ⬤B₁⬤B₂              Same speaker: Close
              
    [Distance between A and B: Large]  ← Different speakers: Far
```

**Mathematical Formulation:**

```python
def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    """
    Args:
        embedding1, embedding2: 512-dim vectors
        label: 0 (different speaker) or 1 (same speaker)
        margin: How far apart different speakers should be
    
    Returns:
        Loss value to minimize
    """
    distance = euclidean_distance(embedding1, embedding2)
    
    if label == 1:  # Same speaker
        # Pull embeddings closer (minimize distance)
        loss = distance²
    else:  # Different speakers
        # Push embeddings apart (maximize distance up to margin)
        loss = max(0, margin - distance)²
    
    return loss
```

### Training Process Flow

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
└─────────────────────────────────────────────────────────┘

1. Sample Pair Generation:
   ┌─────────────────────────────────────────┐
   │ LibriSpeech Dataset (1000 speakers)     │
   │                                         │
   │ Positive Pairs (same speaker):          │
   │   Speaker_42, Segment_1 + Segment_2     │
   │   Speaker_42, Segment_3 + Segment_5     │
   │                                         │
   │ Negative Pairs (different speakers):    │
   │   Speaker_42, Segment_1 + Speaker_89, Segment_1 │
   │   Speaker_42, Segment_2 + Speaker_101, Segment_3│
   └─────────────────────────────────────────┘
                    ↓
2. Feature Extraction (both samples):
   ┌─────────────────────────────────────────┐
   │ Audio → MFCC (40 coefficients × frames) │
   │ Normalize, augment (pitch, speed)       │
   └─────────────────────────────────────────┘
                    ↓
3. Forward Pass (both through same network):
   ┌─────────────────────────────────────────┐
   │ MFCC → FastCNN → Embedding_1 (512-dim)  │
   │ MFCC → FastCNN → Embedding_2 (512-dim)  │
   └─────────────────────────────────────────┘
                    ↓
4. Compute Loss:
   ┌─────────────────────────────────────────┐
   │ distance = ||Embedding_1 - Embedding_2||│
   │ loss = contrastive_loss(distance, label)│
   └─────────────────────────────────────────┘
                    ↓
5. Backpropagation:
   ┌─────────────────────────────────────────┐
   │ Adjust network weights to:              │
   │ - Decrease distance for same speaker    │
   │ - Increase distance for different       │
   └─────────────────────────────────────────┘
                    ↓
6. Repeat for 50,000+ pairs
```

### Alternative: Triplet Loss (More Powerful)

```
Triplet: (Anchor, Positive, Negative)

┌────────────────────────────────────────────────────┐
│  Anchor:   Audio from Speaker A, segment 1         │
│  Positive: Audio from Speaker A, segment 2  ✓ Same │
│  Negative: Audio from Speaker B, segment 1  ✗ Diff │
└────────────────────────────────────────────────────┘

Goal: Make distance(Anchor, Positive) < distance(Anchor, Negative)

Triplet Loss Formula:
loss = max(0, distance(A,P) - distance(A,N) + margin)

Visual:
       ⬤A (anchor)
      ╱ ╲
     ╱   ╲
    ╱     ╲
   ⬤P      ⬤N
(close)   (far)
```

**Why Triplet Loss is Better:**
- Directly optimizes relative distances
- More stable training
- Better separation between speakers
- Used by Google's FaceNet (face recognition) and speaker verification systems

---

## 5. Inference and Clustering

### The Inference Pipeline

```
┌────────────────────────────────────────────────────────┐
│              REAL-TIME DIARIZATION FLOW                │
└────────────────────────────────────────────────────────┘

Step 1: Sliding Window Segmentation
┌─────────────────────────────────────────────────────┐
│ Audio Stream: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      │
│                                                     │
│ Window 1: [▓▓▓▓] 3 seconds                          │
│ Window 2:   [▓▓▓▓] (1s hop)                         │
│ Window 3:     [▓▓▓▓]                                │
│ Window 4:       [▓▓▓▓]                              │
└─────────────────────────────────────────────────────┘
                        ↓
Step 2: Feature Extraction (per window)
┌─────────────────────────────────────────────────────┐
│ Each 3-second window → MFCC (40 × 300 frames)       │
└─────────────────────────────────────────────────────┘
                        ↓
Step 3: Embedding Extraction (ONNX Model in Rust)
┌─────────────────────────────────────────────────────┐
│ MFCC → FastCNN (ONNX) → Embedding (512-dim)         │
│ Latency: <5ms per window (Rust + ONNX Runtime)      │
└─────────────────────────────────────────────────────┘
                        ↓
Step 4: Clustering (Online Algorithm)
┌─────────────────────────────────────────────────────┐
│ Method: HDBSCAN or Agglomerative Clustering         │
│                                                     │
│ Input: [Emb₁, Emb₂, Emb₃, ..., Embₙ]               │
│ Output: [0, 0, 1, 1, 0, 2, 2, ...]                  │
│         Speaker IDs for each window                 │
└─────────────────────────────────────────────────────┘
                        ↓
Step 5: Temporal Smoothing & Segmentation
┌─────────────────────────────────────────────────────┐
│ Raw:      [0, 0, 1, 0, 0, 1, 1, 2, 2, 2]            │
│ Smoothed: [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]            │
│                                                     │
│ Output Segments:                                    │
│   Speaker_0: 00:00 - 00:05                          │
│   Speaker_1: 00:05 - 00:07                          │
│   Speaker_2: 00:07 - 00:10                          │
└─────────────────────────────────────────────────────┘
```

### Clustering Algorithms Explained

#### Option A: Agglomerative Clustering (Bottom-Up)

```
Algorithm: Start with each embedding as its own cluster, merge closest

Initial State (each point is a cluster):
    ⬤₁  ⬤₂        ⬤₃  ⬤₄
    
            ⬤₅
                ⬤₆  ⬤₇

Step 1: Merge closest pairs
    ⬤₁₂         ⬤₃₄     (Merged 1-2 and 3-4)
    
            ⬤₅
                ⬤₆₇     (Merged 6-7)

Step 2: Continue merging
    ⬤₁₂₃₄                (Merged clusters)
    
            ⬤₅₆₇         (Merged clusters)

Final: Two speakers detected
    [Cluster A: embeddings 1,2,3,4]  → Speaker_0
    [Cluster B: embeddings 5,6,7]    → Speaker_1
```

**Pros:**
- Simple and fast
- Works well for small meetings (2-5 speakers)
- Deterministic results

**Cons:**
- Needs to know number of speakers beforehand
- Can merge too aggressively

#### Option B: HDBSCAN (Density-Based)

```
Algorithm: Find dense regions of embeddings

Embedding Space:
    ⬤ ⬤ ⬤               Dense region → Speaker A
    ⬤ ⬤ ⬤ ⬤
      ⬤ ⬤
                
               ⬤ ⬤      Dense region → Speaker B
              ⬤ ⬤ ⬤
               ⬤
    
    ⬤                   Outlier (ignore or assign)

Output:
    Cluster 0: 8 embeddings (Speaker A)
    Cluster 1: 5 embeddings (Speaker B)
    Noise: 1 embedding (unclear)
```

**Pros:**
- Automatically determines number of speakers
- Robust to outliers
- Handles varying cluster densities

**Cons:**
- More computationally expensive
- Requires parameter tuning (min_cluster_size)

### Online vs Batch Clustering

**Batch Clustering** (entire recording):
```python
# Collect all embeddings first
embeddings = [extract_embedding(window) for window in audio_windows]
# Cluster all at once
labels = clustering.fit_predict(embeddings)
# Output complete diarization
return segments
```

**Online Clustering** (streaming):
```python
# Process as audio arrives
for window in audio_stream:
    embedding = extract_embedding(window)
    
    # Update clusters incrementally
    speaker_id = update_clusters(embedding)
    
    # Output segment immediately
    yield (speaker_id, timestamp)
```

---

## 6. Production Architecture

### Your VoiceFlow System

```
┌──────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM                       │
└──────────────────────────────────────────────────────────┘

┌─────────────────────┐         ┌──────────────────────────┐
│   TRAINING PHASE    │         │    INFERENCE PHASE       │
│   (Python/Colab)    │         │    (Rust Production)     │
└─────────────────────┘         └──────────────────────────┘

1. DATA PREPARATION                1. AUDIO INPUT
   ├─ LibriSpeech                     ├─ WebSocket stream
   ├─ Generate pairs                  └─ HTTP upload
   └─ Augmentation
                                   2. PREPROCESSING (Rust)
2. TRAINING                           ├─ Decode audio
   ├─ FastCNN architecture            ├─ Segment (3s windows)
   ├─ Contrastive loss                └─ Extract MFCC
   └─ 50k pairs, 20 epochs
                                   3. EMBEDDING (ONNX/Rust)
3. VALIDATION                         ├─ Load model
   ├─ Test on unseen speakers         ├─ Inference (<5ms)
   └─ Verify clustering works         └─ Output 512-dim vector

4. EXPORT                          4. CLUSTERING (Rust)
   ├─ PyTorch → ONNX                  ├─ HDBSCAN or Agglo
   ├─ Optimize graph                  └─ Assign speaker IDs
   └─ Quantize (FP32→FP16)
                                   5. OUTPUT
5. DEPLOY                             ├─ Speaker segments
   └─ Copy to /models/                ├─ Timestamps
        ↓                             └─ JSON response
        └─────────────┐
                      ↓
            ┌─────────────────────┐
            │   ONNX Model File   │
            │   (shared storage)  │
            └─────────────────────┘
                      ↑
        ┌─────────────┘
        │
   Loaded by Rust at startup
```

### Performance Requirements Breakdown

```
Target: <100ms end-to-end latency

┌────────────────────────────────────────────────────┐
│  Latency Budget (per 3-second audio window)       │
├────────────────────────────────────────────────────┤
│  1. Network (WebSocket)         : 10-20ms          │
│  2. Audio decode (Rust)         : 5-10ms           │
│  3. MFCC extraction (Rust)      : 10-15ms          │
│  4. ONNX inference (Rust)       : 4-5ms ✓ ACHIEVED │
│  5. Clustering (Rust)           : 20-30ms          │
│  6. Response formatting         : 5-10ms           │
├────────────────────────────────────────────────────┤
│  Total:                         : 54-90ms          │
│                                   ✓ WITHIN TARGET  │
└────────────────────────────────────────────────────┘

Bottleneck: Clustering (if online)
Solution: Batch every 10 windows, cluster together
```

### Why Rust for Inference?

```
Python vs Rust Performance:

Metric              Python (PyTorch)    Rust (ONNX)    Speedup
────────────────────────────────────────────────────────────────
Cold start          2-5 seconds         50-100ms       20-50x
Memory usage        500-800 MB          50-100 MB      5-8x
Inference latency   15-25ms             4-5ms          3-5x
Throughput          50-80 req/s         250-350 req/s  3-5x
Concurrent clients  Limited (GIL)       Thousands      ∞
```

---

## 7. Implementation Roadmap

### Phase 1: Training Pipeline (Python)

**File: `voiceflow-ml/models/diarization/train.py`**

```python
"""
Training pipeline for embedding-based speaker diarization.
"""
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .model import FastCNNDiarization

class ContrastiveDataset(Dataset):
    """
    Generate positive and negative pairs from LibriSpeech.
    
    Positive pair: Two segments from same speaker
    Negative pair: Two segments from different speakers
    
    Args:
        dataset: HuggingFace dataset
        num_pairs: Total pairs to generate (50% positive, 50% negative)
        segment_duration: Audio segment length in seconds
    
    Returns:
        (mfcc1, mfcc2, label) where label is 0 (different) or 1 (same)
    """
    def __init__(
        self, 
        dataset,
        num_pairs: int = 50000,
        segment_duration: float = 3.0
    ):
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.segment_duration = segment_duration
        
        # Build speaker index for efficient sampling
        self.speaker_index = self._build_speaker_index()
    
    def _build_speaker_index(self) -> dict:
        """Map speaker_id → list of audio samples"""
        speaker_index = {}
        for sample in self.dataset:
            speaker_id = sample['speaker_id']
            if speaker_id not in speaker_index:
                speaker_index[speaker_id] = []
            speaker_index[speaker_id].append(sample)
        return speaker_index
    
    def __len__(self) -> int:
        return self.num_pairs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Randomly choose positive (50%) or negative (50%) pair
        is_positive = idx % 2 == 0
        
        if is_positive:
            # Sample two segments from same speaker
            speaker_id = random.choice(list(self.speaker_index.keys()))
            samples = self.speaker_index[speaker_id]
            
            if len(samples) < 2:
                # Fallback if speaker has only one sample
                sample1 = sample2 = samples[0]
            else:
                sample1, sample2 = random.sample(samples, 2)
            
            label = 1  # Same speaker
        else:
            # Sample from two different speakers
            speaker1, speaker2 = random.sample(
                list(self.speaker_index.keys()), 2
            )
            sample1 = random.choice(self.speaker_index[speaker1])
            sample2 = random.choice(self.speaker_index[speaker2])
            
            label = 0  # Different speakers
        
        # Extract MFCC features
        mfcc1 = extract_mfcc(sample1['audio']['array'])
        mfcc2 = extract_mfcc(sample2['audio']['array'])
        
        return mfcc1, mfcc2, label


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for embedding learning.
    
    Pulls together same-speaker embeddings, pushes apart different-speaker.
    
    Args:
        margin: How far apart different speakers should be (default: 1.0)
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embedding1: Batch of embeddings, shape (batch_size, 512)
            embedding2: Batch of embeddings, shape (batch_size, 512)
            label: 0 for different speakers, 1 for same speaker
        
        Returns:
            Loss value (scalar)
        """
        # Euclidean distance
        distance = torch.nn.functional.pairwise_distance(
            embedding1, embedding2
        )
        
        # Contrastive loss formula
        loss_positive = label * distance.pow(2)
        loss_negative = (1 - label) * torch.clamp(
            self.margin - distance, min=0.0
        ).pow(2)
        
        return (loss_positive + loss_negative).mean()


def train_embedding_model(
    train_dataset,
    val_dataset,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """
    Train FastCNN model with contrastive loss.
    
    Args:
        train_dataset: Training dataset (LibriSpeech)
        val_dataset: Validation dataset
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        Trained model
    """
    # Initialize model
    model = FastCNNDiarization(
        input_channels=40,  # MFCC features
        embedding_dim=512
    ).to(device)
    
    # Create datasets
    train_pairs = ContrastiveDataset(train_dataset, num_pairs=50000)
    val_pairs = ContrastiveDataset(val_dataset, num_pairs=5000)
    
    # Dataloaders
    train_loader = DataLoader(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=batch_size,
        num_workers=2
    )
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for mfcc1, mfcc2, labels in train_loader:
            mfcc1 = mfcc1.to(device)
            mfcc2 = mfcc2.to(device)
            labels = labels.to(device)
            
            # Forward pass (both samples through same network)
            emb1 = model(mfcc1)
            emb2 = model(mfcc2)
            
            # Compute loss
            loss = criterion(emb1, emb2, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2%}")
        
        # Save checkpoint
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            best_acc = val_acc
    
    return model


def validate(model, val_loader, criterion, device):
    """
    Validate embedding quality.
    
    Accuracy metric: 
    - For same-speaker pairs: distance < threshold → correct
    - For different-speaker pairs: distance > threshold → correct
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    threshold = 0.5  # Distance threshold
    
    with torch.no_grad():
        for mfcc1, mfcc2, labels in val_loader:
            mfcc1 = mfcc1.to(device)
            mfcc2 = mfcc2.to(device)
            labels = labels.to(device)
            
            emb1 = model(mfcc1)
            emb2 = model(mfcc2)
            
            loss = criterion(emb1, emb2, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            distances = torch.nn.functional.pairwise_distance(emb1, emb2)
            predictions = (distances < threshold).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return val_loss / len(val_loader), correct / total
```

### Phase 2: Inference with Clustering (Rust)

**File: `voiceflow-inference/src/streaming/clustering.rs`**

```rust
/// Online speaker clustering for real-time diarization.
///
/// Maintains running clusters of speaker embeddings and assigns
/// new embeddings to existing clusters or creates new ones.

use ndarray::{Array1, Array2};
use hdbscan::Hdbscan;

pub struct OnlineClusterer {
    /// All embeddings collected so far
    embeddings: Vec<Array1<f32>>,
    
    /// Speaker ID assignments for each embedding
    speaker_ids: Vec<usize>,
    
    /// Minimum cluster size (adjust based on requirements)
    min_cluster_size: usize,
    
    /// Distance threshold for assigning to existing cluster
    distance_threshold: f32,
}

impl OnlineClusterer {
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            speaker_ids: Vec::new(),
            min_cluster_size: 5,
            distance_threshold: 0.6,
        }
    }
    
    /// Add new embedding and return speaker ID
    pub fn add_embedding(&mut self, embedding: Array1<f32>) -> usize {
        // Option 1: Assign to nearest existing cluster
        if let Some(speaker_id) = self.assign_to_nearest_cluster(&embedding) {
            self.embeddings.push(embedding);
            self.speaker_ids.push(speaker_id);
            return speaker_id;
        }
        
        // Option 2: Create new cluster if can't assign confidently
        let new_speaker_id = self.get_next_speaker_id();
        self.embeddings.push(embedding);
        self.speaker_ids.push(new_speaker_id);
        
        // Periodically re-cluster to improve accuracy
        if self.embeddings.len() % 50 == 0 {
            self.recluster();
        }
        
        new_speaker_id
    }
    
    fn assign_to_nearest_cluster(&self, embedding: &Array1<f32>) -> Option<usize> {
        if self.embeddings.is_empty() {
            return None;
        }
        
        // Find closest existing embedding
        let (min_distance, closest_idx) = self.embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| (euclidean_distance(embedding, emb), idx))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();
        
        // Only assign if distance is below threshold
        if min_distance < self.distance_threshold {
            Some(self.speaker_ids[closest_idx])
        } else {
            None
        }
    }
    
    fn recluster(&mut self) {
        // Run HDBSCAN on all embeddings to improve clustering
        // This is expensive but only done periodically
        
        info!("Re-clustering {} embeddings", self.embeddings.len());
        
        // Convert to 2D array for HDBSCAN
        let data: Array2<f32> = Array2::from_shape_vec(
            (self.embeddings.len(), 512),
            self.embeddings.iter().flat_map(|e| e.iter().cloned()).collect()
        ).unwrap();
        
        // Run clustering
        let clusterer = Hdbscan::new(self.min_cluster_size);
        let labels = clusterer.fit_predict(&data);
        
        // Update speaker IDs
        self.speaker_ids = labels;
    }
    
    fn get_next_speaker_id(&self) -> usize {
        self.speaker_ids.iter().max().map(|&x| x + 1).unwrap_or(0)
    }
}

fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
```

---

## Summary: The Complete Mental Model

```
┌─────────────────────────────────────────────────────────┐
│  FROM AUDIO TO "WHO SPOKE WHEN" - THE COMPLETE PICTURE │
└─────────────────────────────────────────────────────────┘

1. TRAINING (Learn voice characteristics, not identities)
   Audio Pairs → Features → Neural Net → Embeddings
                                    ↓
                            Contrastive Loss
                                    ↓
                    "Same speaker? → Close together"
                    "Diff speaker? → Far apart"

2. INFERENCE (Apply to new, unseen speakers)
   Audio Stream → Segments → Features → Trained Net
                                    ↓
                               Embeddings (512-dim)
                                    ↓
                               Clustering
                                    ↓
                          Speaker IDs (0, 1, 2, ...)
                                    ↓
                          Time-aligned Segments

3. PRODUCTION (Fast, scalable, <100ms)
   Python: Training, export ONNX
   Rust: Real-time inference, clustering
   ONNX: 4.48ms model latency
   
KEY INSIGHT: We don't predict who someone is.
            We learn how to measure if two voices are similar.
            Then we group similar voices together.
            This works for anyone, anytime!
```

---

## Next Steps

Now that you understand the theory, let's implement:

1. ✅ **This document** - Theory mastered
2. 🔨 **Training script** - `train.py` with contrastive loss
3. 🔨 **Colab notebook** - User-friendly training interface
4. 🔨 **Rust clustering** - Online diarization engine
5. 🔨 **Integration** - End-to-end pipeline
6. ✅ **Deploy** - Production-ready system

You now have the complete mental framework. Let's build it! 🚀
