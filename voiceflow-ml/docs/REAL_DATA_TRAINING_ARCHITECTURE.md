# Real Data Training Architecture for Speaker Diarization

## 🎯 Goal: Production-Grade Speaker Diarization with <100ms P99 Latency

This document outlines the complete architecture and training strategy to train a **high-accuracy speaker diarization model** using real audio data while maintaining the fast inference speed validated on GPU (2-4ms P99).

---

## 📊 Current Status vs Target

| Metric | Current State | Target | Status |
|--------|--------------|---------|--------|
| **Inference Speed** | 2-4ms P99 (GPU T4) ✅ | <10ms P99 | ✅ Achieved |
| **Model Size** | 9.3 KB (Fast CNN) ✅ | <50 MB | ✅ Achieved |
| **Training Data** | Synthetic random noise ❌ | Real speaker audio | ⚠️ **CRITICAL** |
| **Accuracy (DER)** | Untested ❌ | <15% DER | ⚠️ **CRITICAL** |
| **E2E Latency** | 403ms P99 (CPU) | <100ms P99 | ⏳ Pending GPU |

**Key Insight**: We've validated that the **Fast CNN architecture achieves ultra-low inference latency**. Now we need to train it on **real speaker audio** to achieve production-grade accuracy.

---

## 🏗️ Proposed Architecture: Enhanced Fast CNN with Temporal Modeling

### Current Architecture (Baseline)
```
Input: [batch, audio_length] (raw waveform @ 16kHz)
    ↓
LightweightCNNEncoder (6 conv layers)
    ├─ Conv1D(1→64) + BatchNorm + ReLU + MaxPool
    ├─ Conv1D(64→128) + BatchNorm + ReLU + MaxPool
    ├─ Conv1D(128→256) + BatchNorm + ReLU + MaxPool
    ├─ Conv1D(256→512) + BatchNorm + ReLU + MaxPool
    ├─ Conv1D(512→512) + BatchNorm + ReLU + MaxPool
    └─ Conv1D(512→512) + BatchNorm + ReLU + MaxPool
    ↓
Global Average Pooling → [batch, 512]
    ↓
Linear(512 → 256) → [batch, 256]
    ↓
MLP Classifier (256 → 128 → num_speakers)
    ↓
Output: [batch, num_speakers] (speaker logits)
```

**Parameters**: 2.3M  
**Pros**: Ultra-fast (2-4ms), small model size  
**Cons**: No temporal modeling, limited capacity for speaker nuances

---

### Enhanced Architecture (Production Target)

```
Input: [batch, audio_length] (raw waveform @ 16kHz, 3-5 seconds)
    ↓
┌──────────────────────────────────────────────────────┐
│ STAGE 1: Feature Extraction (Improved CNN)          │
└──────────────────────────────────────────────────────┘
    ↓
Enhanced CNN Encoder (8 layers, residual connections)
    ├─ Conv1D(1→64, k=7) + BatchNorm + ReLU + MaxPool(4)
    ├─ ResBlock(64→128)   ↓ downsample: stride=2
    ├─ ResBlock(128→128)  ← residual
    ├─ ResBlock(128→256)  ↓ downsample
    ├─ ResBlock(256→256)  ← residual
    ├─ ResBlock(256→512)  ↓ downsample
    ├─ ResBlock(512→512)  ← residual
    └─ ResBlock(512→512)  ← residual
    ↓
Output: [batch, time_steps, 512] (frame-level features)

┌──────────────────────────────────────────────────────┐
│ STAGE 2: Temporal Modeling (NEW)                     │
└──────────────────────────────────────────────────────┘
    ↓
Multi-Head Self-Attention (4 heads)
    • Captures long-range speaker dependencies
    • Learns: "same speaker" across time
    • Params: ~0.5M
    ↓
[batch, time_steps, 512]
    ↓
Bidirectional GRU (1 layer, hidden=256)
    • Sequential speaker modeling
    • Smooths temporal predictions
    • Params: ~0.8M
    ↓
[batch, time_steps, 512] (bidirectional → 256*2)

┌──────────────────────────────────────────────────────┐
│ STAGE 3: Aggregation & Classification                │
└──────────────────────────────────────────────────────┘
    ↓
Temporal Attention Pooling (NEW)
    • Learns to weight important frames
    • Better than simple average pooling
    • Params: ~0.05M
    ↓
[batch, 512]
    ↓
Classification Head
    ├─ Linear(512 → 256) + ReLU + Dropout(0.3)
    ├─ Linear(256 → 128) + ReLU + Dropout(0.3)
    └─ Linear(128 → num_speakers)
    ↓
Output: [batch, num_speakers] (speaker probabilities)
```

**Enhanced Architecture Stats**:
- **Total Parameters**: ~4.8M (2x larger, still 20x smaller than Wav2Vec2)
- **Expected GPU Inference**: 3-6ms P99 (slight increase, still <10ms ✅)
- **Expected CPU Inference**: 8-15ms P99 (still fast enough for many use cases)
- **Model Size**: ~20 MB ONNX (FP16 quantized)
- **Key Improvements**:
  1. ✅ Residual connections → better gradient flow
  2. ✅ Self-attention → captures speaker context
  3. ✅ GRU temporal modeling → smooths predictions
  4. ✅ Attention pooling → focuses on discriminative frames

---

## 📦 Dataset Strategy

### Option 1: VoxCeleb2 (Recommended for Speaker Recognition)
**Dataset**: https://www.robots.ox.ac.uk/~vg/data/voxceleb/

**Specs**:
- **Size**: 5,994 speakers, 1,128,246 utterances
- **Duration**: ~2,000 hours
- **Quality**: YouTube celebrity interviews, high variability
- **Format**: m4a (convert to 16kHz mono WAV)
- **License**: Academic/research use

**Preparation**:
```bash
# Download VoxCeleb2
python scripts/data/download_voxceleb.py --output data/voxceleb2

# Convert to training format
python scripts/data/prepare_voxceleb.py \
    --input data/voxceleb2 \
    --output data/processed/voxceleb2 \
    --sample-rate 16000 \
    --duration 3-5  # Extract 3-5 second clips
```

**Training Splits**:
- Train: 80% (~5,000 speakers, 900k clips)
- Validation: 10% (~500 speakers, 110k clips)
- Test: 10% (~500 speakers, 110k clips)

---

### Option 2: AMI Meeting Corpus (Real Meetings, Best for Diarization)
**Dataset**: https://groups.inf.ed.ac.uk/ami/corpus/

**Specs**:
- **Size**: 100 hours of meeting recordings
- **Speakers**: 4-person meetings
- **Quality**: High-quality diarization annotations
- **Format**: WAV files with RTTM speaker timestamps
- **License**: Free for research

**Why AMI is IDEAL**:
✅ Real meeting scenarios (target use case)  
✅ Multiple speakers per recording  
✅ Overlapping speech annotations  
✅ Ground-truth speaker boundaries  

**Preparation**:
```bash
# Download AMI
python scripts/data/download_ami.py --output data/ami

# Extract speaker segments
python scripts/data/prepare_ami.py \
    --input data/ami \
    --output data/processed/ami \
    --segment-duration 3-5 \
    --overlap 0.5
```

**Training Format**:
Each sample = (audio_segment, speaker_id, meeting_id)

---

### Option 3: LibriSpeech + Synthetic Mixing (Fast Prototyping)
**Dataset**: https://www.openslr.org/12

**Specs**:
- **Size**: 1,000 hours of read English speech
- **Speakers**: 2,484 speakers
- **Quality**: Clean audiobooks
- **Format**: FLAC (16kHz)

**Strategy**: Create synthetic multi-speaker segments
```python
# Mix 2-3 speakers randomly
def create_mixed_segment(speaker1_audio, speaker2_audio):
    # Random SNR between speakers
    snr = random.uniform(-5, 5)  # dB
    mixed = speaker1_audio + speaker2_audio * (10 ** (snr / 20))
    
    # Label: dominant speaker
    label = 0 if snr > 0 else 1
    return mixed, label
```

**Pros**: Fast to set up, large speaker diversity  
**Cons**: Not realistic multi-speaker dynamics  

---

## 🎓 Training Strategy

### Phase 1: Initial Training (Baseline)
**Objective**: Validate that model can learn speaker features from real audio

**Configuration**:
```python
config = TrainingConfig(
    # Model
    model_type="fast_cnn_enhanced",
    hidden_size=256,
    num_layers=8,
    use_attention=True,
    use_gru=True,
    
    # Data
    dataset="voxceleb2",  # or "ami"
    batch_size=32,
    audio_duration=3.0,  # seconds
    sample_rate=16000,
    
    # Training
    epochs=30,
    learning_rate=1e-3,
    optimizer="adamw",
    weight_decay=1e-4,
    lr_scheduler="cosine",
    
    # Augmentation (critical for robustness)
    augmentations=[
        "time_mask",     # Mask random time segments
        "freq_mask",     # Frequency masking (if using spectrograms)
        "add_noise",     # Background noise (SNR 10-30 dB)
        "pitch_shift",   # ±2 semitones
        "time_stretch",  # 0.9-1.1x speed
    ],
    
    # Validation
    val_every_n_epochs=1,
    save_best_model=True,
    early_stopping_patience=5,
)
```

**Expected Results**:
- Epoch 1-5: Loss drops rapidly (model learns basic patterns)
- Epoch 10-20: Validation accuracy plateaus
- Epoch 20-30: Fine-tuning, small improvements
- **Target**: >85% validation accuracy on VoxCeleb2

**Training Time**:
- GPU: ~12-24 hours (V100/A100)
- Cost: $20-$50 on cloud GPU

---

### Phase 2: Optimization & Distillation (Optional)
**Objective**: Improve accuracy further without sacrificing speed

**Strategy 1: Knowledge Distillation**
Train enhanced model to mimic a larger "teacher" model:

```python
# Teacher: Wav2Vec2-based model (high accuracy)
teacher = SophisticatedProductionGradeDiarizationModel(...)
teacher.load_state_dict(torch.load("teacher_checkpoint.pth"))
teacher.eval()

# Student: Fast CNN (our deployment model)
student = FastDiarizationModelEnhanced(...)

# Distillation loss
def distillation_loss(student_logits, teacher_logits, true_labels, T=3.0):
    # KL divergence between teacher and student
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    
    # Standard cross-entropy
    ce_loss = F.cross_entropy(student_logits, true_labels)
    
    # Combine
    return 0.7 * kl_loss + 0.3 * ce_loss
```

**Expected Gain**: +2-5% accuracy without speed impact

---

**Strategy 2: Multi-Task Learning**
Train model on multiple related tasks:

```python
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, num_speakers, num_emotions=6):
        super().__init__()
        self.speaker_head = nn.Linear(input_dim, num_speakers)
        self.emotion_head = nn.Linear(input_dim, num_emotions)  # Auxiliary task
        
    def forward(self, features):
        speaker_logits = self.speaker_head(features)
        emotion_logits = self.emotion_head(features)  # Improves feature quality
        return speaker_logits, emotion_logits
```

**Auxiliary Tasks**:
- Emotion recognition (happy, sad, angry, neutral, etc.)
- Gender classification
- Age estimation
- Language identification

**Expected Gain**: +1-3% accuracy (richer learned features)

---

### Phase 3: Production Validation
**Objective**: Verify model meets <15% DER on real test data

**Evaluation Metrics**:

1. **DER (Diarization Error Rate)** - Primary metric
   ```
   DER = (False Alarm + Missed Speech + Speaker Error) / Total Speech Time
   
   Target: <15% DER on AMI test set
   Best-in-class: 5-8% DER (pyannote.audio)
   ```

2. **Accuracy** - Secondary metric
   ```
   Accuracy = Correct Speaker Predictions / Total Predictions
   
   Target: >85% on VoxCeleb2 test set
   ```

3. **Inference Latency** - Already validated
   ```
   P99 Latency: 2-4ms (GPU T4) ✅
   Target: <10ms
   ```

4. **F1 Score** - Per-speaker performance
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   
   Target: >0.85 macro-averaged F1
   ```

---

## 🔬 Evaluation Pipeline

### Test Script: `scripts/evaluate_diarization.py`

```python
def evaluate_diarization(model, test_loader, device):
    """
    Comprehensive evaluation on test set.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for audio, labels in tqdm(test_loader):
            audio = audio.to(device)
            
            # Measure inference time
            start = time.perf_counter()
            logits = model(audio)
            inference_time = (time.perf_counter() - start) * 1000 / len(audio)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(preds)
            all_labels.extend(labels.numpy())
            inference_times.append(inference_time)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Inference stats
    p50 = np.percentile(inference_times, 50)
    p99 = np.percentile(inference_times, 99)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:       {accuracy*100:.2f}%")
    print(f"F1 Score:       {f1:.3f}")
    print(f"P50 Latency:    {p50:.2f} ms")
    print(f"P99 Latency:    {p99:.2f} ms")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'p50_ms': p50,
        'p99_ms': p99,
        'confusion_matrix': cm.tolist()
    }
```

---

## 🚀 Implementation Roadmap

### Week 1: Data Preparation
- [ ] Download VoxCeleb2 or AMI corpus (2-3 days)
- [ ] Create data preprocessing pipeline
- [ ] Implement data loader with augmentations
- [ ] Validate: Load 100 samples, verify audio quality

**Deliverable**: `scripts/data/dataset.py` with `VoxCelebDataset` class

---

### Week 2: Enhanced Model Implementation
- [ ] Implement residual CNN blocks
- [ ] Add multi-head self-attention layer
- [ ] Add bidirectional GRU
- [ ] Add attention pooling
- [ ] Test forward pass with dummy data

**Deliverable**: `models/diarization/enhanced_model.py` (4.8M params)

---

### Week 3-4: Training
- [ ] Train baseline model (30 epochs)
- [ ] Monitor validation accuracy
- [ ] Implement early stopping
- [ ] Save checkpoints every 5 epochs
- [ ] Visualize training curves

**Deliverable**: `checkpoints/fast_cnn_enhanced_best.pth`

---

### Week 5: Optimization
- [ ] Export to ONNX (FP16 quantization)
- [ ] Benchmark on GPU (verify <10ms P99)
- [ ] Benchmark on CPU (measure regression)
- [ ] Run full evaluation on test set
- [ ] Calculate DER metric

**Deliverable**: `fast_cnn_enhanced_optimized.onnx` + evaluation report

---

### Week 6: Production Validation
- [ ] Deploy to Rust inference server
- [ ] End-to-end latency testing (audio → inference → response)
- [ ] Load testing (concurrent requests)
- [ ] Error analysis (failure cases)
- [ ] Documentation

**Deliverable**: Production-ready model with <100ms E2E P99 latency

---

## 📈 Success Criteria

| Metric | Target | Stretch Goal | Status |
|--------|--------|--------------|--------|
| **Accuracy** | >85% | >90% | ⏳ Pending training |
| **DER** | <15% | <10% | ⏳ Pending training |
| **F1 Score** | >0.85 | >0.90 | ⏳ Pending training |
| **GPU P99 Latency** | <10ms | <5ms | ✅ 2-4ms achieved |
| **E2E P99 Latency** | <100ms | <50ms | ⏳ Pending GPU deploy |
| **Model Size** | <50 MB | <20 MB | ✅ 9.3 KB (current) |
| **Training Time** | <48h | <24h | ⏳ Pending training |
| **Training Cost** | <$100 | <$50 | ⏳ Pending training |

---

## 💡 Key Architectural Decisions

### Why Enhanced CNN over Wav2Vec2?

| Aspect | Wav2Vec2-base | Enhanced Fast CNN | Winner |
|--------|---------------|-------------------|--------|
| **Inference Speed** | 20-50ms (GPU), 220ms (CPU) | 3-6ms (GPU), 8-15ms (CPU) | 🏆 Fast CNN |
| **Model Size** | 362 MB | 20 MB | 🏆 Fast CNN |
| **Accuracy** | 92-95% (proven) | 85-90% (target) | 🏆 Wav2Vec2 |
| **Training Time** | 3-5 days | 1-2 days | 🏆 Fast CNN |
| **Training Cost** | $200-$500 | $20-$100 | 🏆 Fast CNN |
| **Deployment Cost** | $150/mo (GPU required) | $30/mo (CPU capable) | 🏆 Fast CNN |

**Decision**: Enhanced Fast CNN achieves the **<100ms E2E latency requirement** while being cost-effective for production deployment. The 5-10% accuracy trade-off is acceptable for most real-world applications.

---

### Why Attention + GRU?

**Self-Attention** (4 heads):
- Learns: "This voice at t=0 matches this voice at t=2"
- Captures: Long-range speaker dependencies
- Cost: ~0.5M params, +1-2ms latency

**Bidirectional GRU** (1 layer):
- Learns: Temporal smoothing ("same speaker for 0.5 seconds")
- Captures: Sequential patterns, reduces jitter
- Cost: ~0.8M params, +1-2ms latency

**Combined Benefit**: +5-10% accuracy improvement while keeping latency <10ms

---

## 🔧 Implementation Details

### Audio Preprocessing
```python
def preprocess_audio(audio_path, sample_rate=16000, duration=3.0):
    """
    Load and preprocess audio for model input.
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Extract fixed-duration segment
    target_length = int(sample_rate * duration)
    if waveform.shape[1] > target_length:
        # Random crop during training
        start = random.randint(0, waveform.shape[1] - target_length)
        waveform = waveform[:, start:start + target_length]
    elif waveform.shape[1] < target_length:
        # Pad with zeros
        pad_length = target_length - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_length))
    
    return waveform.squeeze(0)  # [audio_length]
```

### Data Augmentation
```python
class AudioAugmentation:
    """Data augmentation for speaker diarization."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def add_noise(self, waveform, snr_db=20):
        """Add Gaussian noise at specified SNR."""
        signal_power = waveform.norm(p=2)
        noise_power = signal_power / (10 ** (snr_db / 20))
        noise = torch.randn_like(waveform) * noise_power
        return waveform + noise
    
    def time_mask(self, waveform, mask_ratio=0.1):
        """Mask random time segments."""
        length = waveform.shape[0]
        mask_length = int(length * mask_ratio)
        start = random.randint(0, length - mask_length)
        waveform[start:start + mask_length] = 0
        return waveform
    
    def pitch_shift(self, waveform, n_steps=2):
        """Shift pitch by ±n semitones."""
        # Requires librosa or torchaudio.functional
        return torchaudio.functional.pitch_shift(
            waveform, self.sample_rate, n_steps
        )
    
    def __call__(self, waveform):
        """Apply random augmentation."""
        aug_type = random.choice(['noise', 'time_mask', 'pitch_shift', 'none'])
        
        if aug_type == 'noise':
            return self.add_noise(waveform, snr_db=random.uniform(10, 30))
        elif aug_type == 'time_mask':
            return self.time_mask(waveform, mask_ratio=random.uniform(0.05, 0.15))
        elif aug_type == 'pitch_shift':
            return self.pitch_shift(waveform, n_steps=random.randint(-2, 2))
        else:
            return waveform
```

---

## 🎯 Next Steps

1. **Immediate**: Choose dataset (VoxCeleb2 recommended for quick start)
2. **Week 1**: Implement data loader and verify with 100 samples
3. **Week 2**: Implement enhanced model architecture
4. **Week 3-4**: Train baseline model (30 epochs)
5. **Week 5**: Evaluate, optimize, export ONNX
6. **Week 6**: Deploy to production, validate E2E latency

---

## 📚 References

- **VoxCeleb**: https://www.robots.ox.ac.uk/~vg/data/voxceleb/
- **AMI Corpus**: https://groups.inf.ed.ac.uk/ami/corpus/
- **pyannote.audio**: https://github.com/pyannote/pyannote-audio (SOTA diarization)
- **ONNX Runtime**: https://onnxruntime.ai/ (deployment)
- **Speaker Diarization SOTA**: https://paperswithcode.com/task/speaker-diarization

---

**Status**: 📝 Architecture documented, ready for implementation  
**Next Action**: Implement dataset loader (`scripts/data/voxceleb_dataset.py`)
