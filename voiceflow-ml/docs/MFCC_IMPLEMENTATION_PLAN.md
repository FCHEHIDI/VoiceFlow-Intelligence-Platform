# MFCC Feature Engineering Implementation Plan

**Date**: December 16, 2025  
**Objective**: Boost speaker diarization accuracy from 61.5% → 75-80% using MFCC features

---

## 🎯 Overview

Replace raw waveform input with **Mel-Frequency Cepstral Coefficients (MFCCs)** - the industry standard for speaker recognition.

**Expected Impact**:
- ✅ Val Acc: 61.5% → **75-80%** (+13-18%)
- ✅ Training time: Similar (~2 hours)
- ✅ Reduced overfitting: More stable features
- ✅ Better generalization: Explicit vocal tract features

---

## 📋 Implementation Checklist

### Step 1: Create New Notebook ✅
- [x] Copy `streaming_training_colab.ipynb`
- [ ] Rename to `streaming_training_mfcc_colab.ipynb`
- [ ] Update title and description

### Step 2: Modify Dataset Class
- [ ] Update `StreamingAudioDataset.__iter__()` to extract MFCCs
- [ ] Add delta and delta-delta features
- [ ] Update shape documentation

### Step 3: Update Model Configuration
- [ ] Change `in_channels=1` → `in_channels=80` in CONFIG
- [ ] Verify FastDiarizationModel accepts new input shape
- [ ] Test forward pass with dummy MFCC input

### Step 4: Training
- [ ] Choose dataset: 'clean' or 'other' (recommend 'clean' first)
- [ ] Train for 30 epochs
- [ ] Monitor for improved convergence

### Step 5: Evaluation
- [ ] Compare Val Acc vs baseline (61.5%)
- [ ] Check Train-Val gap (should be <10%)
- [ ] Analyze learning curves

---

## 🔧 Code Changes

### Change 1: StreamingAudioDataset.__iter__() 

**Location**: Cell 6, lines ~290-330

**OLD CODE** (Raw waveform):
```python
def __iter__(self):
    count = 0
    for sample in self.dataset:
        # ... existing preprocessing ...
        
        # Use persistent label mapping
        speaker_id = str(sample.get('speaker_id', 0))
        label = self.speaker_to_label.get(speaker_id, 0)
        
        yield audio, label  # Shape: (48000,)
        count += 1
```

**NEW CODE** (MFCC features):
```python
def __iter__(self):
    count = 0
    for sample in self.dataset:
        # ... existing preprocessing ...
        
        # Extract MFCC features
        audio_np = audio.numpy()  # Convert to numpy for librosa
        
        # 40 MFCC coefficients
        mfcc = librosa.feature.mfcc(
            y=audio_np,
            sr=self.target_sr,
            n_mfcc=40,
            n_fft=512,      # 32ms window
            hop_length=160,  # 10ms hop
            n_mels=80
        )
        
        # Delta (velocity) features
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Stack: 40 MFCCs + 40 deltas = 80 features
        features = np.vstack([mfcc, mfcc_delta])  # Shape: (80, ~300 frames)
        
        # Convert back to tensor
        features = torch.FloatTensor(features)
        
        # Use persistent label mapping
        speaker_id = str(sample.get('speaker_id', 0))
        label = self.speaker_to_label.get(speaker_id, 0)
        
        yield features, label  # NEW Shape: (80, ~300)
        count += 1
```

**Key Points**:
- 40 MFCCs capture vocal tract shape (speaker-specific)
- Deltas capture temporal dynamics (speech patterns)
- Output shape: (80, ~300) for 3-second audio @ 16kHz

---

### Change 2: collate_fn()

**Location**: Cell 6, line ~352

**OLD CODE**:
```python
def collate_fn(batch):
    audios, labels = zip(*batch)
    return torch.stack(audios), torch.LongTensor(labels)
```

**NEW CODE** (same, but handles 2D features):
```python
def collate_fn(batch):
    features, labels = zip(*batch)
    # Stack along batch dimension
    # Input: list of (80, ~300)
    # Output: (batch, 80, ~300)
    return torch.stack(features), torch.LongTensor(labels)
```

---

### Change 3: Model Configuration

**Location**: Cell 5, line ~185

**OLD CODE**:
```python
CONFIG = {
    'num_speakers': 2,
    'hidden_size': 256,
    'encoder_type': 'lightweight-cnn',
    'dropout': 0.3,
}
```

**NEW CODE** (explicit in_channels):
```python
CONFIG = {
    'num_speakers': 2,
    'hidden_size': 256,
    'encoder_type': 'lightweight-cnn',
    'dropout': 0.3,
    'in_channels': 80,  # NEW: MFCC features (40 + 40 deltas)
}
```

---

### Change 4: Model Architecture (model.py)

**Location**: `voiceflow-ml/models/diarization/model.py`

**Current LightweightCNNEncoder.__init__()**:
```python
class LightweightCNNEncoder(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        
        # First conv: 1 channel (raw audio) → 64
        self.conv1 = ResidualBlock(1, 64, dropout=dropout)
        # ...
```

**NEW (Updated)**:
```python
class LightweightCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, dropout=0.3):
        super().__init__()
        
        # First conv: in_channels (1 for waveform, 80 for MFCC) → 64
        self.conv1 = ResidualBlock(in_channels, 64, dropout=dropout)
        self.conv2 = ResidualBlock(64, 128, dropout=dropout)
        # ... rest unchanged
```

**FastDiarizationModel.__init__() update**:
```python
def __init__(self, num_speakers, hidden_size=256, encoder_type='lightweight-cnn', dropout=0.3, in_channels=1):
    super().__init__()
    
    if encoder_type == 'lightweight-cnn':
        self.encoder = LightweightCNNEncoder(in_channels=in_channels, dropout=dropout)
    # ... rest unchanged
```

---

## 📊 Expected Results

### Baseline (Raw Waveform)
- 'clean': 61.5% Val Acc, high variance
- 'other': 56.6% Val Acc, overfitting after epoch 26

### Target (MFCC Features)
- **'clean' + MFCC**: 75-80% Val Acc
- **'other' + MFCC**: 70-75% Val Acc
- **Convergence**: Faster (peaks around epoch 15-20)
- **Stability**: Lower variance, reduced overfitting

### Why MFCCs Work Better

| Feature | Raw Waveform | MFCC |
|---------|--------------|------|
| **Input** | Amplitude samples | Cepstral coefficients |
| **What it captures** | Everything (signal+noise) | Vocal tract shape |
| **Speaker info** | Implicit | Explicit |
| **Noise robustness** | Poor | Good (mel-scale filtering) |
| **Learning** | CNN learns from scratch | CNN learns patterns on features |
| **Industry use** | Research only | Production (Alexa, Google) |

---

## 🔬 MFCC Theory (Quick Reference)

**What are MFCCs?**
1. **Mel-scale**: Mimics human ear's frequency perception
2. **Cepstral**: Separates vocal tract (speaker) from pitch (content)
3. **Coefficients**: Lower coefficients = speaker identity, higher = noise

**Feature Breakdown**:
- **40 MFCCs**: Vocal tract shape (speaker-specific)
- **40 deltas**: Temporal dynamics (how voice changes)
- **Total**: 80 features per time frame (~300 frames for 3s audio)

**Why 40 coefficients?**
- Lower 13: Speaker identity (used in voice recognition)
- Middle 14-30: Speech content
- Upper 31-40: Fine details + some noise
- We keep all 40 to let CNN learn which are useful

---

## 🧪 Testing Strategy

### Phase 1: Smoke Test (Quick Validation)
1. Run 1 epoch on 'clean' with MFCCs
2. Verify:
   - No shape errors
   - Loss descends (should be < 0.7 after epoch 1)
   - Train Acc > 55% (vs 45% for raw waveform)
3. If successful → proceed to full training

### Phase 2: Full Training
1. Train 30 epochs on 'clean'
2. Monitor metrics:
   - Val Acc should reach 70%+ by epoch 10
   - Train-Val gap should be <10%
   - Best Val Acc expected: 75-80%

### Phase 3: Comparison
1. Plot learning curves: MFCC vs Raw Waveform
2. Compare convergence speed
3. Analyze overfitting patterns

---

## 🚨 Potential Issues & Solutions

### Issue 1: MFCC Extraction Too Slow
**Symptom**: Training much slower than 2 hours  
**Cause**: librosa.feature.mfcc() computed per sample  
**Solution**: Pre-compute and cache MFCCs during initialization

### Issue 2: NaN Loss
**Symptom**: Loss becomes NaN during training  
**Cause**: MFCC values too large or contain inf  
**Solution**: Normalize MFCCs:
```python
features = (features - features.mean()) / (features.std() + 1e-8)
```

### Issue 3: No Improvement vs Baseline
**Symptom**: Val Acc still ~60%  
**Cause**: Wrong hyperparameters for feature input  
**Solution**: 
- Increase dropout: 0.3 → 0.4
- Reduce learning rate: 1e-3 → 5e-4
- Add batch normalization

---

## 🎯 Success Criteria

**Minimum Viable**:
- ✅ Val Acc ≥ 70% (vs 61.5% baseline)
- ✅ Trains without errors
- ✅ Converges within 30 epochs

**Target**:
- ✅ Val Acc 75-80%
- ✅ Train-Val gap <10%
- ✅ Stable learning (low variance)

**Stretch Goal**:
- ✅ Val Acc >80%
- ✅ Faster convergence (<20 epochs to peak)
- ✅ Export ONNX and benchmark inference

---

## 📝 Next Steps (Tomorrow)

1. **Morning**: 
   - Create `streaming_training_mfcc_colab.ipynb`
   - Implement MFCC extraction in dataset class
   - Update model.py with `in_channels` parameter

2. **Afternoon**:
   - Run smoke test (1 epoch)
   - If successful, launch full 30-epoch training
   - Monitor progress

3. **Evening**:
   - Analyze results
   - Compare vs baseline
   - Export ONNX if Val Acc >75%

---

## 🔗 References

- **librosa MFCC docs**: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
- **Speaker Recognition Tutorial**: https://www.sciencedirect.com/topics/computer-science/mel-frequency-cepstral-coefficient
- **Original MFCC paper**: Davis & Mermelstein (1980)

---

**Status**: Ready for implementation! 🚀
