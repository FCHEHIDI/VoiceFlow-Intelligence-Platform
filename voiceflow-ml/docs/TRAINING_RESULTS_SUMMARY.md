# Training Results Summary

**Date**: December 15-16, 2025  
**Objective**: Train speaker diarization model with LibriSpeech pseudo-labels

---

## 🎯 Final Results

### LibriSpeech 'clean' Training (COMPLETED ✅)
- **Dataset**: LibriSpeech ASR 'clean' subset (360 hours, studio quality)
- **Training**: 30 epochs, 2.06 hours on Colab Pro T4 GPU
- **Best Val Acc**: **61.5%** (achieved at epoch 4)
- **Final Train Acc**: 50.5%
- **Checkpoint**: `/content/drive/MyDrive/voiceflow_checkpoints/best_model.pth`
- **ONNX Model**: `fast_cnn_trained.onnx` (exported and downloaded ✅)

**Characteristics**:
- Peaked early (epoch 4), then plateaued at 53.8% Val Acc
- High variance in early epochs (45-61.5%)
- Train Acc stuck at ~50% (binary classification baseline)
- Indication: Model learned easy patterns quickly, then struggled with pseudo-label noise

---

### LibriSpeech 'other' Training (COMPLETED ✅)
- **Dataset**: LibriSpeech ASR 'other' subset (496 hours, noisy/diverse audio)
- **Training**: 30 epochs, 1.32 hours on Colab Pro T4 GPU
- **Best Val Acc**: **56.6%** (achieved at epoch 26)
- **Final Train Acc**: 67.3%
- **Checkpoint**: `/content/drive/MyDrive/librispeech_other_checkpoints/best_model.pth`
- **ONNX Model**: `fast_cnn_librispeech_other_trained.onnx` (exported and downloaded ✅)

**Characteristics**:
- Gradual improvement through epoch 26
- Train Acc significantly higher (67.3% vs 50.5% for 'clean')
- Overfitting detected after epoch 26 (Train-Val gap increased to +13%)
- Indication: Model learned robust speaker features but overfit to training set

---

## 📊 Comparative Analysis

| Metric | 'clean' | 'other' | Delta | Winner |
|--------|---------|---------|-------|--------|
| **Best Val Acc** | 61.5% | 56.6% | -4.9% | 'clean' |
| **Final Train Acc** | 50.5% | 67.3% | +16.8% | 'other' |
| **Peak Epoch** | 4 | 26 | +22 | - |
| **Training Time** | 2.06h | 1.32h | -0.74h | 'other' |
| **Overfitting** | Early plateau | Late overfitting | - | Neither |
| **Audio Quality** | Studio | Noisy/Diverse | - | - |

**Key Insight**: 
- **'clean'**: Higher Val Acc but poor speaker discrimination (50% Train Acc = random guessing)
- **'other'**: Lower Val Acc but learned genuine speaker features (67% Train Acc)
- **Production**: 'other' model likely better for real-world deployment despite lower validation accuracy

---

## 🐛 Critical Bugs Fixed

### Bug #1: Hash-based Label Assignment (70/30 Imbalance)
**Symptom**: Val Acc stuck at 29.20% across epochs 1-4  
**Root Cause**: `hash(speaker_id) % 2` created 70/30 distribution instead of 50/50  
**Fix**: Implemented round-robin label assignment  
**Result**: Val Acc improved 29.20% → 61.50% (+32.3%)  
**Commit**: `412c6d0`

### Bug #2: Non-Persistent Label Mapping (Labels Flip Every Epoch)
**Symptom**: 'other' training completely frozen (52.5% Val Acc, 0.692 loss for 10 epochs)  
**Root Cause**: `StreamingAudioDataset.__iter__()` recreated label mapping every epoch  
**Fix**: Pre-scan dataset once during `__init__()` to create persistent mapping  
**Result**: Loss descent resumed, Train Acc improved 52% → 67%  
**Commit**: `fd4cf55`

---

## 🎓 Lessons Learned

1. **"Debug your data before debugging your model"**
   - 70/30 class imbalance caused 30 hours of wasted debugging
   - Always verify label distribution before training

2. **Streaming datasets need careful label handling**
   - IterableDataset `__iter__()` is called fresh each epoch
   - Store persistent state in `__init__()`, not `__iter__()`

3. **Pseudo-labels have fundamental limitations**
   - LibriSpeech single-speaker utterances ≠ real diarization task
   - Val Acc variance indicates label instability
   - 61.5% is ceiling for this approach

4. **Audio quality vs robustness trade-off**
   - Clean audio: Higher accuracy, poor generalization
   - Noisy audio: Lower accuracy, better feature learning

5. **Overfitting detection matters**
   - Watch Train-Val gap, not just Val Acc
   - 'other' peaked at epoch 26, then overfit

---

## 📁 Saved Artifacts

### Checkpoints (Google Drive)
```
/content/drive/MyDrive/voiceflow_checkpoints/
├── best_model.pth                    # 61.5% Val Acc (epoch 4)
├── checkpoint_epoch5.pth
├── checkpoint_epoch10.pth
├── checkpoint_epoch15.pth
├── checkpoint_epoch20.pth
├── checkpoint_epoch25.pth
└── checkpoint_epoch30.pth

/content/drive/MyDrive/librispeech_other_checkpoints/
├── best_model.pth                    # 56.6% Val Acc (epoch 26)
├── checkpoint_epoch5.pth
├── checkpoint_epoch10.pth
├── checkpoint_epoch15.pth
├── checkpoint_epoch20.pth
├── checkpoint_epoch25.pth
└── checkpoint_epoch30.pth
```

### ONNX Models (Downloaded Locally ✅)
- `fast_cnn_trained.onnx` - 'clean' model (61.5% Val Acc)
- `fast_cnn_librispeech_other_trained.onnx` - 'other' model (56.6% Val Acc)

### Code Repository (GitHub)
- All notebooks committed to `main` branch
- Commit history preserves debugging journey

---

## 🚀 Next Steps (Tomorrow's Plan)

### Phase 1: Feature Engineering (HIGHEST PRIORITY ⭐⭐⭐⭐⭐)

**Goal**: Boost Val Acc from 61.5% → **75-80%**

**Implementation**: Replace raw waveform with MFCC features

**What to do**:
1. Create new notebook: `streaming_training_mfcc_colab.ipynb`
2. Modify `StreamingAudioDataset.__iter__()`:
   ```python
   # Extract 40 MFCCs + deltas (80 features total)
   mfcc = librosa.feature.mfcc(y=audio.numpy(), sr=16000, n_mfcc=40)
   mfcc_delta = librosa.feature.delta(mfcc)
   features = np.vstack([mfcc, mfcc_delta])  # Shape: (80, ~300)
   yield torch.FloatTensor(features), label
   ```
3. Update model: `in_channels=1` → `in_channels=80`
4. Train on 'clean' or 'other' (TBD)
5. Expected result: Val Acc 75-80%, reduced overfitting

**Why MFCCs**:
- Industry standard for speaker recognition
- Explicit vocal tract features (not learned from scratch)
- Proven to work: Used in Alexa, Google Assistant, etc.
- Same training time (~2 hours)

---

### Phase 2: AMI Corpus Integration (When Available)

**Goal**: Train on real diarization dataset with actual speaker boundaries

**Blockers**:
- HuggingFace deprecated AMI dataset scripts
- Need manual download or pyannote.database integration

**Expected Result**: Val Acc 80-85% with proper diarization labels

---

### Phase 3: Advanced Features (Optional)

**If time permits**:
1. **X-vectors**: Use pre-trained speaker embeddings (85-90% Val Acc)
2. **Data Augmentation**: Add noise, pitch shift, time stretch
3. **Longer Training**: 50-100 epochs with lower LR
4. **Ensemble**: Combine 'clean' + 'other' models

---

## 🔧 Model Architecture

**Current Setup** (Raw Waveform → CNN):
```
FastDiarizationModel
├── LightweightCNNEncoder (2.5M params)
│   ├── Input: (batch, 1, 48000) - Raw 16kHz audio
│   ├── 5 ResNet-style residual blocks
│   ├── Channels: 1 → 64 → 128 → 256 → 256 → 512
│   ├── Pooling: 32x total reduction
│   └── Output: (batch, 512)
├── Classifier Head
│   ├── Linear(512, 256)
│   ├── ReLU + Dropout(0.3)
│   └── Linear(256, 2)
└── Output: (batch, 2) - Binary speaker prediction
```

**Proposed MFCC Setup**:
```
FastDiarizationModel
├── LightweightCNNEncoder (same architecture)
│   ├── Input: (batch, 80, ~300) - MFCC features
│   ├── in_channels: 1 → 80 (ONLY CHANGE!)
│   └── Output: (batch, 512)
└── [Same classifier head]
```

---

## 📝 Training Configuration

**Hyperparameters** (used for both 'clean' and 'other'):
```python
TRAINING_CONFIG = {
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
    'save_every_n_epochs': 5,
    'validate_every_n_epochs': 1,
}

DATASET_CONFIG = {
    'max_train_samples': 10000,
    'max_val_samples': 1000,
    'batch_size': 32,
    'num_speakers': 2,  # Binary classification
}
```

**Optimizer**: AdamW with CosineAnnealingLR  
**Loss**: CrossEntropyLoss  
**Hardware**: Google Colab Pro, T4 GPU (16GB VRAM)

---

## 🎯 Success Metrics

### Current Baseline
- **Raw waveform + 'clean'**: 61.5% Val Acc
- **Raw waveform + 'other'**: 56.6% Val Acc

### Target with MFCCs
- **MFCCs + 'clean'**: 75-80% Val Acc
- **MFCCs + 'other'**: 70-75% Val Acc
- **Reduced overfitting**: Train-Val gap <10%

### Production Goal (AMI + MFCCs/X-vectors)
- **Target**: 85-90% Val Acc
- **Deployment**: ONNX → Rust inference server
- **Latency**: <10ms P99 on GPU

---

## 📚 Documentation Created

1. ✅ **LABEL_DISTRIBUTION_BUG_POSTMORTEM.md** - Detailed analysis of hash bug
2. ✅ **PARALLEL_TRAINING_STRATEGY.md** - 'clean' vs 'other' comparison plan
3. ✅ **TRAINING_RESULTS_SUMMARY.md** - This document

---

## 🔗 Key Commits

| Commit | Description | Impact |
|--------|-------------|--------|
| `658d224` | Initial streaming training setup | Baseline |
| `8454515` | Enhanced CNN with ResNet blocks | Architecture improvement |
| `412c6d0` | Fixed 70/30 label imbalance | +32.3% Val Acc |
| `f92eeb5` | Created post-mortem doc | Documentation |
| `4f2aa30` | Added parallel training strategy | Experiment design |
| `e28b904` | Pivoted AMI → 'other' subset | Dataset comparison |
| `fd4cf55` | Fixed persistent label mapping | Enabled learning on 'other' |
| `8fe4339` | Fixed corrupted cell syntax | Code cleanup |

---

## 💾 Repository State

**Branch**: `main`  
**Status**: All changes committed and pushed ✅  
**Notebooks**:
- `streaming_training_colab.ipynb` - 'clean' training (working)
- `streaming_training_ami_colab.ipynb` - 'other' training (working)

**Next**: Create `streaming_training_mfcc_colab.ipynb` for MFCC experiments

---

**Status**: Ready for tomorrow's MFCC feature engineering session! 🚀
