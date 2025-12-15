# Parallel Training Strategy: Pseudo-Labels vs Real Diarization

**Date**: December 15, 2025  
**Status**: In Progress  
**Context**: Running dual training experiments on Google Colab Pro to compare LibriSpeech pseudo-labels vs real diarization annotations

---

## 🎯 Objective

Evaluate the **real capacity** of our FastDiarizationModel by training in parallel on:
1. **LibriSpeech** (pseudo-labels from speaker_id) - Baseline
2. **AMI Corpus** (real diarization annotations) - Target quality

This allows direct comparison to understand the performance gap between synthetic and production-quality data.

---

## 📊 Training Configurations

### Training 1: LibriSpeech (Baseline)
- **Dataset**: `librispeech_asr` clean split
- **Labels**: Pseudo-labels from `speaker_id` (round-robin assignment)
- **Train**: 10,000 samples from train.360
- **Val**: 1,000 samples
- **Checkpoints**: `/content/drive/MyDrive/voiceflow_checkpoints/`
- **Status**: Running (Epoch 7/30, ~1h30 remaining)
- **Current Results**:
  - Best Val Acc: 61.50% (Epoch 4)
  - Train Acc: ~48% (plateauing)
  - Variance: ±15% (pseudo-label instability)

### Training 2: AMI Corpus (Production Quality)
- **Dataset**: `edinburghcstr/ami` with real diarization annotations
- **Labels**: Real speaker segments with timestamps
- **Train**: ~5,000 samples (multi-speaker meetings)
- **Val**: ~500 samples
- **Checkpoints**: `/content/drive/MyDrive/ami_checkpoints/`
- **Status**: To launch in parallel
- **Expected Results**:
  - Val Acc: 75-85% (real labels)
  - Lower variance (consistent labels)
  - Better Train Acc progression

---

## 🔬 Key Differences

| Aspect | LibriSpeech (Pseudo) | AMI Corpus (Real) |
|--------|---------------------|-------------------|
| **Label Source** | speaker_id (single-speaker utterances) | Manual diarization annotations |
| **Speaker Overlap** | None (clean single-speaker) | Yes (meetings with interruptions) |
| **Task Realism** | Discriminate between speakers | Actual diarization boundaries |
| **Label Consistency** | Changes with streaming order | Stable across epochs |
| **Data Quality** | Clean studio recordings | Natural meeting environment |
| **Expected Acc** | 55-65% (limited by pseudo-labels) | 75-85% (real task) |

---

## 🚀 Implementation Details

### AMI Dataset Structure
```python
# Sample format from AMI Corpus
{
    'audio': {
        'array': np.ndarray,  # Audio waveform
        'sampling_rate': 16000
    },
    'speaker_segments': [
        {'speaker': 'A', 'start': 0.0, 'end': 2.5},
        {'speaker': 'B', 'start': 2.3, 'end': 5.1},
        # ...
    ],
    'meeting_id': 'ES2002a',
    'num_speakers': 4
}
```

### Preprocessing Strategy
For binary classification (2 speakers):
- **Option 1**: Extract segments with exactly 2 active speakers
- **Option 2**: Group speakers into 2 clusters (A+B vs C+D)
- **Option 3**: Train on all speakers, test binary discrimination

### Checkpoint Isolation
```
Google Drive Structure:
├── voiceflow_checkpoints/          # LibriSpeech training
│   ├── checkpoint_epoch5.pth
│   ├── checkpoint_epoch10.pth
│   └── best_model.pth              # 61.50% Val Acc
│
└── ami_checkpoints/                # AMI training (parallel)
    ├── checkpoint_epoch5.pth
    ├── checkpoint_epoch10.pth
    └── best_model.pth              # Expected: 75-85% Val Acc
```

---

## 📈 Success Metrics

### Training Convergence
- **Train Loss**: Should descend smoothly (not plateauing at 0.694)
- **Train Acc**: Should reach 70-80% (not stuck at 48%)
- **Val Acc**: Target 75-85% with real labels

### Label Quality Indicators
- **Variance**: Should be <5% (currently ±15% with pseudo-labels)
- **Best Model**: Should improve beyond epoch 4
- **Loss Descent**: Smooth curve without plateaus

### Production Readiness
- ✅ **Val Acc > 75%**: Ready for production deployment
- ⚠️ **Val Acc 65-75%**: Suitable for PoC, needs refinement
- ❌ **Val Acc < 65%**: Data quality issues (like pseudo-labels)

---

## 🔄 Parallel Execution Plan

### Timeline (Next 2 Hours)
```
Now                 +30min              +1h30              +2h
|-------------------|-------------------|------------------|
LibriSpeech: Epoch 7 → Epoch 15 → Epoch 30 (Complete)
AMI:         Setup → Epoch 5 → Epoch 15 → Continue
```

### Resource Usage (Colab Pro)
- **GPU**: T4 (shared between 2 notebooks)
- **RAM**: 12GB per notebook (24GB total available)
- **Storage**: 0GB (both use streaming)
- **Checkpoints**: ~500MB per training (~1GB total on Drive)

### Monitoring Strategy
1. **LibriSpeech**: Let run to completion (30 epochs)
2. **AMI**: Monitor first 5 epochs for quick validation
3. **Comparison**: Evaluate Val Acc curves at epoch 10

---

## 📝 Hypotheses to Validate

### H1: Real Labels → Higher Accuracy
**Hypothesis**: AMI training will reach 75-85% Val Acc (vs 61.5% LibriSpeech)  
**Validation**: Compare best_model.pth from both trainings  
**Expected**: +15-25% absolute improvement

### H2: Real Labels → Stable Training
**Hypothesis**: AMI Val Acc variance <5% (vs ±15% LibriSpeech)  
**Validation**: Track Val Acc across epochs 1-10  
**Expected**: Smooth progression without jumps

### H3: Real Labels → Better Learning
**Hypothesis**: AMI Train Acc >70% (vs ~48% LibriSpeech)  
**Validation**: Monitor Train Acc at epoch 10  
**Expected**: Model learns beyond data distribution

---

## 🎓 Lessons Applied

From previous debugging experience:
1. ✅ **Data First**: We fixed distribution before architecture
2. ✅ **Balanced Labels**: Round-robin assignment for 50/50 split
3. ✅ **Streaming**: Zero storage usage on Colab
4. ✅ **Checkpointing**: Resume training after disconnects
5. 🆕 **Real Data**: Now testing with production-quality annotations

---

## 🔮 Expected Outcomes

### Best Case Scenario
- AMI reaches 80%+ Val Acc
- Smooth training curves
- Model ready for production deployment
- Clear proof that architecture is sound (data was the bottleneck)

### Realistic Scenario
- AMI reaches 75-78% Val Acc
- Some variance due to dataset complexity
- Model suitable for production with minor tuning
- Validates that real data solves pseudo-label limitations

### Worst Case Scenario
- AMI reaches similar ~60% Val Acc
- Indicates architecture limitations (not just data)
- Need to revisit model design (attention mechanisms, deeper network)
- More complex debugging required

---

## 📊 Results Summary (To be updated)

| Metric | LibriSpeech (Pseudo) | AMI Corpus (Real) | Δ Improvement |
|--------|---------------------|-------------------|---------------|
| Best Val Acc | 61.50% | TBD | TBD |
| Train Acc (Epoch 10) | ~48% | TBD | TBD |
| Val Acc Variance | ±15% | TBD | TBD |
| Training Stability | Low | TBD | TBD |
| Production Ready | ❌ No | TBD | TBD |

---

## 🚀 Next Actions

1. ✅ Create AMI training notebook
2. ✅ Launch parallel training on second Colab tab
3. ⏳ Monitor first 5 epochs for quick validation
4. ⏳ Let LibriSpeech complete (30 epochs)
5. ⏳ Compare final results and document findings
6. ⏳ Export best ONNX model from winning approach

**Goal**: Determine if our architecture is production-ready with real data, or if we need deeper model improvements.
