# 📓 Training Notebooks

## Current Status (December 2025)

### ✅ Production-Ready Notebooks

#### 1. `streaming_training_wavlm_colab.ipynb` 🌟 **RECOMMENDED**
**Status**: Ready for training  
**Purpose**: Train speaker diarization with pre-trained WavLM embeddings  
**Approach**: Uses Microsoft WavLM-Base-Plus-SV (speaker verification model)  
**Configuration**: Configured for D:/ drive caching with full dataset download  

**Key Features**:
- Pre-trained WavLM embeddings (512-dim speaker features)
- Simple MLP classifier on top (256K params, very fast to train)
- 50,000 training samples from LibriSpeech
- Expected accuracy: **>85%** (significantly better than previous attempts)
- Training time: 4-6 hours on Colab T4 GPU

**Use Cases**:
- ✅ **Local training** with D:/ drive caching
- ⏳ **Colab T4 GPU** (needs optimization - see below)

**Next Steps**: Run this notebook locally or wait for Colab-optimized version

---

#### 2. `gpu_benchmark_colab.ipynb`
**Status**: Utility notebook  
**Purpose**: Benchmark GPU performance for model inference  
**Use**: Performance testing, not for training

---

### 📦 Archived Experiments

Location: `archive_experiments/`

These notebooks were experimental approaches that **hit accuracy limitations** due to:
1. **Pseudo-label issues**: Single-speaker audio doesn't teach true diarization
2. **Architecture limitations**: Hand-crafted features (MFCC, x-vectors) underperformed
3. **Data quality**: AMI dataset had HuggingFace compatibility issues

#### Archived Notebooks:

1. **`streaming_training_colab.ipynb`**
   - Original CNN approach with pseudo-labels
   - Accuracy: 61.5% (peaked early, then plateaued)
   - Issue: Learned easy patterns but couldn't generalize
   - Lesson: Simple CNN isn't enough for complex speaker features

2. **`streaming_training_mfcc_colab.ipynb`**
   - MFCC feature-based approach
   - Issue: Hand-crafted features too simple for modern diarization
   - Lesson: Need learned representations (like WavLM)

3. **`streaming_training_xvectors_colab.ipynb`**
   - X-vector embeddings approach
   - Issue: Complex pipeline, harder to optimize
   - Lesson: Pre-trained models (WavLM) are more reliable

4. **`streaming_training_ami_colab.ipynb`**
   - AMI corpus approach (multi-speaker meetings)
   - Issue: HuggingFace deprecated AMI dataset scripts
   - Lesson: Stick with well-supported datasets (LibriSpeech, VoxCeleb)

---

## 🎓 Lessons Learned

### What Worked ✅
1. **Pre-trained speaker models** (WavLM) dramatically improve accuracy
2. **Streaming datasets** enable training without massive downloads
3. **Google Colab T4** provides free GPU for fast training
4. **Persistent label mapping** critical for consistent training
5. **Real speaker data** essential (not synthetic/pseudo-labels)

### What Didn't Work ❌
1. **Pseudo-labels** from single-speaker audio (50-61% accuracy ceiling)
2. **Hand-crafted features** (MFCC) insufficient for modern diarization
3. **Simple CNN architectures** without pre-trained embeddings
4. **Deprecated datasets** (AMI on HuggingFace)
5. **Hash-based label assignment** creates class imbalance

### Key Insights 💡
1. **Use transfer learning**: Pre-trained models (WavLM) >> training from scratch
2. **Start with proven approaches**: WavLM/Wav2Vec2 are industry standard
3. **Validate early**: 5 epochs shows if approach works
4. **Monitor Train-Val gap**: Overfitting detection critical
5. **Data quality > Data quantity**: Real speaker data beats synthetic

---

## 🚀 Recommended Training Path

### Option 1: Local Training (Current Setup)
1. Run `streaming_training_wavlm_colab.ipynb` locally
2. Uses D:/ drive for dataset caching (~60GB)
3. Requires: GPU (NVIDIA) or patience (CPU is slow)
4. Expected: >85% accuracy with full dataset

### Option 2: Google Colab T4 (Coming Soon)
1. Use upcoming `streaming_training_wavlm_colab_gpu.ipynb`
2. Optimized for Colab T4 GPU (streaming mode)
3. Saves checkpoints to Google Drive
4. Auto-resume after 12h session timeout
5. Expected: 4-6 hours to train, >85% accuracy

---

## 📊 Performance Benchmarks

| Approach | Architecture | Accuracy | Training Time | Status |
|----------|--------------|----------|---------------|--------|
| **WavLM + MLP** | Pre-trained embeddings | **>85%** (expected) | 4-6h (T4) | ✅ Ready |
| CNN + Pseudo-labels | Simple CNN | 61.5% | 2h (T4) | ❌ Archived |
| MFCC + CNN | Hand-crafted | <60% (estimated) | - | ❌ Archived |
| X-vectors | Speaker embeddings | Not tested | - | ❌ Archived |
| AMI Corpus | Real meetings | Dataset unavailable | - | ❌ Archived |

---

## 📁 File Structure

```
notebooks/
├── README.md (this file)
├── streaming_training_wavlm_colab.ipynb ⭐ Use this
├── gpu_benchmark_colab.ipynb
├── diarization_transformer_optimized.onnx (old export)
└── archive_experiments/
    ├── streaming_training_colab.ipynb (61.5% accuracy)
    ├── streaming_training_mfcc_colab.ipynb
    ├── streaming_training_xvectors_colab.ipynb
    └── streaming_training_ami_colab.ipynb
```

---

## 🔥 Quick Start

### For Local Training
```bash
# 1. Open the notebook
code streaming_training_wavlm_colab.ipynb

# 2. Configure Python environment
# Make sure you have the .venv activated

# 3. Run cells in order
# First run downloads ~60GB to D:/ (one-time)
# Subsequent runs use cache (instant)

# 4. Monitor training
# Expected: >85% accuracy after 30 epochs (4-6h on GPU)
```

### For Colab Training (When Ready)
```bash
# 1. Upload streaming_training_wavlm_colab_gpu.ipynb to Colab
# 2. Change runtime to GPU (T4)
# 3. Run all cells
# 4. Training starts immediately (streaming mode, no download)
# 5. Checkpoints auto-save to Google Drive
```

---

## 🆘 Need Help?

### Troubleshooting
- **Low accuracy (<70%)**: Check label distribution in dataset creation cell
- **Out of memory**: Reduce batch size (64 → 32)
- **Slow training**: Verify GPU is being used (`torch.cuda.is_available()`)
- **Session timeout (Colab)**: Checkpoints auto-resume on restart

### Documentation
- **Training results**: See `docs/TRAINING_RESULTS_SUMMARY.md`
- **Architecture details**: See `docs/REAL_DATA_TRAINING_ARCHITECTURE.md`
- **Dataset strategy**: See `docs/DATASET_SELECTION_SUMMARY.md`
- **D:/ drive setup**: See `docs/D_DRIVE_TRAINING_SETUP.md`

---

**Last Updated**: December 18, 2025  
**Recommended Approach**: WavLM embeddings + MLP classifier  
**Next Milestone**: Train on Colab T4, achieve >85% accuracy, export to ONNX
