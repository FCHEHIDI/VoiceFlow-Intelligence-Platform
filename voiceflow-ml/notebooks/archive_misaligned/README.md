# 🗂️ Archive: Misaligned Training Approaches

This directory contains notebooks that used **incorrect approaches** for speaker diarization.

## Why These Were Archived

These notebooks trained **multi-class classification models** (predicting speaker IDs from a fixed set of speakers), which is fundamentally wrong for diarization because:

1. **Closed World Problem**: Only works for speakers seen during training
2. **No Generalization**: Can't handle new speakers in production
3. **Wrong Task**: Classification ≠ Clustering/Grouping unknown speakers

## What Was Wrong

### `streaming_training_wavlm_colab_gpu.ipynb`
- Trained WavLM classifier with 479 output classes
- Used LibriSpeech speaker IDs as labels
- Achieved only ~60% accuracy
- **Problem**: Would only recognize those 479 specific people!

## The Correct Approach

See `/docs/EMBEDDING_BASED_DIARIZATION_GUIDE.md` for the proper embedding-based approach:

```
❌ OLD: Audio → Neural Net → Speaker ID (0-478)
✅ NEW: Audio → Neural Net → Embedding (512-dim) → Clustering → Speaker ID
```

The new approach:
- Uses **contrastive/triplet loss** (learn similarity, not identity)
- Produces **speaker embeddings** that cluster automatically
- Works with **any speakers**, never seen before
- Generalizes to production use cases

## Lessons Learned

1. Always validate that your training task matches your inference task
2. For open-world problems (unknown speakers), use embedding approaches
3. Classification is only appropriate when you know all classes beforehand
4. LibriSpeech speaker IDs are for research, not production diarization

---

**For current training approach**, see:
- `/docs/EMBEDDING_BASED_DIARIZATION_GUIDE.md` (theory)
- `/notebooks/diarization_embedding_training_colab.ipynb` (implementation)
- `/voiceflow-ml/models/diarization/train.py` (training script)
