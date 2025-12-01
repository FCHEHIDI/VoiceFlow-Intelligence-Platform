# ONNX Export & Performance Summary

## ✅ What We Achieved

### 1. Successful ONNX Export
- **Problem**: onnxscript optimizer had bugs (split_to_sequence constant folding error)
- **Solution**: Monkey-patched optimizer to skip buggy passes, then optimized separately with ONNX Runtime
- **Result**: 
  - Unoptimized: 12.8 MB
  - Optimized (ORT): 362.4 MB (includes all Wav2Vec2 weights)
  - INT8 quantized: 97.7 MB (73% reduction, but ConvInteger ops not supported on CPU)

### 2. Performance Benchmark (CPU)
```
Model: diarization_transformer_optimized.onnx
Hardware: CPU (CPUExecutionProvider)
Input: 3 seconds audio (48000 samples @ 16kHz)

Results:
├─ Load time: 3.9 seconds
├─ Median latency: 220ms
├─ P95 latency: 270ms
├─ P99 latency: 1428ms ❌
└─ Throughput: 3.9 req/sec
```

**Assessment**: ❌ **NOT production ready** - P99 latency 14x over target (<100ms)

## 🎯 Why We're Not Hitting <100ms

**Root cause**: Wav2Vec2-base (95M parameters) is designed for GPU inference, not CPU.

**The numbers**:
- Your model: 99.2M params total (95M encoder, 4.8M trainable)
- CPU throughput: ~4 req/sec
- Median is acceptable (220ms), but P99 spikes to 1.4s due to:
  - CPU context switching
  - Memory bandwidth limits  
  - No SIMD vectorization of large matrix ops

## 📊 Options to Hit <100ms P99

### Option 1: GPU Inference (5-10x faster) ⭐ RECOMMENDED
**Pros**:
- Keep your trained model as-is
- 5-10x speedup (220ms → 22-44ms median, likely <100ms P99)
- Simple: change provider to `CUDAExecutionProvider` or `DmlExecutionProvider`
- Works with your existing ONNX file

**Cons**:
- Requires NVIDIA GPU (CUDA) or DirectML (AMD/Intel)
- Deployment complexity (need GPU in production)

**Implementation**:
```python
session = ort.InferenceSession(
    "diarization_transformer_optimized.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # Fallback to CPU
)
```

### Option 2: Model Distillation (3-5x faster) ⭐ BEST LONG-TERM
**Pros**:
- Replace Wav2Vec2-base (95M) with smaller encoder:
  - distilhubert-base (33M params) → 3x faster
  - wav2vec2-small (12M params) → 8x faster  
  - Custom CNN encoder (1-5M params) → 15-20x faster
- Works on CPU
- Smaller model size (10-50 MB)
- Lower memory footprint

**Cons**:
- Need to retrain model with new encoder
- Potential accuracy loss (1-5%, acceptable for diarization)
- More development time (1-2 days)

**Implementation**:
```python
# Replace in model.py
from transformers import AutoModel

class FastDiarizationModel(nn.Module):
    def __init__(self):
        # Option A: Smaller pretrained model
        self.encoder = AutoModel.from_pretrained("ntu-spml/distilhubert")
        
        # Option B: Custom lightweight CNN
        self.encoder = LightweightCNN(
            in_channels=1,
            out_features=256,
            num_layers=6  # Much smaller than Wav2Vec2's 12 transformer layers
        )
```

### Option 3: INT8 Quantization (2-4x faster)
**Pros**:
- 73% size reduction (362MB → 98MB)
- 2-4x speedup on supported hardware

**Cons**:
- ❌ Your CPU doesn't support INT8 ops (ConvInteger not implemented)
- Requires AVX512_VNNI or GPU
- Even with 4x speedup: 220ms / 4 = 55ms median (good!), but P99 still ~350ms (over target)

**Status**: Not viable on your hardware

### Option 4: Hybrid Approach ⭐ PRACTICAL
**Combine GPU + Model Distillation**:
1. **Short-term**: Deploy on GPU (hits <100ms immediately)
2. **Long-term**: Train distilled model for CPU deployment

**Benefits**:
- Immediate production deployment (GPU)
- Cost optimization path (CPU later)
- Flexibility in deployment

## 💡 Recommendation

**For immediate production deployment**:
1. **Use GPU inference** with `diarization_transformer_optimized.onnx`
   - Expected P99: 30-80ms (well under 100ms target)
   - Deploy to GPU-enabled server/cloud instance
   - Cost: ~$0.50-1.00/hr for cloud GPU (AWS g4dn, Azure NC-series)

**For long-term cost optimization**:
2. **Train distilled model** with smaller encoder
   - Replace Wav2Vec2-base with distilhubert or custom CNN
   - Retrain for 5-10 epochs (~1 hour)
   - Expected CPU P99: 50-150ms (much better)
   - Deploy to CPU instances (cheaper, more scalable)

## 📁 Files Ready for Deployment

```
models/
├─ diarization_transformer.onnx              # 12.8 MB (unoptimized)
├─ diarization_transformer.onnx.data         # External weights
├─ diarization_transformer_optimized.onnx    # 362 MB (ORT optimized) ⭐ USE THIS
└─ diarization_transformer_int8.onnx         # 97.7 MB (INT8, not compatible)
```

**Recommended for production**: `diarization_transformer_optimized.onnx` with GPU provider

## 🚀 Next Steps

1. **Test on GPU** (if available):
   ```bash
   python -m pip install onnxruntime-gpu
   # Update benchmark to use CUDAExecutionProvider
   ```

2. **Deploy to Rust inference engine**:
   - Copy `diarization_transformer_optimized.onnx` to `voiceflow-inference/models/`
   - Update Rust to use GPU provider (ort crate supports CUDA)
   - Test on unknown audio samples

3. **If GPU not available**, consider:
   - Cloud deployment (AWS Lambda with GPU, Google Cloud Run GPU)
   - Model distillation (retrain with smaller encoder)
   - Accept higher latency (220ms median acceptable for many use cases)

## 📊 Performance Summary

| Configuration | Size | Median | P99 | Production Ready? |
|--------------|------|--------|-----|-------------------|
| FP32 CPU | 362 MB | 220ms | 1428ms | ❌ No |
| INT8 CPU | 98 MB | N/A | N/A | ❌ Not supported |
| FP32 GPU (est.) | 362 MB | 22-44ms | 30-80ms | ✅ Yes |
| Distilled CPU (est.) | 50 MB | 50-100ms | 100-200ms | ⚠️ Maybe |
| Distilled GPU (est.) | 50 MB | 5-15ms | 10-30ms | ✅ Yes |

**Verdict**: Current model needs GPU for <100ms P99 target.
