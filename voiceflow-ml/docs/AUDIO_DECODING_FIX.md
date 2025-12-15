# Quick Fix for Audio Decoding Error in Google Colab

## Problem
```
ImportError: To support decoding audio data, please install 'torchcodec'.
```

## Solution 1: Install Audio Dependencies (Recommended)

Run this cell in Colab:

```python
# Install datasets with audio support
!pip install -q datasets[audio] librosa soundfile

# Verify installation
import datasets
print(f"datasets version: {datasets.__version__}")
print("✅ Audio decoding dependencies installed")
```

## Solution 2: Use Audio Casting (Alternative)

If Solution 1 doesn't work, modify your dataset loading:

```python
from datasets import load_dataset, Audio

# Load with audio casting
train_dataset = load_dataset(
    "librispeech_asr",
    "clean",
    split="train.360",
    streaming=True,
    trust_remote_code=True
)

# Cast audio column to force proper decoding
train_dataset = train_dataset.cast_column(
    "audio", 
    Audio(sampling_rate=16000, decode=True)
)

print("✅ Dataset ready with audio decoding")
```

## Solution 3: Manual Audio Processing (Fallback)

If both above fail, process audio manually with torchaudio:

```python
import torchaudio
import io

def load_audio_from_bytes(audio_bytes, target_sr=16000):
    """Load audio from bytes using torchaudio."""
    # Convert bytes to file-like object
    audio_file = io.BytesIO(audio_bytes)
    
    # Load with torchaudio
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform.squeeze(0).numpy()

# Use in dataset iteration
for sample in train_dataset:
    try:
        # Try normal decoding first
        audio_array = sample['audio']['array']
    except:
        # Fallback to manual decoding
        audio_array = load_audio_from_bytes(sample['audio']['bytes'])
```

## Root Cause

The `datasets` library changed its audio decoding backend. The fix is to:
1. Install audio extras: `datasets[audio]`
2. Or explicitly cast audio columns with sampling rate
3. Or process audio manually with torchaudio/librosa

## Updated Requirements

Add to your notebook installation cell:

```python
%%capture
!pip install -q datasets[audio] torch torchaudio transformers librosa soundfile
```

The `[audio]` extra includes:
- `librosa` - Audio loading and processing
- `soundfile` - Audio file I/O
- Other audio codec dependencies

## Verification

Run this to verify audio decoding works:

```python
from datasets import load_dataset

# Test audio loading
test_dataset = load_dataset(
    "librispeech_asr",
    "clean",
    split="validation",
    streaming=True
)

# Try to decode one sample
sample = next(iter(test_dataset))
print(f"✅ Audio decoded successfully!")
print(f"   Shape: {sample['audio']['array'].shape}")
print(f"   Sample rate: {sample['audio']['sampling_rate']} Hz")
```

If this works, you're good to go! 🎉
