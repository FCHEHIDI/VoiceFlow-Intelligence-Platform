"""
Generate test audio and validate Rust inference integration.

Creates synthetic audio samples for testing the Rust ONNX inference.
"""

import numpy as np
import wave
import struct
from pathlib import Path


def generate_test_audio(output_path: str, duration_sec: float = 3.0, sample_rate: int = 16000):
    """Generate synthetic audio for testing."""
    
    print(f"🎵 Generating test audio...")
    print(f"   ├─ Duration: {duration_sec}s")
    print(f"   ├─ Sample rate: {sample_rate} Hz")
    print(f"   └─ Output: {output_path}")
    
    # Generate audio (sine wave + noise)
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples)
    
    # Mix of frequencies to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.2 * np.sin(2 * np.pi * 500 * t) +  # Mid frequency
        0.1 * np.sin(2 * np.pi * 1000 * t) + # High frequency
        0.1 * np.random.randn(num_samples)    # Noise
    )
    
    # Normalize to [-1, 1]
    audio = audio / np.max(np.abs(audio))
    
    # Convert to int16 for WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"   ✓ Test audio saved: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    # Also save as raw f32 for direct testing
    raw_path = output_path.replace('.wav', '_f32.bin')
    audio.astype(np.float32).tofile(raw_path)
    print(f"   ✓ Raw f32 audio saved: {Path(raw_path).stat().st_size / 1024:.1f} KB")
    
    return audio.astype(np.float32)


def test_python_inference(audio: np.ndarray):
    """Test inference with Python ONNX Runtime."""
    
    print(f"\n🐍 Testing Python ONNX inference...")
    
    try:
        import onnxruntime as ort
        
        model_path = "../models/diarization_transformer_optimized.onnx"
        
        # Load model
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Prepare input
        input_tensor = audio.reshape(1, -1)  # [1, audio_length]
        
        # Run inference
        outputs = session.run(
            None,
            {'audio': input_tensor}
        )
        
        probabilities = outputs[0][0]
        
        print(f"   ✓ Inference successful!")
        print(f"   ├─ Speaker 1: {probabilities[0] * 100:.2f}%")
        print(f"   └─ Speaker 2: {probabilities[1] * 100:.2f}%")
        
        return probabilities
        
    except Exception as e:
        print(f"   ❌ Python inference failed: {e}")
        return None


def generate_rust_test_code():
    """Generate Rust test code snippet."""
    
    test_code = '''
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn test_transformer_inference() {
        // Load test audio
        let audio_bytes = fs::read("../voiceflow-ml/test_audio_f32.bin")
            .expect("Test audio file not found");
        
        // Convert bytes to f32
        let audio: Vec<f32> = audio_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        println!("Loaded {} audio samples", audio.len());
        
        // Load model
        let runner = ModelRunner::load(
            "models/diarization_transformer_optimized.onnx",
            "transformer"
        ).expect("Failed to load model");
        
        // Run inference
        let result = runner.run_inference(&audio)
            .expect("Inference failed");
        
        println!("Speaker 1: {:.2}%", result[0] * 100.0);
        println!("Speaker 2: {:.2}%", result[1] * 100.0);
        
        // Validate output
        assert_eq!(result.len(), 2, "Should return 2 speaker probabilities");
        assert!(result[0] >= 0.0 && result[0] <= 1.0, "Probability should be in [0, 1]");
        assert!(result[1] >= 0.0 && result[1] <= 1.0, "Probability should be in [0, 1]");
        
        // Probabilities should roughly sum to 1 (with softmax)
        let sum = result[0] + result[1];
        assert!((sum - 1.0).abs() < 0.1, "Probabilities should sum to ~1.0");
    }
}
'''
    
    print(f"\n🦀 Rust test code:")
    print(f"   Add this to voiceflow-inference/src/inference/mod.rs:\n")
    print(test_code)


def main():
    # Generate test audio
    audio = generate_test_audio("test_audio.wav", duration_sec=3.0)
    
    # Test with Python
    test_python_inference(audio)
    
    # Generate Rust test code
    generate_rust_test_code()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Test Setup Complete!")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"1. Run Rust tests: cd ../voiceflow-inference && cargo test")
    print(f"2. Build Rust project: cargo build --release")
    print(f"3. Run server: cargo run --release")
    print(f"\nModel location: models/diarization_transformer_optimized.onnx")


if __name__ == "__main__":
    main()
