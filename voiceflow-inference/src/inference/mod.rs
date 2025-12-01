"""
ONNX Runtime integration for model inference.
"""

use ort::{GraphOptimizationLevel, Session, Value};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;
use tracing::{info, error};

use crate::error::{AppError, AppResult};
use crate::metrics::{INFERENCE_LATENCY, MODEL_LOAD_TIME};

/// Model runner with ONNX Runtime
pub struct ModelRunner {
    session: Arc<Session>,
    version: String,
}

impl ModelRunner {
    /// Load model from ONNX file
    pub fn load(path: &str, version: &str) -> AppResult<Self> {
        let start = Instant::now();
        
        info!("Loading ONNX model from {}", path);
        
        let session = Session::builder()
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            // Enable GPU if available (CUDA/DirectML), fallback to CPU
            .with_execution_providers([
                ort::CUDAExecutionProvider::default().build(),
                ort::CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        
        let duration = start.elapsed();
        MODEL_LOAD_TIME.observe(duration.as_secs_f64());
        
        info!("Model loaded in {:?}", duration);
        
        Ok(Self {
            session: Arc::new(session),
            version: version.to_string(),
        })
    }

    /// Run inference on raw audio samples
    /// 
    /// Input: audio samples (16kHz mono, f32)
    /// Output: speaker probabilities [speaker_1, speaker_2]
    pub fn run_inference(&self, audio: &[f32]) -> AppResult<Vec<f32>> {
        let start = Instant::now();
        
        // Transformer model expects: [batch_size, audio_length]
        // Input shape: [1, audio.len()]
        let batch_size = 1;
        let audio_length = audio.len();
        
        info!("Running inference on {} samples ({:.2}s audio)", 
              audio_length, audio_length as f32 / 16000.0);
        
        // Create input tensor
        use ndarray::Array2;
        let input_array = Array2::from_shape_vec(
            (batch_size, audio_length),
            audio.to_vec()
        ).map_err(|e| AppError::InferenceError(format!("Failed to create input array: {}", e)))?;
        
        let input_tensor = Value::from_array(self.session.allocator(), &input_array)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        // Run inference with named input
        let outputs = self.session.run(ort::inputs!["audio" => input_tensor])
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        // Extract speaker probabilities (shape: [batch_size, num_speakers])
        let output = outputs["speaker_probabilities"]
            .extract_tensor::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        let result: Vec<f32> = output.view().iter().copied().collect();
        
        // Record latency
        let duration = start.elapsed();
        INFERENCE_LATENCY.observe(duration.as_secs_f64());
        
        info!("Inference completed in {:?} - Speaker 1: {:.2}%, Speaker 2: {:.2}%",
              duration, result.get(0).unwrap_or(&0.0) * 100.0, result.get(1).unwrap_or(&0.0) * 100.0);
        
        Ok(result)
    }

    /// Get model version
    pub fn version(&self) -> &str {
        &self.version
    }
}

/// Model manager with hot-reload support
pub struct ModelManager {
    models: Arc<RwLock<std::collections::HashMap<String, Arc<ModelRunner>>>>,
    production_version: Arc<RwLock<String>>,
    models_dir: String,
}

impl ModelManager {
    /// Create new model manager
    pub async fn new(models_dir: &str) -> AppResult<Self> {
        let manager = Self {
            models: Arc::new(RwLock::new(std::collections::HashMap::new())),
            production_version: Arc::new(RwLock::new("transformer".to_string())),
            models_dir: models_dir.to_string(),
        };
        
        // Load transformer model
        info!("Loading default transformer model from {}", models_dir);
        manager.load_model("transformer").await?;
        
        Ok(manager)
    }

    /// Load a specific model version
    pub async fn load_model(&self, version: &str) -> AppResult<()> {
        // Support both versioned and default model names
        let model_path = if version == "transformer" || version == "1.0.0" {
            format!("{}/diarization_transformer_optimized.onnx", self.models_dir)
        } else {
            format!("{}/diarization_model_v{}.onnx", self.models_dir, version)
        };
        
        let runner = ModelRunner::load(&model_path, version)?;
        
        let mut models = self.models.write().await;
        models.insert(version.to_string(), Arc::new(runner));
        
        info!("Model {} loaded and cached from {}", version, model_path);
        
        Ok(())
    }

    /// Get production model
    pub async fn get_production_model(&self) -> AppResult<Arc<ModelRunner>> {
        let version = self.production_version.read().await;
        let models = self.models.read().await;
        
        models
            .get(version.as_str())
            .cloned()
            .ok_or_else(|| AppError::ModelNotFound(version.to_string()))
    }

    /// Set production model version
    pub async fn set_production_version(&self, version: &str) -> AppResult<()> {
        // Ensure model is loaded
        if !self.models.read().await.contains_key(version) {
            self.load_model(version).await?;
        }
        
        let mut prod_version = self.production_version.write().await;
        *prod_version = version.to_string();
        
        info!("Production model set to version {}", version);
        
        Ok(())
    }

    /// Check if models are ready
    pub async fn is_ready(&self) -> bool {
        !self.models.read().await.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn test_transformer_inference() {
        // Load test audio
        let audio_bytes = fs::read("../voiceflow-ml/test_audio_f32.bin")
            .expect("Test audio file not found - run generate_test_audio.py first");
        
        // Convert bytes to f32
        let audio: Vec<f32> = audio_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        println!("Loaded {} audio samples ({:.2}s @ 16kHz)", 
                 audio.len(), audio.len() as f32 / 16000.0);
        
        // Load model
        let runner = ModelRunner::load(
            "models/diarization_transformer_optimized.onnx",
            "transformer"
        ).expect("Failed to load model");
        
        // Run inference
        let result = runner.run_inference(&audio)
            .expect("Inference failed");
        
        println!("Inference results:");
        println!("  Speaker 1: {:.2}%", result[0] * 100.0);
        println!("  Speaker 2: {:.2}%", result[1] * 100.0);
        
        // Validate output
        assert_eq!(result.len(), 2, "Should return 2 speaker probabilities");
        assert!(result[0] >= 0.0 && result[0] <= 1.0, "Speaker 1 probability should be in [0, 1]");
        assert!(result[1] >= 0.0 && result[1] <= 1.0, "Speaker 2 probability should be in [0, 1]");
        
        // Probabilities should roughly sum to 1 (with softmax)
        let sum = result[0] + result[1];
        assert!((sum - 1.0).abs() < 0.1, 
                "Probabilities should sum to ~1.0, got {:.4}", sum);
        
        println!("✅ All assertions passed!");
    }
}
