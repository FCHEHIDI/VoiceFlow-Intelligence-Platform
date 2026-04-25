//! ONNX Runtime integration for model inference.

use ndarray::Array1;
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Value, execution_providers::CPUExecutionProvider};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;
use tracing::{info, warn};

use crate::error::{AppError, AppResult};
use crate::metrics::{INFERENCE_LATENCY, MODEL_LOAD_TIME};

/// Output node name for embedding extraction (per Agent 3 handoff).
pub const EMBEDDING_OUTPUT_NODE: &str = "embedding";

/// Model runner with ONNX Runtime
pub struct ModelRunner {
    session: Arc<RwLock<Session>>,
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
            // Enable CPU execution (GPU can be added later)
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        
        let duration = start.elapsed();
        MODEL_LOAD_TIME.observe(duration.as_secs_f64());
        
        info!("Model loaded in {:?}", duration);
        
        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            version: version.to_string(),
        })
    }

    /// Run inference on raw audio samples
    /// 
    /// Input: audio samples (16kHz mono, f32)
    /// Output: speaker probabilities [speaker_1, speaker_2]
    pub async fn run_inference(&self, audio: &[f32]) -> AppResult<Vec<f32>> {
        let start = Instant::now();
        
        // Transformer model expects: [batch_size, audio_length]
        // Input shape: [1, audio.len()]
        let batch_size = 1;
        let audio_length = audio.len();
        
        info!("Running inference on {} samples ({:.2}s audio)", 
              audio_length, audio_length as f32 / 16000.0);
        
        let input_shape = vec![batch_size, audio_length];
        let input_data = audio.to_vec();
        
        let input_tensor = Value::from_array((input_shape.as_slice(), input_data))
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        // Run inference with named input
        let mut session = self.session.write().await;
        let outputs = session.run(ort::inputs!["audio" => input_tensor])
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        // Extract speaker probabilities (shape: [batch_size, num_speakers])
        let output = outputs["speaker_probabilities"]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        let result: Vec<f32> = output.1.to_vec();
        
        // Record latency
        let duration = start.elapsed();
        INFERENCE_LATENCY.observe(duration.as_secs_f64());
        
        info!(
            "Inference completed in {:?} - Speaker 1: {:.2}%, Speaker 2: {:.2}%",
            duration,
            result.first().unwrap_or(&0.0) * 100.0,
            result.get(1).unwrap_or(&0.0) * 100.0
        );
        
        Ok(result)
    }

    /// Run embedding extraction for diarization streaming.
    ///
    /// Accepts raw PCM samples (16 kHz mono) and returns an L2-normalised vector
    /// extracted from the ONNX node named [`EMBEDDING_OUTPUT_NODE`]. Falls back
    /// to the first output of the session if the named node is absent.
    pub async fn run_embedding(&self, audio: &[f32]) -> AppResult<Array1<f32>> {
        let start = Instant::now();
        let batch_size = 1usize;
        let audio_length = audio.len();

        let input_shape = vec![batch_size, audio_length];
        let input_tensor = Value::from_array((input_shape.as_slice(), audio.to_vec()))
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let mut session = self.session.write().await;
        let outputs = session
            .run(ort::inputs!["audio" => input_tensor])
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let extracted: Vec<f32> = match outputs.get(EMBEDDING_OUTPUT_NODE) {
            Some(value) => value
                .try_extract_tensor::<f32>()
                .map_err(|e| AppError::InferenceError(e.to_string()))?
                .1
                .to_vec(),
            None => {
                warn!(
                    "ONNX model does not expose '{}' output — falling back to first output",
                    EMBEDDING_OUTPUT_NODE
                );
                let mut iter = outputs.iter();
                let (_name, value) = iter.next().ok_or_else(|| {
                    AppError::InferenceError("ONNX model produced no outputs".to_string())
                })?;
                value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AppError::InferenceError(e.to_string()))?
                    .1
                    .to_vec()
            }
        };

        let mut emb = Array1::from(extracted);
        l2_normalize(&mut emb);

        let duration = start.elapsed();
        INFERENCE_LATENCY.observe(duration.as_secs_f64());
        Ok(emb)
    }

    /// Get model version
    pub fn version(&self) -> &str {
        &self.version
    }
}

fn l2_normalize(v: &mut Array1<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.mapv_inplace(|x| x / norm);
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

    /// End-to-end test against a real ONNX model — only runs when the model and
    /// reference audio are available locally. Run with: `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn transformer_inference_smoke() {
        let audio_bytes = fs::read("../voiceflow-ml/test_audio_f32.bin")
            .expect("Test audio file not found - run generate_test_audio.py first");
        let audio: Vec<f32> = audio_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let runner = ModelRunner::load(
            "models/diarization_transformer_optimized.onnx",
            "transformer",
        )
        .expect("Failed to load model");

        let result = runner.run_inference(&audio).await.expect("Inference failed");
        assert!(result.iter().all(|p| (0.0..=1.0).contains(p)));
    }
}
