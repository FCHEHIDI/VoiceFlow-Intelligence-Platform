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

    /// Run inference on audio features
    pub fn run_inference(&self, features: &[f32]) -> AppResult<Vec<f32>> {
        let start = Instant::now();
        
        // Prepare input tensor (stub - actual shape depends on model)
        // Expected shape: [batch_size, channels, time_steps, n_mfcc]
        let input_shape = [1, 1, 100, 40];
        
        // Create ONNX tensor
        let input_tensor = Value::from_array(self.session.allocator(), &features)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        // Run inference
        let outputs = self.session.run(vec![input_tensor])
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        // Extract output (speaker probabilities)
        let output = outputs[0]
            .extract_tensor::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        
        let result: Vec<f32> = output.view().iter().copied().collect();
        
        // Record latency
        let duration = start.elapsed();
        INFERENCE_LATENCY.observe(duration.as_secs_f64());
        
        info!("Inference completed in {:?}", duration);
        
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
            production_version: Arc::new(RwLock::new("1.0.0".to_string())),
            models_dir: models_dir.to_string(),
        };
        
        // Load default model (stub for now)
        // manager.load_model("1.0.0").await?;
        
        Ok(manager)
    }

    /// Load a specific model version
    pub async fn load_model(&self, version: &str) -> AppResult<()> {
        let model_path = format!("{}/diarization_model_v{}.onnx", self.models_dir, version);
        
        let runner = ModelRunner::load(&model_path, version)?;
        
        let mut models = self.models.write().await;
        models.insert(version.to_string(), Arc::new(runner));
        
        info!("Model {} loaded and cached", version);
        
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
