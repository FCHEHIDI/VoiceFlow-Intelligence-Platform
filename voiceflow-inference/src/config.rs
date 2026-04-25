//! Configuration management for the inference service.

use serde::{Deserialize, Serialize};
use std::env;
use std::process;

/// Exit early if required secrets are missing.
pub fn validate_required_secrets() {
    for key in ["JWT_SECRET_KEY"] {
        if std::env::var(key).is_err() {
            eprintln!(
                "Required secret '{}' is not set. Generate with: openssl rand -hex 32",
                key
            );
            process::exit(1);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub models_dir: String,
    pub default_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub num_threads: usize,
    pub optimization_level: String,  // Level1, Level2, Level3
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            server: ServerConfig {
                host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("PORT")
                    .unwrap_or_else(|_| "3000".to_string())
                    .parse()
                    .unwrap_or(3000),
            },
            model: ModelConfig {
                models_dir: env::var("MODELS_DIR").unwrap_or_else(|_| "../models".to_string()),
                default_version: env::var("DEFAULT_MODEL_VERSION")
                    .unwrap_or_else(|_| "1.0.0".to_string()),
            },
            inference: InferenceConfig {
                num_threads: env::var("INFERENCE_THREADS")
                    .unwrap_or_else(|_| "4".to_string())
                    .parse()
                    .unwrap_or(4),
                optimization_level: env::var("OPTIMIZATION_LEVEL")
                    .unwrap_or_else(|_| "Level3".to_string()),
            },
        }
    }
}
