//! VoiceFlow inference engine — library crate.
//!
//! The binary entry point (`main.rs`) wires this library into an Axum service.
//! Integration tests (under `tests/`) consume the public types exported here.

pub mod api;
pub mod config;
pub mod error;
pub mod inference;
pub mod metrics;
pub mod streaming;

use std::sync::Arc;

use crate::inference::ModelManager;

/// Application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
}
