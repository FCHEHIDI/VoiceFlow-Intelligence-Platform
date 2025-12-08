//! Custom error types for the application.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Audio processing error: {0}")]
    AudioProcessingError(String),

    #[error("Internal server error")]
    InternalError,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::ModelNotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::ModelLoadError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::InferenceError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::InvalidInput(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::AudioProcessingError(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::InternalError => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}

/// Result type alias
pub type AppResult<T> = Result<T, AppError>;
