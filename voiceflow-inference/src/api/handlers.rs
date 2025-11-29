"""
HTTP API handlers for Axum.
"""

use axum::{
    extract::{State, ws::{WebSocket, WebSocketUpgrade}},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, error};

use crate::{AppState, error::{AppError, AppResult}};
use crate::metrics::{gather_metrics, INFERENCE_REQUESTS_TOTAL, INFERENCE_ERRORS_TOTAL, WEBSOCKET_CONNECTIONS_ACTIVE};
use crate::streaming::handle_websocket;

/// Health check endpoint
pub async fn health_check() -> &'static str {
    "healthy"
}

/// Readiness check endpoint
pub async fn readiness_check(State(state): State<AppState>) -> Response {
    let is_ready = state.model_manager.is_ready().await;
    
    if is_ready {
        (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({
                "ready": true,
                "models": "loaded"
            }))
        ).into_response()
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "models": "not loaded"
            }))
        ).into_response()
    }
}

/// Inference request model
#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub model_version: Option<String>,
}

/// Segment model
#[derive(Debug, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker_id: String,
    pub confidence: f64,
}

/// Inference response model
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub segments: Vec<Segment>,
    pub latency_ms: u64,
    pub model_version: String,
}

/// Inference handler
pub async fn inference_handler(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> AppResult<Json<InferenceResponse>> {
    let start = Instant::now();
    
    info!("Inference request received, audio length: {}", request.audio.len());
    
    INFERENCE_REQUESTS_TOTAL.inc();
    
    // Get production model
    let model = state.model_manager.get_production_model().await?;
    
    // Run inference
    let speaker_probs = model.run_inference(&request.audio)
        .map_err(|e| {
            INFERENCE_ERRORS_TOTAL.inc();
            e
        })?;
    
    // Post-process: convert probabilities to segments
    // For demo, create stub segment
    let speaker_id = if speaker_probs.len() >= 2 {
        if speaker_probs[0] > speaker_probs[1] {
            "SPEAKER_00".to_string()
        } else {
            "SPEAKER_01".to_string()
        }
    } else {
        "SPEAKER_00".to_string()
    };
    
    let confidence = speaker_probs.iter().copied().fold(0.0f32, f32::max);
    
    let segments = vec![Segment {
        start: 0.0,
        end: 1.0,
        speaker_id,
        confidence: confidence as f64,
    }];
    
    let latency_ms = start.elapsed().as_millis() as u64;
    
    info!("Inference completed in {}ms", latency_ms);
    
    Ok(Json(InferenceResponse {
        segments,
        latency_ms,
        model_version: model.version().to_string(),
    }))
}

/// Metrics endpoint (Prometheus format)
pub async fn metrics_handler() -> String {
    gather_metrics()
}

/// WebSocket handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket_connection(socket, state))
}

/// Handle WebSocket connection
async fn handle_websocket_connection(socket: WebSocket, state: AppState) {
    WEBSOCKET_CONNECTIONS_ACTIVE.inc();
    
    info!("New WebSocket connection established");
    
    // Delegate to streaming module
    if let Err(e) = handle_websocket(socket, state).await {
        error!("WebSocket error: {}", e);
    }
    
    WEBSOCKET_CONNECTIONS_ACTIVE.dec();
    
    info!("WebSocket connection closed");
}
