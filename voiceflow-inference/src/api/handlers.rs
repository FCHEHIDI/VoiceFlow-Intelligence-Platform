//! HTTP API handlers for Axum.

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{error, info};

use crate::error::{AppError, AppResult};
use crate::metrics::{
    gather_metrics, INFERENCE_ERRORS_TOTAL, INFERENCE_REQUESTS_TOTAL, WEBSOCKET_CONNECTIONS_ACTIVE,
};
use crate::streaming::{
    clustering::OnlineClusterer, handle_websocket, sliding_window::sliding_window,
};
use crate::AppState;

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
            Json(serde_json::json!({"ready": true, "models": "loaded"})),
        )
            .into_response()
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"ready": false, "models": "not loaded"})),
        )
            .into_response()
    }
}

/// Inference request model
#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub model_version: Option<String>,
    #[serde(default = "default_window_secs")]
    pub window_secs: f64,
    #[serde(default = "default_hop_secs")]
    pub hop_secs: f64,
}

fn default_window_secs() -> f64 {
    3.0
}
fn default_hop_secs() -> f64 {
    1.0
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
    pub total_speakers: usize,
}

/// POST /infer — runs the full diarization pipeline (sliding window + embedding + clustering).
pub async fn inference_handler(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> AppResult<Json<InferenceResponse>> {
    let start = Instant::now();
    INFERENCE_REQUESTS_TOTAL.inc();

    if request.audio.is_empty() {
        INFERENCE_ERRORS_TOTAL.inc();
        return Err(AppError::InvalidInput(
            "audio must be non-empty".to_string(),
        ));
    }
    if request.sample_rate == 0 {
        INFERENCE_ERRORS_TOTAL.inc();
        return Err(AppError::InvalidInput(
            "sample_rate must be > 0".to_string(),
        ));
    }

    info!(
        "Inference request: {} samples ({:.2}s @ {}Hz)",
        request.audio.len(),
        request.audio.len() as f64 / request.sample_rate as f64,
        request.sample_rate
    );

    let model = state.model_manager.get_production_model().await?;
    let windows = sliding_window(
        &request.audio,
        request.window_secs,
        request.hop_secs,
        request.sample_rate,
    );

    let mut clusterer = OnlineClusterer::new();
    for (start_sec, end_sec, win) in windows {
        let embedding = model.run_embedding(&win).await.inspect_err(|_| {
            INFERENCE_ERRORS_TOTAL.inc();
        })?;
        clusterer.add_embedding_at(embedding, start_sec, end_sec);
    }

    let segments: Vec<Segment> = clusterer
        .get_segments()
        .into_iter()
        .map(|(s, e, sid)| Segment {
            start: s,
            end: e,
            speaker_id: format!("speaker_{}", sid),
            confidence: 1.0,
        })
        .collect();

    let total_speakers = clusterer.num_speakers();
    let latency_ms = start.elapsed().as_millis() as u64;
    info!(
        "Inference completed in {}ms ({} segments, {} speakers)",
        latency_ms,
        segments.len(),
        total_speakers
    );

    Ok(Json(InferenceResponse {
        segments,
        latency_ms,
        model_version: model.version().to_string(),
        total_speakers,
    }))
}

/// Metrics endpoint (Prometheus format)
pub async fn metrics_handler() -> String {
    gather_metrics()
}

/// WebSocket handler
pub async fn websocket_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_websocket_connection(socket, state))
}

async fn handle_websocket_connection(socket: WebSocket, state: AppState) {
    WEBSOCKET_CONNECTIONS_ACTIVE.inc();
    info!("New WebSocket connection established");
    if let Err(e) = handle_websocket(socket, state).await {
        error!("WebSocket error: {}", e);
    }
    WEBSOCKET_CONNECTIONS_ACTIVE.dec();
    info!("WebSocket connection closed");
}
