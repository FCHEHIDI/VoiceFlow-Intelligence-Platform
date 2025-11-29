"""
Main entry point for Rust inference engine.
"""

use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tower_http::cors::{CorsLayer, Any};
use tracing::{info, Level};
use tracing_subscriber;

mod api;
mod inference;
mod streaming;
mod metrics;
mod config;
mod error;

use crate::api::handlers;
use crate::inference::ModelManager;
use crate::metrics::setup_metrics;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .json()
        .init();

    info!("Starting VoiceFlow Inference Engine v1.0.0");

    // Setup metrics
    setup_metrics();

    // Initialize model manager
    let model_manager = Arc::new(ModelManager::new("../models").await?);
    
    // Create application state
    let app_state = AppState { model_manager };

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(handlers::health_check))
        .route("/ready", get(handlers::readiness_check))
        .route("/infer", post(handlers::inference_handler))
        .route("/metrics", get(handlers::metrics_handler))
        .route("/ws/stream", get(handlers::websocket_handler))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(app_state);

    // Bind server
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Listening on {}", addr);

    // Start server with graceful shutdown
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

/// Root endpoint
async fn root() -> &'static str {
    "VoiceFlow Inference Engine v1.0.0"
}

/// Graceful shutdown handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("Received terminate signal");
        },
    }

    info!("Starting graceful shutdown...");
}
