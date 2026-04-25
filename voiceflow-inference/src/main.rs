//! Binary entry point — wires the library crate into an Axum service.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::{get, post}, Router};
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, Level};

use voiceflow_inference::api::handlers;
use voiceflow_inference::config::validate_required_secrets;
use voiceflow_inference::inference::ModelManager;
use voiceflow_inference::metrics::setup_metrics;
use voiceflow_inference::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    validate_required_secrets();

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .json()
        .init();

    info!("Starting VoiceFlow Inference Engine v1.0.0");

    setup_metrics();

    let models_dir = std::env::var("MODELS_DIR")
        .unwrap_or_else(|_| "../voiceflow-ml/models".to_string());
    info!("Using models directory: {}", models_dir);

    let model_manager = Arc::new(ModelManager::new(&models_dir).await?);
    let app_state = AppState { model_manager };

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(handlers::health_check))
        .route("/ready", get(handlers::readiness_check))
        .route("/infer", post(handlers::inference_handler))
        .route("/metrics", get(handlers::metrics_handler))
        .route("/ws/stream", get(handlers::websocket_handler))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

async fn root() -> &'static str {
    "VoiceFlow Inference Engine v1.0.0"
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
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
        _ = ctrl_c => info!("Received Ctrl+C signal"),
        _ = terminate => info!("Received terminate signal"),
    }

    info!("Starting graceful shutdown...");
}
