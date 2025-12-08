//! Prometheus metrics setup and registry.

use lazy_static::lazy_static;
use prometheus::{register_histogram, register_counter, register_gauge, Histogram, Counter, Gauge, TextEncoder, Encoder};

lazy_static! {
    // Latency metrics
    pub static ref INFERENCE_LATENCY: Histogram = register_histogram!(
        "inference_latency_seconds",
        "Time spent performing inference"
    ).unwrap();

    // Counter metrics
    pub static ref INFERENCE_REQUESTS_TOTAL: Counter = register_counter!(
        "inference_requests_total",
        "Total number of inference requests"
    ).unwrap();

    pub static ref INFERENCE_ERRORS_TOTAL: Counter = register_counter!(
        "inference_errors_total",
        "Total number of inference errors"
    ).unwrap();

    // Gauge metrics
    pub static ref WEBSOCKET_CONNECTIONS_ACTIVE: Gauge = register_gauge!(
        "websocket_connections_active",
        "Number of active WebSocket connections"
    ).unwrap();

    pub static ref MODEL_LOAD_TIME: Histogram = register_histogram!(
        "model_load_duration_seconds",
        "Time to load ONNX model"
    ).unwrap();
}

/// Setup metrics registry
pub fn setup_metrics() {
    // Metrics are registered via lazy_static
    tracing::info!("Metrics registry initialized");
}

/// Gather and encode metrics in Prometheus format
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
