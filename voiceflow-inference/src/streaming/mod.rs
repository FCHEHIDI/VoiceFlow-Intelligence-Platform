//! WebSocket streaming handler for real-time audio processing.

pub mod clustering;
pub mod sliding_window;

use axum::extract::ws::{Message, WebSocket};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{error, info, warn};

use crate::api::Segment;
use crate::error::AppResult;
use crate::metrics::INFERENCE_LATENCY;
use crate::streaming::clustering::OnlineClusterer;
use crate::streaming::sliding_window::sliding_window;
use crate::AppState;

const SAMPLE_RATE: u32 = 16_000;
const WINDOW_SECS: f64 = 3.0;
const HOP_SECS: f64 = 1.0;
const PROCESS_THRESHOLD_SAMPLES: usize = (SAMPLE_RATE as usize) * 3; // 3s

/// Streaming request message
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamRequest {
    #[serde(rename = "audio_chunk")]
    AudioChunk {
        data: Vec<f32>,
        timestamp: Option<f64>,
        sequence: usize,
    },
    #[serde(rename = "end_stream")]
    EndStream,
}

/// Streaming response message (NDJSON-friendly).
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamResponse {
    #[serde(rename = "segment")]
    Segment {
        start: f64,
        end: f64,
        speaker: String,
        confidence: f64,
    },
    #[serde(rename = "diarization_result")]
    DiarizationResult {
        segments: Vec<Segment>,
        latency_ms: u64,
        sequence: usize,
    },
    #[serde(rename = "end_stream")]
    EndStream {
        total_speakers: usize,
        duration_seconds: f64,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

/// Handle WebSocket connection
pub async fn handle_websocket(mut socket: WebSocket, state: AppState) -> AppResult<()> {
    info!("WebSocket handler started");
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(SAMPLE_RATE as usize * 10);
    let mut clusterer = OnlineClusterer::new();
    let mut total_seconds: f64 = 0.0;
    let mut emitted_segments: usize = 0;

    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => match serde_json::from_str::<StreamRequest>(&text) {
                Ok(StreamRequest::AudioChunk { data, sequence, .. }) => {
                    let start = Instant::now();
                    audio_buffer.extend_from_slice(&data);

                    if audio_buffer.len() < PROCESS_THRESHOLD_SAMPLES {
                        continue;
                    }

                    let model = state.model_manager.get_production_model().await?;
                    let windows = sliding_window(&audio_buffer, WINDOW_SECS, HOP_SECS, SAMPLE_RATE);

                    for (s, e, win) in &windows {
                        match model.run_embedding(win).await {
                            Ok(emb) => {
                                let speaker = clusterer.add_embedding_at(
                                    emb,
                                    total_seconds + *s,
                                    total_seconds + *e,
                                );
                                let response = StreamResponse::Segment {
                                    start: total_seconds + *s,
                                    end: total_seconds + *e,
                                    speaker: format!("speaker_{}", speaker),
                                    confidence: 1.0,
                                };
                                if let Ok(json) = serde_json::to_string(&response) {
                                    socket.send(Message::Text(json)).await.ok();
                                    emitted_segments += 1;
                                }
                            }
                            Err(err) => {
                                error!("Embedding error: {}", err);
                                let response = StreamResponse::Error {
                                    code: "EMBEDDING_ERROR".to_string(),
                                    message: err.to_string(),
                                };
                                if let Ok(json) = serde_json::to_string(&response) {
                                    socket.send(Message::Text(json)).await.ok();
                                }
                            }
                        }
                    }

                    let latency_ms = start.elapsed().as_millis() as u64;
                    INFERENCE_LATENCY.observe(latency_ms as f64 / 1000.0);
                    info!(
                        "Processed seq={} ({} windows) in {}ms",
                        sequence,
                        windows.len(),
                        latency_ms
                    );

                    // Keep one window of overlap so the next `sliding_window` call covers
                    // the boundary between successive chunks without re-emitting segments.
                    let overlap = SAMPLE_RATE as usize * 2; // 2s overlap given window=3s, hop=1s
                    let advance = audio_buffer.len().saturating_sub(overlap);
                    audio_buffer.drain(..advance);
                    total_seconds += advance as f64 / SAMPLE_RATE as f64;
                }
                Ok(StreamRequest::EndStream) => {
                    info!("End stream received");
                    let response = StreamResponse::EndStream {
                        total_speakers: clusterer.num_speakers(),
                        duration_seconds: total_seconds,
                    };
                    if let Ok(json) = serde_json::to_string(&response) {
                        socket.send(Message::Text(json)).await.ok();
                    }
                    info!(
                        "Emitted {} segments across {} speakers",
                        emitted_segments,
                        clusterer.num_speakers()
                    );
                    break;
                }
                Err(e) => {
                    warn!("Invalid message format: {}", e);
                    let response = StreamResponse::Error {
                        code: "INVALID_MESSAGE".to_string(),
                        message: format!("Failed to parse message: {}", e),
                    };
                    if let Ok(json) = serde_json::to_string(&response) {
                        socket.send(Message::Text(json)).await.ok();
                    }
                }
            },
            Ok(Message::Close(_)) => {
                info!("Client closed connection");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    info!("WebSocket handler completed");
    Ok(())
}

/// Audio buffer for streaming (kept for backwards compatibility / tests)
pub struct AudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
}

impl AudioBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.buffer.len() >= self.capacity {
                self.buffer.remove(0);
            }
            self.buffer.push(sample);
        }
    }

    pub fn get_chunk(&self) -> &[f32] {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}
