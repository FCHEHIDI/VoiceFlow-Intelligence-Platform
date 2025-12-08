//! WebSocket streaming handler for real-time audio processing.

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, warn, error};

use crate::{AppState, error::{AppError, AppResult}};
use crate::api::Segment;
use crate::metrics::INFERENCE_LATENCY;

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

/// Streaming response message
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamResponse {
    #[serde(rename = "diarization_result")]
    DiarizationResult {
        segments: Vec<Segment>,
        latency_ms: u64,
        sequence: usize,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

/// Handle WebSocket connection
pub async fn handle_websocket(mut socket: WebSocket, state: AppState) -> AppResult<()> {
    info!("WebSocket handler started");
    
    // Audio buffer for accumulating chunks
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(16000); // 1 second @ 16kHz
    
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse JSON message
                match serde_json::from_str::<StreamRequest>(&text) {
                    Ok(StreamRequest::AudioChunk { data, sequence, .. }) => {
                        let start = Instant::now();
                        
                        // Append to buffer
                        audio_buffer.extend_from_slice(&data);
                        
                        // Process if buffer has enough data (1 second)
                        if audio_buffer.len() >= 16000 {
                            // Get production model
                            let model = state.model_manager.get_production_model().await?;
                            
                            // Run inference
                            match model.run_inference(&audio_buffer[..16000]).await {
                                Ok(speaker_probs) => {
                                    // Post-process
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
                                    INFERENCE_LATENCY.observe(latency_ms as f64 / 1000.0);
                                    
                                    // Send result
                                    let response = StreamResponse::DiarizationResult {
                                        segments,
                                        latency_ms,
                                        sequence,
                                    };
                                    
                                    let response_json = serde_json::to_string(&response).unwrap();
                                    socket.send(Message::Text(response_json)).await.ok();
                                    
                                    info!("Processed chunk {} in {}ms", sequence, latency_ms);
                                }
                                Err(e) => {
                                    error!("Inference error: {}", e);
                                    let error_response = StreamResponse::Error {
                                        code: "INFERENCE_ERROR".to_string(),
                                        message: e.to_string(),
                                    };
                                    let response_json = serde_json::to_string(&error_response).unwrap();
                                    socket.send(Message::Text(response_json)).await.ok();
                                }
                            }
                            
                            // Clear processed data from buffer
                            audio_buffer.drain(..16000);
                        }
                    }
                    Ok(StreamRequest::EndStream) => {
                        info!("End stream received");
                        break;
                    }
                    Err(e) => {
                        warn!("Invalid message format: {}", e);
                        let error_response = StreamResponse::Error {
                            code: "INVALID_MESSAGE".to_string(),
                            message: format!("Failed to parse message: {}", e),
                        };
                        let response_json = serde_json::to_string(&error_response).unwrap();
                        socket.send(Message::Text(response_json)).await.ok();
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!("Client closed connection");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {
                // Ignore binary, ping, pong messages
            }
        }
    }
    
    info!("WebSocket handler completed");
    Ok(())
}

/// Audio buffer for streaming
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
                self.buffer.remove(0); // Simple FIFO (could use VecDeque for efficiency)
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

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}
