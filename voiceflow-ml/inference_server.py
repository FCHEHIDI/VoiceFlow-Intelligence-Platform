#!/usr/bin/env python3
"""
Simple FastAPI inference server for testing performance.
Simulates the Rust server but in Python for quick validation.

Performance targets:
- Model inference: <10ms P99
- End-to-end: <100ms P99
"""

import time
import logging
from typing import List
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="VoiceFlow Inference Server",
    description="High-performance speaker diarization inference",
    version="1.0.0"
)

# Global model session
MODEL_SESSION = None
MODEL_PATH = "models/fast_cnn_diarization_optimized.onnx"

# Metrics
inference_count = 0
total_inference_time = 0.0
latencies = []


class InferenceRequest(BaseModel):
    audio: List[float]
    sample_rate: int = 48000


class InferenceResponse(BaseModel):
    speakers: List[float]
    latency_ms: float
    timestamp: float


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global MODEL_SESSION
    
    logger.info(f"Loading model: {MODEL_PATH}")
    try:
        MODEL_SESSION = ort.InferenceSession(
            MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Provider: {MODEL_SESSION.get_providers()[0]}")
        logger.info(f"   Input: {MODEL_SESSION.get_inputs()[0].name} {MODEL_SESSION.get_inputs()[0].shape}")
        logger.info(f"   Output: {MODEL_SESSION.get_outputs()[0].name} {MODEL_SESSION.get_outputs()[0].shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VoiceFlow Inference Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_SESSION is not None,
        "timestamp": time.time()
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if MODEL_SESSION is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "ready",
        "model_path": MODEL_PATH,
        "provider": MODEL_SESSION.get_providers()[0]
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    global inference_count, total_inference_time, latencies
    
    avg_latency = total_inference_time / max(inference_count, 1)
    
    # Calculate percentiles
    if latencies:
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    else:
        p50 = p95 = p99 = 0.0
    
    metrics_text = f"""# HELP inference_requests_total Total number of inference requests
# TYPE inference_requests_total counter
inference_requests_total {inference_count}

# HELP inference_latency_seconds Inference latency in seconds
# TYPE inference_latency_seconds summary
inference_latency_seconds_sum {total_inference_time}
inference_latency_seconds_count {inference_count}
inference_latency_seconds{{quantile="0.5"}} {p50 / 1000}
inference_latency_seconds{{quantile="0.95"}} {p95 / 1000}
inference_latency_seconds{{quantile="0.99"}} {p99 / 1000}

# HELP inference_latency_milliseconds Inference latency in milliseconds
# TYPE inference_latency_milliseconds gauge
inference_latency_ms_avg {avg_latency}
inference_latency_ms_p50 {p50}
inference_latency_ms_p95 {p95}
inference_latency_ms_p99 {p99}
"""
    
    return JSONResponse(content=metrics_text, media_type="text/plain")


@app.post("/infer", response_model=InferenceResponse)
async def inference_handler(request: InferenceRequest):
    """
    Perform speaker diarization inference
    
    Expected performance:
    - Model inference: <10ms P99
    - Including overhead: <20ms P99
    """
    global inference_count, total_inference_time, latencies
    
    if MODEL_SESSION is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if len(request.audio) != 48000:
            raise HTTPException(
                status_code=400,
                detail=f"Audio must be exactly 48000 samples (1 second at 48kHz), got {len(request.audio)}"
            )
        
        # Prepare input
        start_time = time.perf_counter()
        audio_input = np.array(request.audio, dtype=np.float32).reshape(1, -1)
        
        # Run inference
        input_name = MODEL_SESSION.get_inputs()[0].name
        outputs = MODEL_SESSION.run(None, {input_name: audio_input})
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Update metrics
        inference_count += 1
        total_inference_time += latency_ms
        latencies.append(latency_ms)
        
        # Keep only last 1000 latencies
        if len(latencies) > 1000:
            latencies = latencies[-1000:]
        
        # Log performance
        if inference_count % 100 == 0:
            logger.info(f"Processed {inference_count} requests, Avg latency: {total_inference_time/inference_count:.2f}ms")
        
        return InferenceResponse(
            speakers=outputs[0][0].tolist(),
            latency_ms=latency_ms,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_handler(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio
    
    Client sends: {"audio": [float...], "sample_rate": 48000}
    Server responds: {"speakers": [float...], "latency_ms": float}
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_json()
            
            # Validate
            if "audio" not in data:
                await websocket.send_json({"error": "Missing 'audio' field"})
                continue
            
            # Prepare input
            start_time = time.perf_counter()
            audio = np.array(data["audio"], dtype=np.float32).reshape(1, -1)
            
            # Pad or truncate to 48000 samples
            if audio.shape[1] < 48000:
                audio = np.pad(audio, ((0, 0), (0, 48000 - audio.shape[1])))
            elif audio.shape[1] > 48000:
                audio = audio[:, :48000]
            
            # Run inference
            input_name = MODEL_SESSION.get_inputs()[0].name
            outputs = MODEL_SESSION.run(None, {input_name: audio})
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Send response
            await websocket.send_json({
                "speakers": outputs[0][0].tolist(),
                "latency_ms": latency_ms,
                "timestamp": time.time()
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    import sys
    
    # Override model path if provided
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    
    print(f"""
==================================================================
    VoiceFlow Inference Server - Performance Testing
==================================================================

Model: {MODEL_PATH}
Port: 3000

Performance Targets:
  * Model Inference: <10ms P99
  * End-to-End: <100ms P99

Endpoints:
  * GET  /              - Service info
  * GET  /health        - Health check
  * GET  /ready         - Readiness check
  * GET  /metrics       - Prometheus metrics
  * POST /infer         - Inference endpoint
  * WS   /ws/stream     - WebSocket streaming

Starting server...
""")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3000,
        log_level="info"
    )
