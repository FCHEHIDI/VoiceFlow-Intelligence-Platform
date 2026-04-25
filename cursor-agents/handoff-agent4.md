# Agent 4 Handoff — Rust Pipeline

## Files modified / created
- `voiceflow-inference/src/lib.rs` — new library crate exposing `api`, `config`, `error`, `inference`, `metrics`, `streaming`, plus `AppState`. The binary `main.rs` was reduced to a thin entry point that depends on the library.
- `voiceflow-inference/src/inference/mod.rs` — added `run_embedding(audio: &[f32]) -> Array1<f32>` (L2-normalised, pulls the ONNX node `embedding` with first-output fallback). Constant `EMBEDDING_OUTPUT_NODE`.
- `voiceflow-inference/src/streaming/sliding_window.rs` — new `sliding_window(audio, window_secs, hop_secs, sample_rate)` returning `(start, end, samples)` tuples + 3 unit tests.
- `voiceflow-inference/src/streaming/clustering.rs` — added `add_embedding_at`, `get_segments`, `smooth_labels`, `Default`, timestamp tracking, `is_multiple_of` / `or_default` clean-ups.
- `voiceflow-inference/src/streaming/mod.rs` — pipeline rewritten to use `sliding_window` + `run_embedding` + `OnlineClusterer`; emits NDJSON `{"type":"segment", ...}` events plus `{"type":"end_stream", ...}` finaliser.
- `voiceflow-inference/src/api/handlers.rs` — `POST /infer` now drives the full pipeline and returns `{ segments, total_speakers, latency_ms, model_version }`.
- `voiceflow-inference/tests/inference_tests.rs` — replaced the broken Python-docstring test file with a compilable integration test suite for `AudioBuffer`.

## Verification (Windows, this machine)
```powershell
$env:JWT_SECRET_KEY = "<32+ chars>"; cargo test --all-targets        # 8 passed, 1 ignored
$env:JWT_SECRET_KEY = "<32+ chars>"; cargo clippy --all-targets -- -D warnings  # 0 warning
```

## Latency
The full diarization pipeline is exercised by `cargo test`; an actual model run is gated behind `#[ignore]`. To collect P99 latency, run the binary against the ONNX model and let Prometheus scrape `inference_latency_seconds`.

## Hand-off to Agents 7 / 8
- `OnlineClusterer::add_embedding_at(emb, start, end)` is the canonical streaming entry point.
- `OnlineClusterer::get_segments()` returns smoothed `(start_seconds, end_seconds, speaker_id)` triples and is the primitive used by the Rust tests in Agent 8.
- The library crate exposes `voiceflow_inference::streaming::{clustering::OnlineClusterer, sliding_window::sliding_window, AudioBuffer}` and `voiceflow_inference::inference::{ModelManager, ModelRunner, EMBEDDING_OUTPUT_NODE}` for tests.
