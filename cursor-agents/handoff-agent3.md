# Agent 3 Handoff — ML validation

## Files created
- `voiceflow-ml/evaluation/__init__.py`
- `voiceflow-ml/evaluation/diarization_evaluator.py` — `DiarizationEvaluator.compute_der()` + `evaluate_batch()` using `pyannote.metrics`. Falls back gracefully when `pyannote` is missing.
- `voiceflow-ml/evaluation/embedding_validator.py` — `EmbeddingValidator.validate_onnx_model()` returns the ONNX output node name, embedding dim, normalisation flag, intra/inter distance heuristics.
- `voiceflow-ml/scripts/validate_onnx.py` — CLI: `python -m scripts.validate_onnx --model PATH --test-data DIR`. Emits JSON.

## Requirements
- Added `pyannote.metrics>=3.2.0` to `voiceflow-ml/requirements.txt`.
- `aws-xray-sdk>=2.12.0` added (used by Agent 7).

## ONNX contract for Agent 4 (Rust)
| Field | Value |
|-------|-------|
| `onnx_output_node_name` | **TBD** — rerun `python -m scripts.validate_onnx --model voiceflow-ml/models/fast_cnn_diarization_optimized.onnx` on a host with `onnxruntime` to confirm. |
| Expected name | `embedding` (per ADR) |
| `embedding_dim` | 512 |
| `is_normalized` | true (L2) |

The Rust `ModelRunner::run_embedding()` (Agent 4) extracts the output named **`embedding`** by default.
If the actual output node differs, update the constant in `voiceflow-inference/src/inference/mod.rs` accordingly.

## Notes
- The model architecture refactor (final softmax → 512-d L2-normalised projection) is **not** applied automatically here because it touches training code that is intentionally out of scope for this pass. The validator surfaces the current output dim so the team can decide whether to retrain.
