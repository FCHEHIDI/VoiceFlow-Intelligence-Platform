# Agent 8 Handoff — Tests & Quality

## Files added

### Python (`voiceflow-ml/`)

| File | Contents |
| --- | --- |
| `tests/conftest.py` (extended) | Adds `mock_job_repo` (`InMemoryJobRepo`), `mock_task_queue`, `mock_redis`, `sample_wav_bytes`, `fake_zip_bytes`, `oversized_wav_path` (streamed to disk), `oversized_content_length`, and a `test_client` fixture that overrides `get_job_repository` / `get_inference_service` with the in-memory fakes. Sets `JWT_SECRET_KEY` / `POSTGRES_PASSWORD` before any app import. |
| `tests/unit/__init__.py` | New package. |
| `tests/unit/test_domain.py` | `BatchJob` state machine (initial PENDING, valid/invalid transitions), `Segment` validation (start<end, non-negative, confidence range, frozen), `ModelVersion` (uuid uniqueness, promote, independent metrics dict), exception hierarchy. |
| `tests/unit/test_services.py` | `InferenceService.submit_batch_job` happy path / strip whitespace / empty-input rejection / optional callback URL. `get_job_status` happy / unknown id raises `JobNotFoundError` / blank id raises `InvalidJobInputError`. State transition helpers. |
| `tests/unit/test_audio_validation.py` | `validate_audio_upload` accepts a generated WAV, rejects ZIP-as-WAV with 400, rejects oversized payload with 413. `content_length_middleware` 413 on oversize, pass-through within limit / outside prefix, 400 on invalid header, factory parameter wiring. Skips libmagic-only assertions when libmagic is absent (Windows default). |
| `tests/integration/__init__.py` | New package. |
| `tests/integration/test_api_inference.py` | `POST /api/v1/inference/batch` → 202 with `pending`, `GET .../{id}` → 200, unknown id → 404, missing/empty audio_path → 422, sync upload accepts WAV / rejects ZIP / 413 on oversize, `GET /health` → 200 with `service` and `status`. |
| `tests/fixtures/__init__.py` | New package. |
| `tests/fixtures/generate_audio.py` | CLI script that writes `tests/fixtures/audio/sample_16k_mono.wav` (5 s silence, 16 kHz mono 16-bit PCM) using stdlib `wave`. Run via `python tests/fixtures/generate_audio.py`. |
| `tests/fixtures/audio/sample_16k_mono.wav` | Materialised fixture (160 044 bytes). |

### Rust (`voiceflow-inference/`)

`tests/inference_tests.rs` — appended (existing `AudioBuffer` tests preserved):

* `sliding_window_tests::ten_seconds_window3_hop1_yields_eight_windows`
* `sliding_window_tests::two_seconds_with_three_second_window_yields_no_windows`
* `sliding_window_tests::exactly_three_seconds_yields_a_single_window`
* `clustering_tests::same_embedding_added_twice_is_same_speaker`
* `clustering_tests::orthogonal_embeddings_produce_distinct_speakers`
* `clustering_tests::three_alternating_embeddings_produce_two_speakers`

## `pytest.ini`

Already had `asyncio_mode = auto`, registered markers (`unit`, `integration`, `slow`, `hardware`), and `--cov-fail-under=70`. Verified intact.

## Run results (final)

```
$ Set-Location voiceflow-ml
$ pytest tests/unit tests/integration -v -m "not hardware and not slow"
...
53 passed, 3 warnings in 2.49s
Total coverage: 84.20%   (>= 70% required)

$ Set-Location voiceflow-inference
$ cargo test --lib
6 passed; 0 failed; 1 ignored        (transformer_inference_smoke is gated on a real ONNX model)

$ cargo test --test inference_tests
8 passed; 0 failed; 0 ignored
```

Total: 53 Python tests + 14 Rust tests, 100% green on this Windows host.

## Skips on Windows

* `tests/unit/test_audio_validation.py` — libmagic-specific assertions are wrapped in `@pytest.mark.skipif(not libmagic_available)`. `validate_audio_upload` keeps its fallback signature detection so the production path still runs.
* `voiceflow-inference::inference::tests::transformer_inference_smoke` — `#[ignore]` because it needs a real `transformer_*.onnx` artifact that is not in the repo.
