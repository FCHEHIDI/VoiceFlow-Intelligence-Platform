# VoiceFlow — Full Automated Refactoring

You are a senior full-stack engineer. Execute all 8 agents below **sequentially and completely**, without stopping for confirmation. After each agent, create the handoff file and move immediately to the next one.

**Working directory**: `voiceflow-ml/` and `voiceflow-inference/` (monorepo root).

---

## AGENT 1 — Security Foundation

Read `cursor-agents/agent1-security-foundation.md` in full, then execute every task it describes:

1. Audit all Python files for hardcoded secrets (`SECRET_KEY`, `password`, `redis://`, `postgresql://`). Move every secret to `core/config.py` via `pydantic-settings` + env vars.
2. Create `voiceflow-ml/api/middleware/input_validation.py`:
   - Validate audio file magic bytes (RIFF for WAV, must not be ZIP/PDF/EXE)
   - Reject files > 100 MB with HTTP 413
   - Reject non-audio MIME types with HTTP 400
   - Use `python-magic` for magic byte detection
3. Register the middleware in `api/main.py`.
4. Verify: `grep -r "SECRET_KEY\s*=" voiceflow-ml/ --include="*.py"` returns zero results.

Create `cursor-agents/handoff-agent1.md` listing what was done.

---

## AGENT 2 — Python Clean Architecture

Read `cursor-agents/agent2-python-clean-arch.md` in full, then execute every task:

1. Create `voiceflow-ml/domain/job.py` — `BatchJob` dataclass + `JobStatus` enum.
2. Create `voiceflow-ml/domain/segment.py` — `Segment`, `Speaker` dataclasses.
3. Create `voiceflow-ml/domain/model_version.py` — `ModelVersion` dataclass.
4. Create `voiceflow-ml/domain/__init__.py`.
5. Create `voiceflow-ml/repositories/base.py` — abstract `BaseRepository[T]`.
6. Create `voiceflow-ml/repositories/job_repository.py` — implements `BaseRepository[BatchJob]`.
7. Create `voiceflow-ml/services/inference_service.py` — `InferenceService` with `submit_batch_job()` and `get_job_status()`.
8. Create `voiceflow-ml/workers/celery_app.py` + `voiceflow-ml/workers/inference_tasks.py`.
9. Refactor `api/routes/inference.py` to thin controller (≤ 40 lines, zero business logic).
10. Add `celery[redis]>=5.3.0` and `httpx>=0.27.0` to `requirements.txt`.
11. Verify: `grep -r "from sqlalchemy" voiceflow-ml/domain/` returns zero results.

Create `cursor-agents/handoff-agent2.md`.

---

## AGENT 3 — ML Model Validation

Read `cursor-agents/agent3-ml-validation.md` in full, then execute every task:

1. Create `voiceflow-ml/evaluation/__init__.py`.
2. Create `voiceflow-ml/evaluation/diarization_evaluator.py` — `DiarizationEvaluator` class with `compute_der()` and `evaluate_batch()` using `pyannote.metrics`.
3. Create `voiceflow-ml/evaluation/embedding_validator.py` — `EmbeddingValidator` class with `validate_onnx_model()`.
4. Inspect `voiceflow-ml/models/diarization/` — find the model architecture file. If the final layer is a softmax classifier, replace it with an L2-normalized 512-dim embedding projection.
5. Create `voiceflow-ml/scripts/validate_onnx.py` — CLI script that outputs a JSON validation report.
6. Add `pyannote.metrics>=3.2.0` to `requirements.txt`.
7. Run `python voiceflow-ml/scripts/validate_onnx.py --model voiceflow-ml/models/fast_cnn_diarization_optimized.onnx` and note the output node name.

Create `cursor-agents/handoff-agent3.md` with the ONNX output node name and embedding_dim.

---

## AGENT 4 — Rust Diarization Pipeline

Read `cursor-agents/agent4-rust-engine.md` in full, then execute every task:

1. Read `voiceflow-inference/src/inference/mod.rs` fully. Add `run_embedding()` method to `ModelRunner` — takes MFCC features, runs ONNX session, returns `Array1<f32>` shape [512] normalized L2. Use the output node name from `handoff-agent3.md`.
2. Create `voiceflow-inference/src/streaming/sliding_window.rs` — `sliding_window()` function, 3s window, 1s hop, 16kHz.
3. Read `voiceflow-inference/src/streaming/clustering.rs` fully. Complete `OnlineClusterer::add_embedding()`, `smooth_labels()`, `get_segments()` if not already implemented.
4. Rewrite `voiceflow-inference/src/streaming/mod.rs` — connect WebSocket pipeline: audio chunks → sliding_window → MFCC → `run_embedding()` → `OnlineClusterer` → NDJSON segments.
5. Update `voiceflow-inference/src/api/handlers.rs` — `POST /infer` returns diarization segments (not logits).
6. Run `cargo build --release --manifest-path voiceflow-inference/Cargo.toml` — fix all errors until it compiles clean.
7. Run `cargo clippy --manifest-path voiceflow-inference/Cargo.toml -- -D warnings` — fix all warnings.
8. Run `cargo test --manifest-path voiceflow-inference/Cargo.toml`.

Create `cursor-agents/handoff-agent4.md`.

---

## AGENT 5 — Terraform AWS Infrastructure

Read `cursor-agents/agent5-terraform-aws.md` in full, then execute every task:

1. Create `infra/modules/networking/main.tf` — VPC 10.0.0.0/16, public/private/DB subnets, NAT GW, security groups (ALB, ml-service, inference, RDS, Redis) with least-privilege rules.
2. Create `infra/modules/ecs/main.tf` — ECS Fargate cluster, task definitions (secrets from Secrets Manager, never `environment{}`), services, IAM task roles, ALB target groups with sticky sessions for WebSocket, auto-scaling policies.
3. Create `infra/modules/rds/main.tf` — Aurora PostgreSQL Serverless v2, private subnet only, KMS encryption, managed master password.
4. Create `infra/modules/elasticache/main.tf` — Redis 7.x, private subnet, auth token from Secrets Manager.
5. Create `infra/modules/s3/main.tf` — models bucket (versioning ON, public access blocked) + audio bucket (7-day lifecycle, public access blocked).
6. Create `infra/modules/secrets/main.tf` — Secrets Manager for JWT, DB, Redis.
7. Create `infra/modules/ecr/main.tf` — 2 ECR repositories with image scanning enabled.
8. Create `infra/modules/monitoring/main.tf` — CloudWatch log groups, alarms (error_rate > 5%, P99 > 100ms, ECS tasks < 2), SNS topic + email subscription.
9. Create `infra/environments/dev/main.tf` + `variables.tf` + `terraform.tfvars.example`.
10. Create `infra/environments/prod/main.tf` + `variables.tf` + `terraform.tfvars.example`.
11. Run `terraform validate` in `infra/environments/dev/` — fix all errors.

Create `cursor-agents/handoff-agent5.md`.

---

## AGENT 6 — CI/CD Pipeline

Read `cursor-agents/agent6-cicd-pipeline.md` in full, then execute every task:

1. Create `.github/workflows/ci.yml` — jobs: `test-python` (pytest + mypy + flake8 + bandit), `test-rust` (cargo test + clippy + fmt + audit), `security-scan` (gitleaks + semgrep OWASP).
2. Create `.github/workflows/deploy-staging.yml` — triggered on CI success on main; uses AWS OIDC (no static keys); docker buildx → ECR (tag: `sha-${{ github.sha }}`); ECS update-service; wait for stability; smoke tests.
3. Create `.github/workflows/deploy-prod.yml` — `workflow_dispatch` with `DEPLOY` confirmation string; `environment: production` (requires approvers in GitHub Settings); ECS update prod; wait for stability.
4. Create `infra/iam/ci-policy.json` — least-privilege IAM policy for CI role (ECR push + ECS update staging only).

Create `cursor-agents/handoff-agent6.md`.

---

## AGENT 7 — Observability

Read `cursor-agents/agent7-observability.md` in full, then execute every task:

1. Verify `voiceflow-ml/core/logging_config.py` uses `structlog` with JSON renderer in production mode. Fix if not.
2. Verify `voiceflow-inference/src/main.rs` uses `.json()` format for `tracing_subscriber`. Fix if not.
3. Add CloudWatch alarms to `infra/modules/monitoring/main.tf` (if not done in Agent 5): error_rate, P99 latency, ECS task count.
4. Add AWS X-Ray to Python: `aws-xray-sdk>=2.12.0` in `requirements.txt`, `XRayMiddleware` in `api/main.py` (production only).
5. Add OpenTelemetry to Rust `Cargo.toml` + init in `src/main.rs`.
6. Create `grafana/dashboards/voiceflow-overview.json` — dashboard with panels: Inference Latency P99, Requests/s, Active WebSocket Connections, Error Rate %, ECS Task Count.
7. Update `voiceflow-ml/api/routes/health.py` — add `GET /health/ready` endpoint that checks DB + Redis + Rust service, returns 503 if any are down.
8. Update `docker-compose.yml` to mount `grafana/dashboards/` and set default dashboard.

Create `cursor-agents/handoff-agent7.md`.

---

## AGENT 8 — Tests & Quality

Read `cursor-agents/agent8-tests-quality.md` in full, then execute every task:

1. Create `voiceflow-ml/tests/conftest.py` — fixtures: `test_client`, `mock_job_repo`, `mock_redis`, `sample_wav_bytes` (valid 5s 16kHz mono WAV), `fake_zip_bytes`.
2. Create `voiceflow-ml/tests/unit/test_domain.py` — test BatchJob transitions, Segment, ModelVersion.
3. Create `voiceflow-ml/tests/unit/test_services.py` — test `InferenceService.submit_batch_job()`, `get_job_status()`, error cases (not found, empty path).
4. Create `voiceflow-ml/tests/unit/test_audio_validation.py` — test valid WAV passes, ZIP disguised as WAV rejected (400), oversized file rejected (413).
5. Create `voiceflow-ml/tests/integration/test_api_inference.py` — test POST /batch → 202, GET /batch/{id} → 200, unknown job → 404, missing field → 422, GET /health → 200.
6. Create audio test fixture: `voiceflow-ml/tests/fixtures/audio/sample_16k_mono.wav` — generate programmatically with Python `wave` module (5s silence).
7. Update `voiceflow-ml/pytest.ini` — add `--cov-fail-under=70`, `asyncio_mode = auto`, markers.
8. Add to `voiceflow-inference/tests/inference_tests.rs` — tests for `sliding_window` (correct window count, correct lengths), `OnlineClusterer` (same embedding → same speaker, orthogonal → different speakers).
9. Run `pytest voiceflow-ml/tests/unit/ -v` — all tests must pass.
10. Run `cargo test --manifest-path voiceflow-inference/Cargo.toml` — all tests must pass.

Create `cursor-agents/handoff-agent8.md`.

---

## Final Verification

After all 8 agents complete, run these checks and fix any failures:

```bash
# Python: zero hardcoded secrets
grep -rn "SECRET_KEY\s*=\s*['\"]" voiceflow-ml/ --include="*.py"

# Python: clean architecture
grep -r "from sqlalchemy" voiceflow-ml/domain/
grep -r "from fastapi" voiceflow-ml/services/

# Python: tests pass
cd voiceflow-ml && pytest tests/unit/ -q

# Rust: compiles clean
cargo build --release --manifest-path voiceflow-inference/Cargo.toml 2>&1 | grep "^error" | wc -l

# Rust: tests pass
cargo test --manifest-path voiceflow-inference/Cargo.toml -q

# Terraform: valid
cd infra/environments/dev && terraform init -backend=false && terraform validate

# All handoff files present
ls cursor-agents/handoff-agent*.md
```

If any check fails, fix it before moving on. Do not stop until all checks pass.
