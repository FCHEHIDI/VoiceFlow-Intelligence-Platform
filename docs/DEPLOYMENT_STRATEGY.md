# Deployment Strategy — VoiceFlow Intelligence Platform

## Overview

4-phase deployment over ~4 weeks, from security hardening to production go-live on AWS ECS Fargate.

---

## Phase 1 — Security Foundation + Clean Architecture (Week 1–2)

**Cursor Agents**: Agent 1 (Security) → Agent 2 (Python Clean Arch)

### Deliverables

- All hardcoded secrets moved to Secrets Manager / env vars
- Audio input validation middleware (magic bytes, size limits)
- `domain/`, `repositories/`, `services/`, `workers/` layers created
- Celery async job queue wired to Redis
- POST /api/inference/batch → 202 Accepted + job_id
- GET /api/inference/batch/{job_id} → status + results

### Exit Criteria

```bash
# Zero hardcoded secrets
grep -r "secret\|password\|api_key" voiceflow-ml/ --include="*.py" | grep -v "\.env\|test\|config\."
# -> Zero results

# Architecture layers
pytest voiceflow-ml/tests/unit/ -v
# -> All pass

grep -r "from sqlalchemy" voiceflow-ml/domain/    # -> Zero results
grep -r "from fastapi" voiceflow-ml/services/     # -> Zero results
```

---

## Phase 2 — ML Pipeline + Rust Engine (Week 2–3)

**Cursor Agents**: Agent 3 (ML Validation) → Agent 4 (Rust Engine)

### Deliverables

- ONNX model produces 512-dim L2-normalized embeddings
- DER evaluation pipeline (`evaluation/diarization_evaluator.py`)
- Rust `run_embedding()` connected to ONNX session
- `sliding_window.rs` (3s window, 1s hop, 16kHz)
- `OnlineClusterer` connected to WebSocket pipeline
- WebSocket streams NDJSON diarization segments

### Exit Criteria

```bash
# ONNX model validation
python voiceflow-ml/scripts/validate_onnx.py \
  --model voiceflow-ml/models/fast_cnn_diarization_optimized.onnx
# -> validation_passed: true, embedding_dim: 512

# Rust pipeline
cargo test --manifest-path voiceflow-inference/Cargo.toml
# -> All tests pass

# WebSocket streaming
wscat -c ws://localhost:3000/ws/stream
# -> Returns NDJSON segments
```

---

## Phase 3 — Infrastructure + CI/CD (Week 3–4)

**Cursor Agents**: Agent 5 (Terraform AWS) → Agent 6 (CI/CD)

### AWS Architecture

```
Internet
  └── ALB (HTTPS 443)
       ├── /api/*    → ECS Fargate: voiceflow-ml (FastAPI, port 8000)
       └── /ws/*     → ECS Fargate: voiceflow-inference (Rust Axum, port 3000)

Private Subnets
  ├── Aurora PostgreSQL Serverless v2 (ml-service only)
  └── ElastiCache Redis 7.x (ml-service + inference)

S3
  ├── voiceflow-models-{env}    (ONNX files, versioned)
  └── voiceflow-audio-{env}     (uploads, 7-day lifecycle)
```

### Service Sizing

| Service | CPU | Memory | Min | Max |
|---------|-----|--------|-----|-----|
| ml-service | 1024 | 2048 MB | 1 | 4 |
| inference-engine | 512 | 1024 MB | 2 | 10 |

### Deliverables

- Terraform modules: networking, ecs, rds, elasticache, s3, secrets, ecr, monitoring
- `infra/environments/dev/` + `infra/environments/prod/`
- GitHub Actions CI (pytest + mypy + cargo test + gitleaks)
- Auto-deploy to staging on merge to main
- Manual approval gate for production

### Exit Criteria

```bash
cd infra/environments/dev && terraform validate
# -> Success! The configuration is valid.

# After deploy
aws ecs describe-services \
  --cluster voiceflow-dev \
  --services ml-service inference-engine \
  --query 'services[].{name:serviceName,running:runningCount,desired:desiredCount}'
# -> running == desired for both services
```

---

## Phase 4 — Observability + Tests + Go-Live (Week 4)

**Cursor Agents**: Agent 7 (Observability) → Agent 8 (Tests & Quality)

### Deliverables

- CloudWatch Alarms: error_rate > 5%, P99 latency > 100ms, ECS tasks < 2
- AWS X-Ray tracing (Python + Rust)
- Grafana dashboards committed as JSON (`grafana/dashboards/`)
- `/health/ready` checks DB + Redis + Rust service
- Python test coverage ≥ 70%
- Rust: clustering + sliding_window tests > 80%
- Load test: P99 < 100ms at 100 concurrent WebSocket clients

### Exit Criteria

```bash
# Coverage
pytest voiceflow-ml/ --cov --cov-fail-under=70
# -> Coverage >= 70%

# Load test
python voiceflow-ml/load_test.py
# -> P99 < 100ms for 100 concurrent clients

# Health check
curl https://<alb-dns>/health/ready
# -> {"status": "ready", "checks": {"database": ..., "redis": ..., "rust_service": ...}}
```

---

## Rollback Plan

| Situation | Action |
|-----------|--------|
| Staging deploy broken | Re-deploy previous `staging-latest` image tag |
| Prod broken | ECS update-service with `sha-<previous-commit>` tag |
| DB migration failed | Aurora point-in-time restore (< 5 min RPO) |
| Catastrophic failure | Terraform destroy staging, rebuild from code |

---

## Secrets Management

**NEVER commit to Git**:
- `JWT_SECRET_KEY` → AWS Secrets Manager
- `POSTGRES_PASSWORD` → AWS Secrets Manager (Aurora managed)
- `REDIS_AUTH_TOKEN` → AWS Secrets Manager

**Local development** (`.env` file, gitignored):
```
JWT_SECRET_KEY=dev-only-not-production-value
POSTGRES_PASSWORD=localdevpassword
REDIS_URL=redis://localhost:6379
```

---

## Agent Execution Order

```
Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6 → Agent 7 → Agent 8
Security   PythonArch  MLValid   Rust      Terraform  CI/CD    Observ    Tests
  Week 1    Week 1-2   Week 2   Week 2-3   Week 3    Week 3   Week 4    Week 4
```

Each agent produces a `handoff-agentN.md` in `cursor-agents/` for the next agent.
