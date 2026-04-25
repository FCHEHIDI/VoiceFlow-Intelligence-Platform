# Agent 1 Handoff

- [x] Hardcoded defaults removed from `core/config.py` — `POSTGRES_PASSWORD` and `JWT_SECRET_KEY` (with length ≥ 32) must come from the environment, or in production `JWT` may be loaded from `JWT_SECRET_ARN` via `get_secrets_loader().get_aws_secret_string` when `ENV=production`.
- [x] `core/secrets_manager.py` — `SecretsLoader` with TTL cache, `get_secrets_loader()` singleton, `boto3` in `requirements.txt`.
- [x] `.env.example` at monorepo root; `.gitignore` updated (`*.env`, `!.env.example`).
- [x] `api/middleware/input_validation.py` — `validate_audio_upload` (magic / fallback signatures, RIFF/WAVE, block ZIP/PDF/EXE, 100 MB, 413/400), `content_length_middleware` (413 from `Content-Length` on `/api/inference/*` POSTs), `python-magic` in `requirements.txt`.
- [x] `api/main.py` — CORS from `CORS_ORIGINS` / `cors_allowed_origins`, no wildcard in production, content-length middleware registered.
- [x] `api/routes/inference.py` — `validate_audio_upload` on `/batch` and `/sync`.
- [x] `docker-compose.yml` — `POSTGRES_PASSWORD`, `GF_SECURITY_ADMIN_PASSWORD`, `JWT_SECRET_KEY` and related values from environment (`:?` / defaults where safe).
- [x] `voiceflow-inference/src/config.rs` — `validate_required_secrets()`; `main.rs` calls it at startup.
- [x] `voiceflow-ml/tests/conftest.py` — sets `JWT_SECRET_KEY` and `POSTGRES_PASSWORD` for pytest so `Settings` can load.
