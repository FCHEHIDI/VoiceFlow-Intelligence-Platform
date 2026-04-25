# Agent 7 Handoff — Observability

## Files modified / created
- `voiceflow-ml/core/logging_config.py` — accepts `env=` kwarg; emits JSON in production / staging, Console in development. Adds a `service` field to every record.
- `voiceflow-ml/api/main.py` — passes the resolved `app_env` to `configure_logging`. Conditionally enables `aws-xray-sdk` (`patch_all` + `XRayMiddleware`) when `ENV=production` and the SDK is installed.
- `voiceflow-ml/api/routes/health.py` — `/ready` is now an async deep health check that pings PostgreSQL, Redis, and the Rust `/health` endpoint, returning `200` or `503` with detailed component status.
- `voiceflow-ml/requirements.txt` — `aws-xray-sdk>=2.12.0`.
- `grafana/dashboards/voiceflow-overview.json` — main dashboard (latency, RPS, WS, error rate, ECS task count, model load).
- `grafana/dashboards/voiceflow-diarization.json` — diarization-specific dashboard (DER trend, segments / minute, active speakers, embedding latency).
- `grafana/provisioning/dashboards/voiceflow.yml` — Grafana dashboard provisioner pointing at `/var/lib/grafana/dashboards`.
- `grafana/provisioning/datasources/prometheus.yml` — default Prometheus datasource (UID `prometheus`).

## CloudWatch alarms / SNS
The Terraform `monitoring` module (Agent 5) wires the following alarms to an SNS topic with email subscription:
- `voiceflow-{env}-high-error-rate` (5xx > 5% over 5 min)
- `voiceflow-{env}-latency-p99` (TargetResponseTime P99 > 100 ms)
- `voiceflow-{env}-inference-tasks-low` (ECS RunningTaskCount < 2)

## Distributed tracing
- **Python**: `aws-xray-sdk` patches `httpx`, `sqlalchemy`, `boto3`, `requests`. Activated only when `ENV=production` so dev tests don't pull the SDK by default.
- **Rust**: `tracing_subscriber` is initialised in JSON mode (`main.rs`). OpenTelemetry / OTLP exporter wiring is left as a follow-up — recommended crates: `opentelemetry`, `opentelemetry-otlp`, `tracing-opentelemetry` — to be added when the OpenTelemetry collector / X-Ray daemon is provisioned via Terraform.

## Verification
```powershell
$env:ENV="production"
$env:JWT_SECRET_KEY="<32+ chars>"
$env:POSTGRES_PASSWORD="dummy"
python -c "from api.main import app; print('OK')"  # logging now JSON
```
