# AGENT 7 — Observabilite & Monitoring
**Role** : Senior SRE / Observability Engineer
**Duree estimee** : 3-4h
**Prerequis** : Agent 5 termine (infra Terraform deployee)

---

## Contexte

### Ce qui existe deja (local)

- Prometheus scrape `voiceflow-inference:3000/metrics` — OK
- Metriques Rust : `inference_latency_seconds`, `inference_requests_total`, `websocket_connections_active`
- Grafana local port 3001 — OK
- structlog Python configure — a verifier si JSON en prod

### Ce qui manque pour AWS

1. CloudWatch Logs (logs structures JSON en prod)
2. CloudWatch Alarms (SLA alerting)
3. AWS X-Ray distributed tracing (Python + Rust)
4. Dashboards Grafana provisiones en code (JSON commite)

---

## Tache 1 — Logs JSON en production

### Python — verifier `voiceflow-ml/core/logging_config.py`

```python
def configure_logging(env: str = "development") -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]
    if env == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(processors=processors)
```

Chaque log doit contenir : `timestamp`, `level`, `service=voiceflow-ml`, `request_id`, `message`.

### Rust — verifier `voiceflow-inference/src/main.rs`

```rust
tracing_subscriber::fmt()
    .with_max_level(Level::INFO)
    .json()   // REQUIS pour CloudWatch
    .init();
```

---

## Tache 2 — CloudWatch Alarms (dans `infra/modules/monitoring/`)

```hcl
# Alarm 1 : Error rate > 5% sur 5 minutes
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "voiceflow-${var.environment}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 5
  metric_name         = "5xx_errors"
  namespace           = "AWS/ApplicationELB"
  statistic           = "Sum"
  period              = 300
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# Alarm 2 : Latence P99 > 100ms
resource "aws_cloudwatch_metric_alarm" "high_latency_p99" {
  alarm_name          = "voiceflow-${var.environment}-latency-p99"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 100  # ms
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  extended_statistic  = "p99"
  period              = 60
  evaluation_periods  = 3
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Alarm 3 : ECS inference tasks < 2 (service down)
resource "aws_cloudwatch_metric_alarm" "low_inference_tasks" {
  alarm_name          = "voiceflow-${var.environment}-inference-tasks-low"
  comparison_operator = "LessThanThreshold"
  threshold           = 2
  metric_name         = "RunningTaskCount"
  namespace           = "AWS/ECS"
  dimensions = {
    ClusterName = "voiceflow-${var.environment}"
    ServiceName = "inference-engine"
  }
  period             = 60
  evaluation_periods = 2
  alarm_actions      = [aws_sns_topic.alerts.arn]
}

# SNS Topic + subscription email
resource "aws_sns_topic" "alerts" {
  name = "voiceflow-alerts-${var.environment}"
}
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
```

---

## Tache 3 — AWS X-Ray Tracing

### Python (`voiceflow-ml/`)

```python
# requirements.txt : aws-xray-sdk>=2.12.0

# api/main.py — apres creation app
from aws_xray_sdk.core import xray_recorder, patch_all
from aws_xray_sdk.ext.fastapi.middleware import XRayMiddleware

if settings.env == "production":
    xray_recorder.configure(
        service="voiceflow-ml",
        sampling=True
    )
    patch_all()  # Auto-patche httpx, sqlalchemy, boto3
    app.add_middleware(XRayMiddleware, recorder=xray_recorder)
```

### Rust (`voiceflow-inference/`)

```toml
# Cargo.toml
opentelemetry = { version = "0.21", features = ["trace"] }
opentelemetry-otlp = { version = "0.14", features = ["trace"] }
tracing-opentelemetry = "0.22"
```

```rust
// src/main.rs
fn init_tracing(service_name: &str) {
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(opentelemetry_otlp::new_exporter().tonic())
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .expect("Failed to init tracer");

    Registry::default()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::fmt::layer().json())
        .init();
}
```

---

## Tache 4 — Dashboard Grafana commite en JSON

```
grafana/
└── dashboards/
    ├── voiceflow-overview.json    # Dashboard principal
    └── voiceflow-diarization.json # DER trends, speaker distribution
```

Le dashboard `voiceflow-overview.json` doit contenir :
- Panel : Inference Latency P99 (ms) — source Prometheus ou CloudWatch
- Panel : Requests/second — inference_requests_total
- Panel : Active WebSocket Connections
- Panel : Error Rate %
- Panel : Recent Errors (CloudWatch Logs Insights)
- Panel : ECS Task Count

Commiter ces JSON dans le repo et provisioner via Docker Compose :
```yaml
grafana:
  volumes:
    - ./grafana/dashboards:/var/lib/grafana/dashboards
  environment:
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH: /var/lib/grafana/dashboards/voiceflow-overview.json
```

---

## Tache 5 — Health checks approfondis

Modifier `voiceflow-ml/api/routes/health.py` pour ajouter `/ready` :

```python
@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)) -> dict:
    """
    Verifie DB + Redis + service Rust.
    Retourne 503 si un composant est indisponible.
    """
    checks = {
        "database": await _ping_db(db),
        "redis": await _ping_redis(redis),
        "rust_service": await _ping_rust(settings.rust_service_url)
    }
    all_ok = all(c["status"] == "ok" for c in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks}
    )
```

---

## Contraintes (NE PAS toucher)

- `prometheus.yml` local — conserver pour le dev
- Metriques Prometheus Rust existantes — ne pas supprimer
- La stack Grafana locale reste fonctionnelle

---

## Verification finale

```bash
# Logs JSON en prod
ENV=production docker-compose up ml-service
curl http://localhost:8000/health
docker-compose logs ml-service --no-log-prefix | head -3 | python3 -c "
import sys, json
for line in sys.stdin:
    json.loads(line.strip())
    print('JSON OK')
"

# Metriques Prometheus
curl http://localhost:3000/metrics | grep -E "inference_latency|websocket_connections"

# Health readiness
curl http://localhost:8000/ready
# Doit retourner 200 si DB + Redis + Rust up, 503 sinon
```
