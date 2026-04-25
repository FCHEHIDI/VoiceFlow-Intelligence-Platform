# Cursor Agents — VoiceFlow Intelligence Platform Refactoring

Workflow de refactoring production avec agents spécialisés.

## Usage dans Cursor

Dans le chat Cursor, tape `@cursor-agents/agentX-*.md` pour charger le contexte de l'agent voulu, puis lance la conversation.

## Ordre d'exécution

```
AGENT 1 (Security & Foundation)          ← COMMENCER ICI
    │
    ├──► AGENT 2 (Python Clean Arch)  ──► AGENT 3 (ML Validation)
    │
    └──► AGENT 4 (Rust Engine)
    
    Après 1+2+3+4 terminés :
    
    AGENT 5 (Terraform AWS)
    AGENT 6 (CI/CD)            ← parallélisable avec 5
    
    Après infra posée :
    
    AGENT 7 (Observabilité)
    AGENT 8 (Tests & Quality)  ← peut commencer en parallèle de 5+6
```

## Agents disponibles

| Fichier | Rôle | Dépend de |
|---|---|---|
| `agent1-security-foundation.md` | Secrets, JWT, OWASP | — |
| `agent2-python-clean-arch.md` | FastAPI, Clean Architecture, Celery | Agent 1 |
| `agent3-ml-validation.md` | DER, modèle réel, ONNX | Agent 1 |
| `agent4-rust-engine.md` | Diarization pipeline, WebSocket | Agent 1 |
| `agent5-terraform-aws.md` | ECS, RDS, S3, Secrets Manager | Agents 1-4 |
| `agent6-cicd-pipeline.md` | GitHub Actions, CodePipeline, ECR | Agent 5 |
| `agent7-observability.md` | X-Ray, CloudWatch, Alarms | Agent 5 |
| `agent8-tests-quality.md` | Pytest, Rust tests, E2E | Agents 2-4 |
