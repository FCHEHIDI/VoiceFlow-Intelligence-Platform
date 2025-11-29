# 📋 CAHIER DES CHARGES - VoiceFlow Intelligence Platform

## 1. PRÉSENTATION DU PROJET

### 1.1 Contexte
VoiceFlow Intelligence Platform est un système de traitement audio temps-réel conçu pour identifier automatiquement les locuteurs dans un flux audio (Speaker Diarization). Le système doit être production-ready avec une architecture hybride optimisant à la fois la flexibilité du Machine Learning et la performance du traitement temps-réel.

### 1.2 Objectifs Globaux
- Fournir un service de speaker diarization avec latence < 100ms pour le streaming temps-réel
- Supporter à la fois le traitement batch et le streaming en temps-réel
- Architecture scalable et maintenable pour un environnement de production
- Pipeline MLOps complet (CI/CD, monitoring, containerization)

### 1.3 Périmètre
**Inclus:**
- Service ML Python pour l'entraînement et la gestion des modèles
- Service Rust pour l'inférence temps-réel haute performance
- API REST pour le traitement batch
- WebSocket pour le streaming temps-réel
- Infrastructure Docker complète
- Monitoring et observabilité (Prometheus + Grafana)
- CI/CD automatisé
- Documentation technique complète

**Exclu:**
- Interface utilisateur web (frontend)
- Intégration avec des services tiers spécifiques
- Gestion avancée des droits utilisateurs (au-delà de JWT basic)
- Support multilingue de la documentation

---

## 2. ACTEURS DU SYSTÈME

### 2.1 ML Engineers
**Rôle:** Développement et optimisation des modèles de diarization
**Besoins:**
- API pour déclencher des entraînements
- Export de modèles vers ONNX
- Métriques de performance des modèles
- Versioning des modèles
- Interface pour télécharger datasets

**Interactions:**
- Utilisation du service Python via API REST
- Accès aux logs d'entraînement
- Visualisation des métriques dans Grafana

### 2.2 DevOps / SRE
**Rôle:** Déploiement et maintenance de l'infrastructure
**Besoins:**
- Configuration Docker Compose simple
- Health checks et readiness probes
- Monitoring système (CPU, mémoire, latence)
- Logs structurés centralisés
- CI/CD automatisé

**Interactions:**
- Déploiement via Docker Compose
- Surveillance via Grafana dashboards
- Alerting via Prometheus

### 2.3 End Users / Applications Clientes
**Rôle:** Consommation des services de diarization
**Besoins:**
- API REST pour traitement batch d'audio
- WebSocket pour streaming temps-réel
- Documentation API claire (OpenAPI/Swagger)
- Authentification sécurisée (JWT)
- Rate limiting pour éviter les abus

**Interactions:**
- Envoi de fichiers audio via HTTP POST
- Connexion WebSocket pour streaming
- Réception des résultats (JSON avec timestamps et speaker labels)

### 2.4 Data Scientists
**Rôle:** Analyse des performances et amélioration continue
**Besoins:**
- Accès aux métriques d'inférence
- Audit trail des prédictions
- Export des résultats pour analyse
- Dashboard de performance des modèles

**Interactions:**
- Requêtes SQL sur base PostgreSQL (metrics, audit_logs)
- Visualisation via Grafana

---

## 3. FONCTIONNALITÉS PRINCIPALES

### 3.1 F1: Speaker Diarization Temps-Réel (Priorité: CRITIQUE)
**Description:** Identification en temps-réel des locuteurs dans un flux audio

**Spécifications:**
- **Input:** Audio stream (WebSocket) @ 16kHz, mono, PCM
- **Output:** Segments JSON avec `{start_time, end_time, speaker_id, confidence}`
- **Latence:** < 100ms (P99)
- **Format audio supporté:** WAV, PCM raw, MP3 (décodé côté client)

**Scénarios d'usage:**
1. Client établit connexion WebSocket `/ws/stream`
2. Client envoie chunks audio de 1 seconde
3. Service Rust traite en temps-réel
4. Résultats streamés immédiatement au client

**Contraintes:**
- Thread-safe (plusieurs connexions simultanées)
- Gestion gracieuse des déconnexions
- Buffer overflow prevention

### 3.2 F2: Traitement Batch Audio (Priorité: HAUTE)
**Description:** Upload et traitement asynchrone de fichiers audio complets

**Spécifications:**
- **Input:** Fichier audio (POST multipart/form-data)
- **Taille max:** 100 MB
- **Durée max:** 30 minutes
- **Output:** Job ID immédiat, résultats récupérables via polling ou webhook

**Endpoints:**
- `POST /api/inference/batch` → Job ID
- `GET /api/inference/batch/{job_id}` → Status + résultats

**Workflow:**
1. Client upload fichier
2. Service Python valide et enqueue job
3. Worker traite asynchrone (Celery ou RQ)
4. Résultats stockés en DB + optionnel callback webhook
5. Client poll status ou reçoit webhook

### 3.3 F3: Gestion des Modèles (Priorité: HAUTE)
**Description:** Entraînement, versioning, et déploiement de modèles ML

**Fonctionnalités:**
- **Entraînement:** `POST /api/models/train` avec hyperparamètres
- **Export ONNX:** `POST /api/models/{model_id}/export`
- **Listing:** `GET /api/models` (avec filtres: version, status, date)
- **Activation:** `PUT /api/models/{model_id}/activate` (set as production model)
- **Rollback:** `PUT /api/models/{previous_id}/activate`

**Métadonnées modèle:**
```json
{
  "model_id": "uuid",
  "version": "1.2.3",
  "architecture": "ResNet-LSTM",
  "training_date": "2025-11-29T10:00:00Z",
  "accuracy": 0.92,
  "status": "active|deprecated",
  "onnx_path": "/models/model-v1.2.3.onnx"
}
```

### 3.4 F4: A/B Testing Modèles (Priorité: MOYENNE)
**Description:** Test de nouveaux modèles sur un pourcentage du trafic

**Spécifications:**
- Configuration: 90% modèle actuel, 10% nouveau modèle
- Routing basé sur hash user_id (consistant)
- Métriques séparées par modèle version
- Comparaison automatique latence + accuracy

**Configuration:**
```yaml
model_routing:
  - model_version: "1.2.3"
    traffic_percent: 90
  - model_version: "1.3.0-beta"
    traffic_percent: 10
```

### 3.5 F5: Monitoring et Observabilité (Priorité: CRITIQUE)
**Description:** Visibilité complète sur système et performances

**Métriques Rust Service:**
- `inference_latency_seconds` (histogram)
- `inference_requests_total` (counter)
- `inference_errors_total` (counter)
- `websocket_connections_active` (gauge)
- `model_load_duration_seconds` (histogram)

**Métriques Python Service:**
- `training_duration_seconds` (histogram)
- `model_accuracy` (gauge)
- `batch_job_duration_seconds` (histogram)
- `batch_queue_size` (gauge)

**Dashboards Grafana:**
1. Real-Time Inference (latency, throughput, error rate)
2. Model Performance (accuracy trends, version comparison)
3. System Health (CPU, memory, GPU utilization)
4. WebSocket Connections (active, errors, bandwidth)

---

## 4. EXIGENCES NON-FONCTIONNELLES

### 4.1 Performance
| Métrique | Objectif | Critique |
|----------|----------|----------|
| Latence streaming (P99) | < 100ms | OUI |
| Latence batch (P99) | < 5s | NON |
| Throughput batch | > 1000 req/sec | NON |
| Concurrent WebSocket connections | > 1000 | OUI |
| Model load time | < 2s | OUI |

### 4.2 Scalabilité
- **Horizontal scaling:** Services stateless (scale via Docker replicas)
- **Database:** PostgreSQL avec connection pooling (max 100 connections)
- **Cache:** Redis pour rate limiting et résultats temporaires
- **Load balancing:** Nginx ou Traefik devant Rust service

### 4.3 Disponibilité
- **Uptime:** 99.5% (acceptable downtime: 3.6h/mois)
- **Graceful shutdown:** Max 30s pour terminer requêtes en cours
- **Health checks:** `/health` et `/ready` endpoints
- **Auto-restart:** Docker restart policy: `unless-stopped`

### 4.4 Sécurité
- **Authentication:** JWT avec expiration 1h
- **Rate limiting:** 100 req/min par user (Redis)
- **Input validation:** 
  - Audio format check (magic bytes)
  - File size limit (100 MB)
  - Malware scan (optionnel, ClamAV)
- **CORS:** Configurable whitelist domains
- **Secrets:** Variables d'environnement (jamais en dur dans code)

### 4.5 Maintenabilité
- **Code coverage:** > 80% (pytest + cargo test)
- **Documentation:** 
  - README complet avec quick start
  - API docs (OpenAPI/Swagger)
  - Architecture docs (diagrammes ASCII)
  - Code comments (docstrings Python, rustdoc)
- **Linting:** 
  - Python: black, flake8, mypy
  - Rust: cargo clippy, cargo fmt
- **Logs:** Structured JSON (structlog + tracing)

### 4.6 Portabilité
- **OS:** Linux (Ubuntu 22.04 recommandé)
- **Container:** Docker 20.10+
- **Orchestration:** Docker Compose (production: Kubernetes-ready)
- **GPU:** CUDA 11.8+ (optionnel pour entraînement)

---

## 5. CONTRAINTES TECHNIQUES

### 5.1 Technologies Imposées
**Backend Python:**
- FastAPI 0.104+
- PyTorch 2.1+ (training)
- ONNX 1.15+ (export/optimization)
- SQLAlchemy 2.0+ (ORM)
- Redis-py 5.0+

**Backend Rust:**
- Axum 0.7+ (web framework)
- Tokio 1.35+ (async runtime)
- ONNX Runtime 1.16+ (inference)
- Tower (middleware)
- Tonic (gRPC, optionnel)

**Infrastructure:**
- PostgreSQL 15+
- Redis 7.2+
- Prometheus 2.48+
- Grafana 10.2+

### 5.2 Contraintes d'Intégration
- Communication Python ↔ Rust: HTTP REST (gRPC en v2)
- Format modèle: ONNX uniquement
- Audio format: 16kHz mono PCM (conversion côté client)
- API versioning: `/v1/` prefix mandatory

### 5.3 Contraintes Réglementaires
- **RGPD:** Pas de stockage audio brut au-delà du traitement (sauf opt-in user)
- **Audit trail:** Log toutes les inférences avec user_id + timestamp
- **Data retention:** 
  - Audio files: 0 jours (suppression immédiate post-traitement)
  - Results: 30 jours
  - Logs: 90 jours

---

## 6. LIVRABLES ATTENDUS

### 6.1 Code Source
- [ ] Python ML service (voiceflow-ml/)
- [ ] Rust inference engine (voiceflow-inference/)
- [ ] Tests unitaires et intégration (coverage > 80%)
- [ ] Fichiers configuration (docker-compose.yml, prometheus.yml)

### 6.2 Documentation
- [ ] README.md (quick start, architecture, deployment)
- [ ] CAHIER_DES_CHARGES.md (ce document)
- [ ] NOTE_DE_CADRAGE.md (planning détaillé)
- [ ] CONCEPTION_TECHNIQUE.md (architecture, data models, API specs)
- [ ] ARCHITECTURE_FLOW.md (diagrammes flux)
- [ ] API documentation (OpenAPI specs auto-générés)

### 6.3 Infrastructure
- [ ] Dockerfiles (Python + Rust multi-stage builds)
- [ ] docker-compose.yml (stack complète)
- [ ] GitHub Actions workflows (CI/CD)
- [ ] Grafana dashboards (JSON configs)
- [ ] Prometheus alerting rules

### 6.4 Tests et Validation
- [ ] Unit tests (pytest + cargo test)
- [ ] Integration tests (API endpoints)
- [ ] E2E tests (streaming workflow complet)
- [ ] Load testing report (wrk ou k6)
- [ ] Performance benchmarks (latency profiling)

---

## 7. CRITÈRES D'ACCEPTATION

### 7.1 Tests Fonctionnels
| ID | Test | Critère de Succès |
|----|------|-------------------|
| T1 | Upload fichier audio 10s | Résultats diarization reçus en < 5s |
| T2 | Stream audio 30s via WebSocket | Latence moyenne < 80ms, P99 < 100ms |
| T3 | Entraînement nouveau modèle | Modèle exporté en ONNX, chargeable par Rust |
| T4 | A/B testing 90/10 | Trafic correctement réparti, métriques séparées |
| T5 | 1000 requêtes batch simultanées | Aucune erreur, throughput > 800 req/sec |

### 7.2 Tests Non-Fonctionnels
| ID | Test | Critère de Succès |
|----|------|-------------------|
| N1 | Code coverage | > 80% (pytest + cargo test) |
| N2 | Docker build time | < 5 min (builds optimisés) |
| N3 | Startup time | Stack complète up en < 60s |
| N4 | Memory footprint | Rust service < 500 MB, Python < 2 GB |
| N5 | Security scan | 0 vulnerabilités critiques (Dependabot) |

### 7.3 Critères de Production-Readiness
- [x] Health checks actifs (`/health`, `/ready`)
- [x] Graceful shutdown (SIGTERM handling)
- [x] Structured logging (JSON format)
- [x] Metrics exportées (Prometheus format)
- [x] Secrets via env vars (pas de hardcoding)
- [x] Error handling exhaustif (pas de panics Rust)
- [x] Input validation complète
- [x] Rate limiting actif
- [x] Documentation API complète

---

## 8. PLANNING INDICATIF

**Durée totale:** 2 jours (16 heures)

**Jour 1 (8h):**
- Phase 1: Documentation (2h)
- Phase 2: Python ML Service Setup (2h)
- Phase 3: Rust Inference Engine Setup (2h)
- Phase 4: Integration Pipeline (2h)

**Jour 2 (8h):**
- Phase 5: Real-Time Streaming (2h)
- Phase 6: MLOps Pipeline (2h)
- Phase 7: Monitoring & Observability (2h)
- Phase 8: Tests & Documentation (2h)

---

## 9. RISQUES ET MITIGATION

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Latence > 100ms streaming | Moyenne | Critique | Profiling continu, ONNX quantization FP16, optimisation Rust |
| Complexité inter-service Python-Rust | Faible | Moyen | API contract strict, tests d'intégration |
| ONNX export issues (incompatibilités) | Moyenne | Moyen | Tests PyTorch→ONNX→Rust dès jour 1 |
| Performance PostgreSQL (bottleneck) | Faible | Moyen | Connection pooling, indexes, Redis cache |
| Dérive modèle en production | Faible | Moyen | Monitoring accuracy, alerting, rollback rapide |

---

## 10. GLOSSAIRE

| Terme | Définition |
|-------|------------|
| **Speaker Diarization** | Processus d'identification "qui parle quand" dans un audio multi-locuteurs |
| **ONNX** | Open Neural Network Exchange - format standard pour modèles ML |
| **WebSocket** | Protocole full-duplex pour communication temps-réel client-serveur |
| **Data Mapper** | Pattern architectural séparant logique business et accès données |
| **P99 Latency** | 99ème percentile de latence (99% des requêtes sous cette valeur) |
| **Quantization** | Réduction précision modèle (FP32→FP16→INT8) pour accélérer inférence |
| **Graceful Shutdown** | Arrêt propre du service en terminant requêtes en cours |

---

## 11. VALIDATION ET APPROBATION

| Rôle | Nom | Signature | Date |
|------|-----|-----------|------|
| Chef de Projet | [À remplir] | | |
| Tech Lead ML | [À remplir] | | |
| DevOps Lead | [À remplir] | | |
| Client/Sponsor | [À remplir] | | |

---

**Version:** 1.0  
**Date de création:** 29 Novembre 2025  
**Auteur:** VoiceFlow Intelligence Team  
**Statut:** ✅ Validé
