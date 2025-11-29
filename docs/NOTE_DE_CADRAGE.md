# 📊 NOTE DE CADRAGE - VoiceFlow Intelligence Platform

## 1. RÉSUMÉ EXÉCUTIF

**Projet:** VoiceFlow Intelligence Platform  
**Type:** Système de traitement audio temps-réel avec ML  
**Durée:** 2 jours (16 heures de développement)  
**Budget estimé:** N/A (projet interne de démonstration technique)  
**Équipe:** 1 développeur full-stack ML/Rust  
**Date de début:** 29 Novembre 2025  
**Date de livraison:** 30 Novembre 2025  

**Objectif principal:**  
Construire un MVP production-ready démontrant expertise en architecture ML hybride (Python/Rust), traitement temps-réel, et bonnes pratiques MLOps.

**Valeur ajoutée:**
- Démonstration de compétences techniques avancées (ML + Systems Programming)
- Portfolio piece pour recrutement ML Expert
- Base réutilisable pour projets futurs de traitement audio

---

## 2. CONTEXTE ET ENJEUX

### 2.1 Contexte Technique
Le traitement audio en temps-réel nécessite:
- **Performance:** Latence < 100ms incompatible avec Python pur
- **Flexibilité:** Entraînement ML complexe nécessite écosystème Python
- **Solution:** Architecture hybride Python (training) + Rust (inference)

### 2.2 Enjeux Business
- **Différenciation:** Peu de solutions open-source production-ready en diarization temps-réel
- **Scalabilité:** Marché croissant (transcription meetings, assistants vocaux, modération contenu)
- **Technique:** Démonstration maîtrise stack moderne (Rust, ONNX, MLOps)

### 2.3 Contraintes Projet
- **Temporelles:** 2 jours maximum (proof of concept)
- **Humaines:** Solo developer (toutes casquettes: ML, backend, DevOps)
- **Techniques:** Pas de GPU disponible (optimisations CPU uniquement)
- **Budget:** Gratuit (services cloud exclus, tout en local)

---

## 3. PLANNING DÉTAILLÉ

### 3.1 Vue d'Ensemble

| Phase | Durée | Jour | Horaires | Livrables Clés |
|-------|-------|------|----------|----------------|
| Phase 1: Documentation | 2h | J1 | 09:00-11:00 | 4 documents markdown |
| Phase 2: Python ML Setup | 2h | J1 | 11:00-13:00 | FastAPI + modèle PyTorch stub |
| Phase 3: Rust Inference Setup | 2h | J1 | 14:00-16:00 | Axum + ONNX Runtime |
| Phase 4: Integration Pipeline | 2h | J1 | 16:00-18:00 | PyTorch→ONNX→Rust working |
| Phase 5: Real-Time Streaming | 2h | J2 | 09:00-11:00 | WebSocket < 100ms latency |
| Phase 6: MLOps Pipeline | 2h | J2 | 11:00-13:00 | Docker Compose + CI/CD |
| Phase 7: Monitoring | 2h | J2 | 14:00-16:00 | Prometheus + Grafana |
| Phase 8: Tests & Docs | 2h | J2 | 16:00-18:00 | Coverage > 80% + README |

### 3.2 JOUR 1 - Foundation & Core Architecture

#### ⏰ Phase 1: Documentation (09:00-11:00)

**Objectif:** Poser les fondations conceptuelles du projet

**Tâches détaillées:**
1. **CAHIER_DES_CHARGES.md (45 min)**
   - Section 1-3: Présentation, acteurs, fonctionnalités (20 min)
   - Section 4-6: Exigences, contraintes, livrables (15 min)
   - Section 7-9: Critères acceptation, planning, risques (10 min)

2. **NOTE_DE_CADRAGE.md (30 min)**
   - Planning détaillé avec timeline précise
   - Ressources nécessaires (stack technique)
   - Indicateurs de succès (KPIs)
   - Risques identifiés avec mitigation

3. **CONCEPTION_TECHNIQUE.md (30 min)**
   - Architecture layered (diagramme ASCII)
   - Data models (PostgreSQL schema)
   - API contracts (exemple OpenAPI)
   - Flow de traitement audio

4. **ARCHITECTURE_FLOW.md (15 min)**
   - Diagrammes de séquence (streaming vs batch)
   - Communication inter-services
   - Gestion des erreurs

**Livrables:**
- ✅ 4 documents markdown complets
- ✅ Diagrammes ASCII intégrés
- ✅ Specs techniques détaillées

**Critère de succès:** Documentation suffisamment détaillée pour développement sans ambiguïté

---

#### ⏰ Phase 2: Python ML Service Setup (11:00-13:00)

**Objectif:** Service FastAPI fonctionnel avec modèle PyTorch stub

**Tâches détaillées:**
1. **Initialisation projet (20 min)**
   ```bash
   cd voiceflow-ml
   python -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn[standard] torch onnx sqlalchemy redis pytest
   pip freeze > requirements.txt
   ```

2. **Structure Data Mapper (30 min)**
   - `repositories/model_repository.py`: CRUD modèles
   - `services/training_service.py`: Logique métier
   - `api/routes/models.py`: Endpoints REST
   - `api/main.py`: App FastAPI

3. **Modèle PyTorch stub (40 min)**
   - `models/diarization/model.py`: Architecture simple CNN (pas de vraie diarization)
   - `models/diarization/train.py`: Training loop stub (epochs, loss)
   - `models/diarization/export_onnx.py`: PyTorch → ONNX conversion

4. **API endpoints (30 min)**
   - `POST /api/models/train`: Déclencher training
   - `POST /api/models/{id}/export`: Exporter en ONNX
   - `POST /api/inference/batch`: Inférence batch
   - `GET /health`: Health check

**Livrables:**
- ✅ FastAPI server runnable (`uvicorn api.main:app`)
- ✅ Modèle PyTorch exportable en ONNX
- ✅ API docs auto-générées (Swagger)

**Critère de succès:** `curl http://localhost:8000/health` → 200 OK

---

#### ⏰ Phase 3: Rust Inference Engine Setup (14:00-16:00)

**Objectif:** Service Rust avec ONNX Runtime et WebSocket

**Tâches détaillées:**
1. **Initialisation Cargo (15 min)**
   ```bash
   cargo new voiceflow-inference --name voiceflow_inference
   cd voiceflow-inference
   # Ajout dependencies dans Cargo.toml
   cargo build
   ```

2. **ONNX Runtime integration (45 min)**
   - `src/inference/onnx_runtime.rs`:
     - Struct `ModelRunner` avec `Arc<Session>`
     - Méthode `load_model(path)` avec optimizations
     - Méthode `run_inference(input: &[f32])` thread-safe
   - `src/inference/model_manager.rs`:
     - Hot-reload de modèles
     - Version management

3. **Axum HTTP API (30 min)**
   - `src/api/mod.rs`:
     - `POST /infer`: Inférence single audio
     - `GET /health`: Health check
     - `GET /metrics`: Prometheus metrics
   - Middleware: logging, CORS

4. **WebSocket server stub (30 min)**
   - `src/streaming/websocket.rs`:
     - Handler connexion WebSocket
     - Echo server basique (amélioration en Phase 5)
   - `src/streaming/audio_buffer.rs`:
     - Buffer circulaire pour audio chunks

**Livrables:**
- ✅ Rust server compilable et runnable
- ✅ ONNX model chargé et inférence fonctionnelle
- ✅ WebSocket echo server

**Critère de succès:** `cargo run --release` → server écoute sur :3000

---

#### ⏰ Phase 4: Integration Pipeline (16:00-18:00)

**Objectif:** Pipeline complet Python → ONNX → Rust validé

**Tâches détaillées:**
1. **Test end-to-end (45 min)**
   - Python: Entraîner modèle stub → exporter ONNX
   - Copier ONNX vers `/models/`
   - Rust: Charger modèle → run inference
   - Valider outputs identiques Python vs Rust

2. **Communication inter-services (30 min)**
   - Python appelle Rust via HTTP:
     ```python
     response = requests.post("http://localhost:3000/infer", json={...})
     ```
   - Rust récupère metadata modèles depuis Python:
     ```rust
     let models = reqwest::get("http://localhost:8000/api/models").await?;
     ```

3. **Docker setup (45 min)**
   - `voiceflow-ml/Dockerfile`:
     ```dockerfile
     FROM python:3.11-slim
     WORKDIR /app
     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt
     COPY . .
     CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
     ```
   - `voiceflow-inference/Dockerfile`:
     ```dockerfile
     FROM rust:1.75 as builder
     WORKDIR /app
     COPY . .
     RUN cargo build --release
     FROM debian:bookworm-slim
     COPY --from=builder /app/target/release/voiceflow_inference .
     CMD ["./voiceflow_inference"]
     ```

**Livrables:**
- ✅ Pipeline PyTorch→ONNX→Rust validé
- ✅ Dockerfiles fonctionnels
- ✅ Communication Python↔Rust OK

**Critère de succès:** Inférence identique Python vs Rust (tolerance 1e-5)

---

### 3.3 JOUR 2 - Production Features & MLOps

#### ⏰ Phase 5: Real-Time Streaming (09:00-11:00)

**Objectif:** WebSocket streaming avec latence < 100ms

**Tâches détaillées:**
1. **WebSocket complet (50 min)**
   - `src/streaming/websocket.rs`:
     - Recevoir audio chunks (1s @ 16kHz = 16000 samples)
     - Feature extraction (MFCC stub ou mel-spectrogram)
     - Appel `ModelRunner::run_inference()`
     - Stream résultats back to client
   - Gestion erreurs: déconnexion, buffer overflow

2. **Performance optimization (40 min)**
   - ONNX quantization FP16:
     ```python
     from onnxruntime.quantization import quantize_dynamic
     quantize_dynamic("model.onnx", "model-fp16.onnx", weight_type=QuantType.QUInt8)
     ```
   - Batch inference (accumuler 5 chunks → batch inference)
   - Connection pooling
   - Memory profiling (Rust: `heaptrack`, Python: `memory_profiler`)

3. **Client test (30 min)**
   - Script Python WebSocket client:
     ```python
     import websockets
     async with websockets.connect("ws://localhost:3000/ws/stream") as ws:
         for chunk in audio_chunks:
             await ws.send(chunk)
             result = await ws.recv()
             print(f"Latency: {latency}ms")
     ```
   - Mesure latency end-to-end (P99)

**Livrables:**
- ✅ Streaming temps-réel fonctionnel
- ✅ Latency < 100ms (P99)
- ✅ Script test client

**Critère de succès:** 1000 chunks traités sans erreur, latency moyenne < 80ms

---

#### ⏰ Phase 6: MLOps Pipeline (11:00-13:00)

**Objectif:** Infrastructure complète avec CI/CD

**Tâches détaillées:**
1. **Docker Compose (45 min)**
   ```yaml
   version: '3.8'
   services:
     postgres:
       image: postgres:15
       environment:
         POSTGRES_DB: voiceflow
     redis:
       image: redis:7.2-alpine
     ml-service:
       build: ./voiceflow-ml
       depends_on: [postgres, redis]
     inference-engine:
       build: ./voiceflow-inference
       depends_on: [ml-service]
     prometheus:
       image: prom/prometheus
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml
     grafana:
       image: grafana/grafana
       depends_on: [prometheus]
   ```

2. **GitHub Actions CI/CD (45 min)**
   - `.github/workflows/ml-pipeline.yml`:
     ```yaml
     name: ML Pipeline
     on: [push]
     jobs:
       test-python:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v3
           - run: pytest tests/ --cov
       build-docker:
         runs-on: ubuntu-latest
         steps:
           - run: docker build -t voiceflow-ml .
     ```
   - `.github/workflows/inference-pipeline.yml` (similaire pour Rust)

3. **Model versioning (30 min)**
   - Table PostgreSQL `models`:
     ```sql
     CREATE TABLE models (
       id UUID PRIMARY KEY,
       version VARCHAR(50),
       onnx_path TEXT,
       is_active BOOLEAN,
       created_at TIMESTAMP
     );
     ```
   - API endpoint `PUT /api/models/{id}/activate`

**Livrables:**
- ✅ `docker-compose up` one-command startup
- ✅ CI/CD pipelines actifs
- ✅ Model registry opérationnel

**Critère de succès:** `docker-compose up` → stack complète up en < 60s

---

#### ⏰ Phase 7: Monitoring & Observability (14:00-16:00)

**Objectif:** Monitoring complet avec dashboards

**Tâches détaillées:**
1. **Prometheus metrics (50 min)**
   - Rust (`src/metrics/mod.rs`):
     ```rust
     use prometheus::{Histogram, Counter, Gauge};
     lazy_static! {
         pub static ref INFERENCE_LATENCY: Histogram = 
             register_histogram!("inference_latency_seconds", "Inference latency").unwrap();
     }
     ```
   - Python:
     ```python
     from prometheus_client import Histogram, Counter
     inference_duration = Histogram('training_duration_seconds', 'Training duration')
     ```
   - Endpoint `/metrics` (format Prometheus)

2. **Grafana dashboards (40 min)**
   - Dashboard JSON config `grafana/dashboards/inference.json`:
     - Panel: Latency P50/P99/P99.9 (graph)
     - Panel: Throughput (counter rate)
     - Panel: Error rate (heatmap)
     - Panel: Active WebSocket connections (gauge)
   - Import via Grafana API

3. **Structured logging (30 min)**
   - Python:
     ```python
     import structlog
     logger = structlog.get_logger()
     logger.info("inference_completed", model_version="1.2.3", latency_ms=85)
     ```
   - Rust:
     ```rust
     use tracing::{info, instrument};
     #[instrument]
     async fn run_inference() {
         info!(model_version = "1.2.3", latency_ms = 85);
     }
     ```

**Livrables:**
- ✅ Metrics exportées Prometheus
- ✅ 3 Grafana dashboards opérationnels
- ✅ Logs JSON structurés

**Critère de succès:** Dashboard affiche métriques en temps-réel

---

#### ⏰ Phase 8: Tests & Documentation (16:00-18:00)

**Objectif:** Tests coverage > 80% + documentation complète

**Tâches détaillées:**
1. **Tests Python (40 min)**
   - `tests/test_services.py` (unit tests services)
   - `tests/test_repositories.py` (mocks DB)
   - `tests/test_api.py` (TestClient FastAPI)
   - Run: `pytest tests/ --cov=. --cov-report=html`
   - Target: 80%+ coverage

2. **Tests Rust (30 min)**
   - `tests/inference_tests.rs` (unit tests)
   - `tests/integration_tests.rs` (test WebSocket)
   - Run: `cargo test --verbose`
   - Run: `cargo clippy -- -D warnings`

3. **Documentation (40 min)**
   - `README.md`:
     - Quick start guide
     - Architecture diagram
     - API examples (curl)
     - Troubleshooting
   - Code comments:
     - Python docstrings
     - Rust rustdoc (`cargo doc --open`)

4. **Performance benchmarking (10 min)**
   - Load test: `wrk -t4 -c100 -d30s http://localhost:3000/infer`
   - Latency profile: mesure P50/P99/P99.9
   - Document résultats dans README

**Livrables:**
- ✅ Tests passing (coverage > 80%)
- ✅ README complet
- ✅ Benchmark results documented

**Critère de succès:** `cargo test` et `pytest` → 100% passing

---

## 4. RESSOURCES NÉCESSAIRES

### 4.1 Ressources Humaines
| Rôle | Compétences Requises | Allocation |
|------|---------------------|------------|
| Développeur Full-Stack | Python, Rust, ML, DevOps | 16h (100%) |

### 4.2 Ressources Matérielles
| Ressource | Spécifications | Usage |
|-----------|---------------|-------|
| Machine développement | 16 GB RAM, CPU 8 cores | Local dev |
| Stockage | 50 GB disponible | Docker images, models |
| GPU | Non requis (nice-to-have) | Training accéléré |

### 4.3 Stack Technique
**Backend:**
- Python 3.11+ (FastAPI, PyTorch, ONNX)
- Rust 1.75+ (Axum, Tokio, ONNX Runtime)

**Infrastructure:**
- Docker 20.10+, Docker Compose 2.0+
- PostgreSQL 15+
- Redis 7.2+
- Prometheus 2.48+, Grafana 10.2+

**Outils Dev:**
- VS Code (Python + Rust extensions)
- Git + GitHub (versioning + CI/CD)
- Postman/curl (API testing)
- wrk/k6 (load testing)

### 4.4 Données
| Type | Source | Volume |
|------|--------|--------|
| Audio samples (dev) | LibriSpeech ou synthétique | 1 GB |
| Test dataset | Synthétique (speech_recognition) | 100 MB |

---

## 5. INDICATEURS DE SUCCÈS (KPIs)

### 5.1 KPIs Techniques
| KPI | Objectif | Mesure |
|-----|----------|--------|
| **Latency P99 streaming** | < 100ms | Prometheus metrics |
| **Throughput batch** | > 1000 req/sec | Load test (wrk) |
| **Test coverage** | > 80% | pytest + cargo test |
| **Code quality** | 0 warnings | clippy + flake8 |
| **Build time** | < 5 min | CI/CD logs |
| **Startup time** | < 60s | docker-compose up |

### 5.2 KPIs Fonctionnels
| KPI | Objectif | Validation |
|-----|----------|------------|
| **Streaming audio 30s** | Latency < 100ms | Test client WebSocket |
| **Batch processing** | Results < 5s | API test |
| **Model export** | PyTorch → ONNX OK | Integration test |
| **A/B testing** | Traffic split 90/10 | Metrics separated |

### 5.3 KPIs Qualité
| KPI | Objectif | Mesure |
|-----|----------|--------|
| **Documentation** | README complet | Review checklist |
| **API docs** | OpenAPI specs | Swagger UI |
| **Error handling** | 0 unhandled exceptions | Tests E2E |
| **Security** | 0 critical vulns | Dependabot scan |

---

## 6. RISQUES ET MITIGATION

### 6.1 Risques Techniques

| ID | Risque | Probabilité | Impact | Mitigation | Contingence |
|----|--------|-------------|--------|------------|-------------|
| R1 | Latency > 100ms | Moyenne (40%) | Critique | Profiling dès jour 1, ONNX FP16 | Accepter 150ms, documenter |
| R2 | ONNX export incompatible | Faible (20%) | Moyen | Test PyTorch→ONNX→Rust dès phase 4 | Simplifier architecture modèle |
| R3 | WebSocket instabilité | Faible (15%) | Moyen | Tests robustesse (disconnect, timeout) | Fallback HTTP chunking |
| R4 | PostgreSQL bottleneck | Faible (10%) | Faible | Connection pooling, indexes | Redis cache agressif |
| R5 | Rust compilation lente | Moyenne (30%) | Faible | Incremental builds, cache CI/CD | Accepter, optimiser en v2 |

### 6.2 Risques Projet

| ID | Risque | Probabilité | Impact | Mitigation |
|----|--------|-------------|--------|------------|
| P1 | Dépassement planning | Moyenne (35%) | Moyen | Priorisation stricte (MVP first) |
| P2 | Complexité sous-estimée | Moyenne (40%) | Moyen | Buffer 10% dans chaque phase |
| P3 | Bugs bloquants | Faible (20%) | Critique | Tests continus, rollback rapide |
| P4 | Manque expertise Rust | Faible (15%) | Moyen | Documentation + exemples tiers |

### 6.3 Plan de Contingence

**Si dépassement > 2h:**
- Réduire scope: supprimer A/B testing (feature non-critique)
- Simplifier monitoring: metrics basiques uniquement
- Report documentation détaillée en phase post-MVP

**Si bug critique bloquant:**
- Rollback dernière version stable
- Debug isolé (tests unitaires ciblés)
- Demande aide communauté (Discord Rust, Stack Overflow)

---

## 7. LIVRABLES PAR PHASE

### 7.1 Documentation (Fin Phase 1)
- [ ] CAHIER_DES_CHARGES.md (10 sections)
- [ ] NOTE_DE_CADRAGE.md (ce document)
- [ ] CONCEPTION_TECHNIQUE.md (diagrammes + specs)
- [ ] ARCHITECTURE_FLOW.md (séquences)

### 7.2 Code Fonctionnel (Fin Jour 1)
- [ ] Python ML service runnable
- [ ] Rust inference engine runnable
- [ ] Pipeline PyTorch→ONNX→Rust validé
- [ ] Dockerfiles testés

### 7.3 Features Production (Fin Jour 2)
- [ ] WebSocket streaming < 100ms
- [ ] Docker Compose stack complète
- [ ] CI/CD pipelines actifs
- [ ] Monitoring Grafana dashboards

### 7.4 Tests & Validation (Fin Jour 2)
- [ ] Tests coverage > 80%
- [ ] README complet
- [ ] Performance benchmarks
- [ ] Demo video (optionnel)

---

## 8. COMMUNICATION ET REPORTING

### 8.1 Points de Contrôle
| Moment | Type | Participants | Objectif |
|--------|------|--------------|----------|
| Fin Jour 1 (18:00) | Review | Solo (auto-évaluation) | Valider foundation OK |
| Fin Jour 2 (18:00) | Démo | Solo + potentiel review externe | Présenter MVP |

### 8.2 Reporting
**Format:** Git commits structurés
```
feat(ml): add PyTorch diarization model stub
fix(rust): resolve WebSocket connection timeout
docs: update README with quick start guide
test(python): add repository unit tests (coverage 85%)
```

**Dashboard:** GitHub Projects (optionnel)
- Colonnes: To Do, In Progress, Done
- Issues liées aux tâches du planning

---

## 9. CRITÈRES DE VALIDATION FINALE

### 9.1 Checklist Acceptation

**Fonctionnel:**
- [ ] Streaming audio 30s fonctionne sans erreur
- [ ] Latence P99 < 100ms mesurée et documentée
- [ ] Batch processing retourne résultats < 5s
- [ ] Model training → export ONNX → load Rust OK

**Non-Fonctionnel:**
- [ ] Tests coverage > 80% (pytest + cargo test)
- [ ] Docker Compose one-command startup
- [ ] CI/CD pipelines verts (GitHub Actions)
- [ ] Monitoring dashboards opérationnels

**Documentation:**
- [ ] README avec quick start complet
- [ ] API docs auto-générées (Swagger)
- [ ] Architecture docs avec diagrammes
- [ ] Code comments (docstrings + rustdoc)

**Qualité:**
- [ ] 0 warnings (clippy + flake8)
- [ ] 0 vulnerabilités critiques (Dependabot)
- [ ] Graceful shutdown fonctionne
- [ ] Health checks actifs

### 9.2 Critères de Succès Projet

**✅ MVP Validé si:**
1. Streaming audio fonctionne avec latence acceptable (< 150ms acceptable)
2. Architecture hybride Python/Rust démontrée
3. Pipeline MLOps basique opérationnel
4. Documentation permet reproduction par tiers

**🎯 Succès Optimal si:**
1. Tous les critères MVP + latence < 100ms
2. Tests coverage > 85%
3. CI/CD complet avec deployments automatisés
4. Monitoring production-ready

---

## 10. POST-MORTEM ET AMÉLIORATION CONTINUE

### 10.1 Rétrospective (Post-Projet)
**Questions clés:**
- Qu'est-ce qui a bien fonctionné ?
- Quels obstacles rencontrés ?
- Qu'améliorer pour projet similaire ?
- Leçons apprises (techniques + méthodologiques)

### 10.2 Roadmap v2 (Futures Améliorations)
**Features:**
- [ ] Vrai modèle diarization (ResNet + LSTM)
- [ ] Support GPU (CUDA acceleration)
- [ ] gRPC Python↔Rust (remplace HTTP)
- [ ] Kubernetes deployment (replace Docker Compose)
- [ ] Advanced A/B testing (multi-armed bandit)

**Améliorations Techniques:**
- [ ] Quantization INT8 (vs FP16)
- [ ] Model distillation (reduce size)
- [ ] Distributed training (multi-GPU)
- [ ] Caching sophistiqué (Redis + CDN)

---

## 11. ANNEXES

### 11.1 Commandes Rapides

**Setup Environnement:**
```bash
# Python
cd voiceflow-ml
python -m venv venv
.\venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt

# Rust
cd voiceflow-inference
cargo build --release

# Docker
docker-compose up --build -d
```

**Tests:**
```bash
# Python tests
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Rust tests
cargo test --verbose
cargo clippy -- -D warnings
```

**Monitoring:**
```bash
# Access services
Grafana: http://localhost:3001 (admin/admin)
Prometheus: http://localhost:9090
API Docs: http://localhost:8000/docs
```

### 11.2 Ressources Utiles

**Documentation:**
- FastAPI: https://fastapi.tiangolo.com
- Axum: https://docs.rs/axum
- ONNX Runtime: https://onnxruntime.ai/docs/
- PyTorch to ONNX: https://pytorch.org/docs/stable/onnx.html

**Tutoriels:**
- Rust async: https://tokio.rs/tokio/tutorial
- WebSocket Axum: https://github.com/tokio-rs/axum/tree/main/examples/websockets
- Speaker Diarization: https://github.com/pyannote/pyannote-audio

---

## 12. SIGNATURES ET VALIDATION

| Rôle | Nom | Date | Signature |
|------|-----|------|-----------|
| Chef de Projet | [Auto-validation] | 29/11/2025 | ✅ |
| Tech Lead | [Auto-validation] | 29/11/2025 | ✅ |

---

**Version:** 1.0  
**Date:** 29 Novembre 2025  
**Statut:** ✅ Validé - Prêt pour exécution  
**Prochaine étape:** Phase 1 - Documentation (démarrage immédiat)
