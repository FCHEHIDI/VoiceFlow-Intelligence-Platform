# AGENT 2 — Python Clean Architecture
**Role** : Senior Python Backend Engineer
**Duree estimee** : 4-6h
**Prerequis** : Agent 1 termine (handoff-agent1.md present)

---

## Contexte exact du projet

Service `voiceflow-ml/` — FastAPI + PostgreSQL (SQLAlchemy async) + Redis.

### Structure actuelle (etat reel)

```
voiceflow-ml/
├── api/main.py              # FastAPI app — OK
├── api/routes/health.py     # OK
├── api/routes/inference.py  # Logique metier dans le controller
├── api/routes/models.py     # Logique metier dans le controller
├── core/config.py           # Modifie par Agent 1
├── core/database.py         # SQLAlchemy engine + session async
├── core/logging_config.py   # structlog
├── core/redis_client.py     # Redis pool
└── models/diarization/      # ML models — NE PAS TOUCHER
```

### Ce qui manque (tout creer)

```
voiceflow-ml/
├── domain/        # Entites metier pures — zero dependance SQLAlchemy
├── repositories/  # Acces DB uniquement
├── services/      # Logique metier
└── workers/       # Celery tasks asynchrones
```

---

## Regle d'architecture (ADR-001)

- `domain/` : zero import SQLAlchemy, Redis, FastAPI
- `repositories/` : uniquement SQLAlchemy
- `services/` : zero import FastAPI, zero import SQLAlchemy direct
- `api/routes/` : thin controllers <= 40 lignes, zero logique metier

---

## Tache 1 — `domain/job.py` (CREER)

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_path: str = ""
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[dict] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
```

Creer aussi :
- `domain/segment.py` : `Segment(start: float, end: float, speaker_id: str)` + `Speaker(speaker_id, label)`
- `domain/model_version.py` : `ModelVersion(version_id, onnx_path, created_at, metrics: dict)`
- `domain/__init__.py` : exporte BatchJob, JobStatus, Segment, Speaker, ModelVersion

---

## Tache 2 — `repositories/base.py` (CREER)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]: ...

    @abstractmethod
    async def create(self, entity: T) -> T: ...

    @abstractmethod
    async def update(self, entity: T) -> T: ...
```

---

## Tache 3 — `repositories/job_repository.py`

Implemente `BaseRepository[BatchJob]` :
- `create_job(job: BatchJob) -> BatchJob` — INSERT en DB
- `get_job(job_id: str) -> Optional[BatchJob]` — SELECT par ID
- `update_status(job_id: str, status: JobStatus, result: dict | None) -> BatchJob`
- `list_jobs(limit: int = 50, offset: int = 0) -> list[BatchJob]`

Table SQL `batch_jobs` a definir dans `core/database.py`.
Le repository convertit modele SQLAlchemy <-> domain BatchJob.

---

## Tache 4 — `services/inference_service.py`

```python
class InferenceService:
    """
    Orchestration des batch inference jobs.
    Ne depend PAS de FastAPI ou SQLAlchemy directement.
    """

    def __init__(
        self,
        job_repo: JobRepository,
        redis_client: Redis,
        settings: Settings
    ) -> None: ...

    async def submit_batch_job(
        self,
        audio_path: str,
        callback_url: Optional[str] = None
    ) -> BatchJob:
        """Cree un job, persiste en DB, enqueue dans Celery. Retourne PENDING."""
        ...

    async def get_job_status(self, job_id: str) -> BatchJob:
        """Raise JobNotFoundError si job_id inconnu."""
        ...
```

---

## Tache 5 — `workers/celery_app.py` + `workers/inference_tasks.py`

```python
# celery_app.py
from celery import Celery
from core.config import settings

celery_app = Celery(
    "voiceflow_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.inference_tasks"]
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1
```

```python
# inference_tasks.py
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="voiceflow.inference.process_batch"
)
def process_batch_inference(self, job_id: str) -> dict:
    """
    1. Charge job depuis DB
    2. Preprocess audio
    3. Appelle Rust /infer via httpx synchrone
    4. Stocke resultats en DB
    5. POST callback_url si defini
    """
    ...
```

Ajouter dans `requirements.txt` :
- `celery[redis]>=5.3.0`
- `httpx>=0.27.0`

---

## Tache 6 — Refactoring `api/routes/inference.py` (thin controller)

Apres refactoring, <= 40 lignes, zero logique metier :

```python
@router.post("/batch", response_model=JobResponse, status_code=202)
async def submit_batch_inference(
    request: BatchInferenceRequest,
    service: InferenceService = Depends(get_inference_service),
) -> JobResponse:
    """Soumet un job de batch inference. Retourne 202 Accepted."""
    job = await service.submit_batch_job(request.audio_path, request.callback_url)
    return JobResponse.from_domain(job)

@router.get("/batch/{job_id}", response_model=JobResponse)
async def get_batch_status(
    job_id: str,
    service: InferenceService = Depends(get_inference_service),
) -> JobResponse:
    """Retourne le statut et resultats d'un job. Raise 404 si absent."""
    job = await service.get_job_status(job_id)
    return JobResponse.from_domain(job)
```

---

## Contraintes (NE PAS toucher)

- `voiceflow-ml/models/diarization/` — hors scope Agent 2
- `voiceflow-ml/core/` — modifier uniquement pour ajouter la table batch_jobs

---

## Verification finale

```bash
# Architecture propre
grep -r "from sqlalchemy" voiceflow-ml/domain/    # Zero resultat
grep -r "from fastapi" voiceflow-ml/services/     # Zero resultat

# Tests unitaires (sans Docker)
pytest voiceflow-ml/tests/unit/ -v

# Service demarre
docker-compose up ml-service && curl http://localhost:8000/health
```

---

## Handoff pour Agents 3 et 8

Creer `cursor-agents/handoff-agent2.md` :
- [x] domain/ cree : BatchJob, Segment, Speaker, ModelVersion
- [x] repositories/ cree : BaseRepository, JobRepository, ModelRepository
- [x] services/ cree : InferenceService, ModelService, AudioService
- [x] workers/ cree : celery_app + inference_tasks
- [x] Controllers API refactorises (thin, <= 40 lignes)
- [x] POST /api/inference/batch -> 202 + job_id
- [x] GET /api/inference/batch/{job_id} -> statut + resultats
