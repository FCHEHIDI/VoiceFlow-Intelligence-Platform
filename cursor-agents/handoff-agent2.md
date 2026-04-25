# Agent 2 Handoff — Python Clean Architecture

## Layout produced
```
voiceflow-ml/
├── domain/
│   ├── __init__.py
│   ├── exceptions.py          # DomainError, JobNotFoundError, InvalidJobInputError
│   ├── job.py                 # BatchJob + JobStatus enum + transitions
│   ├── model_version.py       # ModelVersion entity
│   └── segment.py             # Segment, Speaker
├── repositories/
│   ├── __init__.py
│   ├── base.py                # BaseRepository[T] generic
│   └── job_repository.py      # JobRepository (sync SQLAlchemy + BatchJobModel)
├── services/
│   ├── __init__.py
│   └── inference_service.py   # InferenceService (no FastAPI/SQLA imports)
├── workers/
│   ├── __init__.py
│   ├── celery_app.py          # celery_app + CeleryTaskQueue adapter
│   └── inference_tasks.py     # process_batch_inference task
└── api/
    ├── dependencies.py        # get_inference_service / get_job_repository
    └── routes/inference.py    # thin controllers — submit/get_batch + sync
```

## Architecture invariants enforced
- `domain/` — zero imports of SQLAlchemy / FastAPI / Redis (pure Python).
- `repositories/` — only place outside `core/` to import SQLAlchemy.
- `services/` — depends only on the `_JobRepoProtocol` and `_TaskQueueProtocol` typing protocols, never on a framework.
- `api/routes/inference.py` — every controller body is < 15 lines, returns DTOs via `JobResponse.from_domain`.

## Endpoints
| Method | Path | Status | Description |
|--------|------|--------|-------------|
| POST | `/api/inference/batch` | 202 Accepted | Persists a `BatchJob` PENDING and enqueues a Celery task. |
| GET | `/api/inference/batch/{job_id}` | 200 / 404 | Returns the current job state. |
| POST | `/api/inference/sync` | 200 | Validates the upload (magic bytes, max size). |

## Worker
- `celery_app` connects to `settings.redis_url` for broker + backend.
- `CeleryTaskQueue.enqueue_inference(job_id)` uses task name `voiceflow.inference.process_batch` and the `inference` queue.
- `process_batch_inference` walks the job through PENDING → PROCESSING → COMPLETED|FAILED, invokes the Rust service via `httpx` and posts the optional callback URL.

## Requirements
- `celery[redis]>=5.3.4`, `httpx>=0.27.0` already present (updated in `requirements.txt`).

## Verification
```powershell
Set-Location voiceflow-ml; rg "from sqlalchemy" domain  # zero match
Set-Location voiceflow-ml; rg "from fastapi" services    # zero match
```
