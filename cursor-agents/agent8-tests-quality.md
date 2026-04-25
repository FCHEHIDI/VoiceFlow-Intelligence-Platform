# AGENT 8 — Tests & Qualite de Code
**Role** : Senior QA Engineer + Test Architect
**Duree estimee** : 4-6h
**Prerequis** : Agents 2 (Python arch) + 4 (Rust pipeline) termines

---

## Contexte

### Etat actuel des tests

```
voiceflow-ml/tests/               # A creer ou completer
voiceflow-ml/pytest.ini           # Existe — verifier configuration
voiceflow-inference/tests/inference_tests.rs  # Existe — completer
```

### Objectifs de coverage

| Layer | Cible |
|-------|-------|
| `domain/` | 100% (logique pure, aucune dep externe) |
| `services/` | > 80% (mocks injectes) |
| `repositories/` | > 70% (SQLite in-memory) |
| `api/routes/` | > 70% (FastAPI TestClient) |
| Rust `clustering.rs` | > 80% (logique critique diarization) |
| Rust `sliding_window.rs` | > 80% |

---

## Tache 1 — Structure de tests Python a creer

```
voiceflow-ml/tests/
├── conftest.py                  # Fixtures globales
├── unit/
│   ├── __init__.py
│   ├── test_domain.py           # BatchJob, Segment, ModelVersion
│   ├── test_services.py         # InferenceService avec mocks
│   └── test_audio_validation.py # Magic bytes, taille, format
├── integration/
│   ├── __init__.py
│   ├── test_api_inference.py    # POST /batch, GET /batch/{id}
│   ├── test_api_models.py       # GET /models, POST /models/train
│   └── test_repositories.py    # JobRepository avec SQLite async
└── fixtures/
    ├── audio/
    │   └── sample_16k_mono.wav  # 5s, 16kHz, mono, valide
    ├── ref.rttm                 # RTTM reference pour DER tests
    └── hyp.rttm                 # RTTM hypothese pour DER tests
```

---

## Tache 2 — `tests/conftest.py`

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import wave, io, struct

@pytest.fixture
def test_client():
    from api.main import app
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_job_repo():
    from domain.job import BatchJob, JobStatus
    repo = AsyncMock()
    repo.create_job.return_value = BatchJob(
        job_id="test-job-123",
        status=JobStatus.PENDING
    )
    repo.get_job.return_value = BatchJob(
        job_id="test-job-123",
        status=JobStatus.COMPLETED,
        result={"segments": []}
    )
    return repo

@pytest.fixture
def mock_redis():
    return AsyncMock()

@pytest.fixture
def sample_wav_bytes() -> bytes:
    """5s de silence PCM, 16kHz, mono, 16-bit — WAV valide."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b'\x00\x00' * 80000)  # 5s silence
    return buf.getvalue()

@pytest.fixture
def fake_zip_bytes() -> bytes:
    """ZIP magic bytes — doit etre rejete par le middleware audio."""
    return b'PK\x03\x04' + b'\x00' * 200
```

---

## Tache 3 — Tests unitaires services

```python
# tests/unit/test_services.py
import pytest
from domain.job import BatchJob, JobStatus

class TestInferenceService:

    @pytest.mark.asyncio
    async def test_submit_creates_pending_job(self, mock_job_repo, mock_redis):
        from services.inference_service import InferenceService
        service = InferenceService(mock_job_repo, mock_redis, settings)
        job = await service.submit_batch_job("/data/test.wav")

        assert job.status == JobStatus.PENDING
        mock_job_repo.create_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_status_raises_if_not_found(self, mock_job_repo, mock_redis):
        from services.inference_service import InferenceService, JobNotFoundError
        mock_job_repo.get_job.return_value = None
        service = InferenceService(mock_job_repo, mock_redis, settings)

        with pytest.raises(JobNotFoundError):
            await service.get_job_status("nonexistent-id")

    @pytest.mark.asyncio
    async def test_submit_rejects_empty_audio_path(self, mock_job_repo, mock_redis):
        from services.inference_service import InferenceService
        service = InferenceService(mock_job_repo, mock_redis, settings)

        with pytest.raises(ValueError, match="audio_path"):
            await service.submit_batch_job("")
```

---

## Tache 4 — Tests integration API

```python
# tests/integration/test_api_inference.py
class TestInferenceAPI:

    def test_submit_batch_returns_202(self, test_client):
        response = test_client.post("/api/inference/batch", json={
            "audio_path": "/data/test.wav"
        })
        assert response.status_code == 202
        body = response.json()
        assert "job_id" in body
        assert body["status"] == "pending"

    def test_get_job_status_returns_job(self, test_client):
        submit = test_client.post("/api/inference/batch", json={"audio_path": "/data/test.wav"})
        job_id = submit.json()["job_id"]

        response = test_client.get(f"/api/inference/batch/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] in ["pending", "processing", "completed", "failed"]

    def test_get_unknown_job_returns_404(self, test_client):
        response = test_client.get("/api/inference/batch/nonexistent-id-000")
        assert response.status_code == 404

    def test_health_returns_200(self, test_client):
        assert test_client.get("/health").status_code == 200

    def test_submit_missing_audio_path_returns_422(self, test_client):
        response = test_client.post("/api/inference/batch", json={})
        assert response.status_code == 422
```

---

## Tache 5 — Tests securite validation audio

```python
# tests/unit/test_audio_validation.py
import pytest
from fastapi import HTTPException

class TestAudioValidation:

    @pytest.mark.asyncio
    async def test_valid_wav_passes(self, sample_wav_bytes):
        from api.middleware.input_validation import validate_audio_upload
        file = MockUploadFile(content=sample_wav_bytes, filename="test.wav")
        await validate_audio_upload(file)  # Ne doit pas lever d'exception

    @pytest.mark.asyncio
    async def test_zip_disguised_as_wav_rejected(self, fake_zip_bytes):
        from api.middleware.input_validation import validate_audio_upload
        file = MockUploadFile(content=fake_zip_bytes, filename="malicious.wav")
        with pytest.raises(HTTPException) as exc:
            await validate_audio_upload(file)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_oversized_file_rejected(self, sample_wav_bytes):
        from api.middleware.input_validation import validate_audio_upload
        large = sample_wav_bytes * 2000  # Bien > 100MB
        file = MockUploadFile(content=large, filename="huge.wav")
        with pytest.raises(HTTPException) as exc:
            await validate_audio_upload(file, max_size_mb=1)
        assert exc.value.status_code == 413
```

---

## Tache 6 — Tests Rust (completer `tests/inference_tests.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn normalize(v: Array1<f32>) -> Array1<f32> {
        let norm = v.dot(&v).sqrt();
        if norm > 1e-8 { v / norm } else { v }
    }

    #[test]
    fn test_sliding_window_produces_correct_number_of_windows() {
        let audio = vec![0.0_f32; 80000];  // 5s at 16kHz
        let windows = sliding_window(&audio, 3.0, 1.0, 16000);
        // Fenetres attendues : [0-3s], [1-4s], [2-5s] = 3 fenetres
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].0, 0.0);
        assert_eq!(windows[0].1, 3.0);
        assert_eq!(windows[0].2.len(), 48000);
    }

    #[test]
    fn test_clusterer_same_embedding_same_speaker() {
        let mut c = OnlineClusterer::new();
        let e1 = normalize(Array1::from_vec(vec![1.0_f32; 512]));
        let e2 = normalize(Array1::from_vec(vec![0.999_f32; 512]));
        let id1 = c.add_embedding(e1, 0.0);
        let id2 = c.add_embedding(e2, 1.0);
        assert_eq!(id1, id2, "Embeddings quasi-identiques -> meme speaker");
    }

    #[test]
    fn test_clusterer_orthogonal_embeddings_different_speakers() {
        let mut c = OnlineClusterer::new();
        let e_a = normalize(Array1::from_vec(vec![1.0_f32; 512]));
        let e_b = normalize(Array1::from_vec(
            (0..512).map(|i| if i < 256 { 1.0_f32 } else { -1.0_f32 }).collect()
        ));
        let id_a = c.add_embedding(e_a, 0.0);
        let id_b = c.add_embedding(e_b, 1.0);
        assert_ne!(id_a, id_b, "Embeddings orthogonaux -> speakers differents");
    }

    #[tokio::test]
    async fn test_health_endpoint_returns_200() {
        let app = build_test_router();
        let response = app.oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(axum::body::Body::empty())
                .unwrap()
        ).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }
}
```

---

## Tache 7 — Load test WebSocket (`voiceflow-ml/load_test.py`)

Mettre a jour pour tester 100 clients simultanes :

```python
async def run_load_test(
    concurrent: int = 100,
    url: str = "ws://localhost:3000/ws/stream",
    chunks_per_client: int = 10
) -> None:
    """
    Simule N clients WebSocket envoyant des chunks audio.
    Affiche les latences P50 et P99.
    Cible : P99 < 100ms pour 100 clients.
    """
    ...
```

---

## Configuration `pytest.ini` (verifier/mettre a jour)

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
markers =
    unit: Tests unitaires sans services externes
    integration: Tests d'integration (necessite DB)
    slow: Tests lents (load, DER)

addopts =
    --strict-markers
    -v
    --tb=short
    --cov=.
    --cov-report=term-missing
    --cov-fail-under=70
    --ignore=tests/fixtures
```

---

## Verification finale

```bash
# Tests unitaires (sans Docker)
pytest voiceflow-ml/tests/unit/ -v -m unit
# Tous les tests doivent passer

# Tests integration
docker-compose up postgres redis -d
pytest voiceflow-ml/tests/integration/ -v -m integration

# Coverage globale
pytest voiceflow-ml/ --cov=voiceflow-ml --cov-report=html
# Ouvrir htmlcov/index.html -> verifier >= 70%

# Tests Rust
cargo test --manifest-path voiceflow-inference/Cargo.toml --verbose

# Load test
python voiceflow-ml/load_test.py
# -> P99 < 100ms pour 100 clients
```
