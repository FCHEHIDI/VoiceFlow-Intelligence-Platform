"""Thin controllers for batch / sync inference (ADR-001)."""
from __future__ import annotations

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from api.dependencies import get_inference_service
from api.middleware.input_validation import validate_audio_upload
from domain.exceptions import InvalidJobInputError, JobNotFoundError
from domain.job import BatchJob
from services.inference_service import InferenceService

router = APIRouter()
logger = structlog.get_logger(__name__)


class BatchInferenceRequest(BaseModel):
    audio_path: str = Field(..., min_length=1, description="Server-side path to the audio file")
    callback_url: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    audio_path: str
    callback_url: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    model_version: Optional[str] = None

    @classmethod
    def from_domain(cls, job: BatchJob) -> "JobResponse":
        return cls(
            job_id=job.job_id,
            status=job.status.value,
            audio_path=job.audio_path,
            callback_url=job.callback_url,
            result=job.result,
            error=job.error,
            model_version=job.model_version,
        )


class SegmentDTO(BaseModel):
    start: float
    end: float
    speaker_id: str
    confidence: float


@router.post("/batch", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_batch_inference(
    request: BatchInferenceRequest,
    service: InferenceService = Depends(get_inference_service),
) -> JobResponse:
    try:
        job = service.submit_batch_job(request.audio_path, request.callback_url)
    except InvalidJobInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JobResponse.from_domain(job)


@router.get("/batch/{job_id}", response_model=JobResponse)
def get_batch_status(
    job_id: str,
    service: InferenceService = Depends(get_inference_service),
) -> JobResponse:
    try:
        job = service.get_job_status(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JobResponse.from_domain(job)


@router.post("/sync", response_model=List[SegmentDTO])
async def sync_inference(file: UploadFile = File(...)) -> List[SegmentDTO]:
    """Validate the upload only — synchronous inference is delegated to the Rust service."""
    await validate_audio_upload(file)
    logger.info("sync_inference_request", filename=file.filename)
    return []
