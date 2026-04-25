"""Service layer — business logic.

Services depend on repositories and infrastructure clients via abstractions,
never on FastAPI / SQLAlchemy directly (ADR-001).
"""
from services.inference_service import InferenceService

__all__ = ["InferenceService"]
