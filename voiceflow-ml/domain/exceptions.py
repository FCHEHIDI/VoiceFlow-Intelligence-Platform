"""Domain-level exceptions (no framework dependencies)."""
from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class JobNotFoundError(DomainError):
    """Raised when a referenced job cannot be located."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Batch job not found: {job_id}")
        self.job_id = job_id


class InvalidJobInputError(DomainError):
    """Raised when input arguments to a job operation are invalid."""
