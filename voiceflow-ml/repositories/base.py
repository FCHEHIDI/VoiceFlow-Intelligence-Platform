"""Abstract repository contract.

Concrete repositories are responsible for translating between SQLAlchemy ORM
models and pure domain entities. The service layer must depend on this
abstraction, never on SQLAlchemy directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Generic CRUD-ish contract for repositories using a sync Session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Return the entity if it exists, else ``None``."""

    @abstractmethod
    def create(self, entity: T) -> T:
        """Persist a new entity. Returns the stored entity."""

    @abstractmethod
    def update(self, entity: T) -> T:
        """Update an existing entity. Returns the updated entity."""

    @abstractmethod
    def list(self, limit: int = 50, offset: int = 0) -> Sequence[T]:
        """List entities with pagination."""
