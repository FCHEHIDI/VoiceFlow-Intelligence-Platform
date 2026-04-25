"""Speaker / segment domain entities — pure, framework-agnostic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise ValueError("speaker_id must not be empty")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Segment timestamps must be non-negative")
        if self.end <= self.start:
            raise ValueError("Segment end must be greater than start")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Segment confidence must be in [0, 1]")

    @property
    def duration(self) -> float:
        return self.end - self.start
