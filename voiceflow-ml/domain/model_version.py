"""Model version domain entity."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ModelVersion:
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "0.0.0"
    onnx_path: str = ""
    created_at: datetime = field(default_factory=_now_utc)
    metrics: dict[str, Any] = field(default_factory=dict)
    is_production: bool = False
    architecture: Optional[str] = None

    def promote_to_production(self) -> None:
        self.is_production = True
