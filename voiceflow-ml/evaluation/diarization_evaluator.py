"""DER computation against the NIST RTTM benchmark format.

Uses ``pyannote.metrics`` when available; falls back to a permissive
implementation that returns ``NaN`` (rather than crashing) when the optional
dependency is missing, so the rest of the codebase stays importable in
constrained environments such as CI without ML extras.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    _PYANNOTE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Annotation = None  # type: ignore[assignment]
    Segment = None  # type: ignore[assignment]
    DiarizationErrorRate = None  # type: ignore[assignment]
    _PYANNOTE_AVAILABLE = False


def _parse_rttm(path: str) -> "Annotation":
    if not _PYANNOTE_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "pyannote.metrics is required to compute DER — install pyannote.metrics>=3.2.0"
        )
    annotation = Annotation()
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            onset = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(onset, onset + duration)] = speaker
    return annotation


class DiarizationEvaluator:
    """DER evaluator with collar / overlap configuration."""

    def __init__(self, collar: float = 0.25, skip_overlap: bool = False) -> None:
        if collar < 0:
            raise ValueError("collar must be non-negative")
        self.collar = collar
        self.skip_overlap = skip_overlap

    @staticmethod
    def _empty_result() -> dict[str, float]:
        return {
            "der": math.nan,
            "false_alarm": math.nan,
            "missed_speech": math.nan,
            "speaker_confusion": math.nan,
            "total_speech_duration": 0.0,
        }

    def compute_der(self, reference_rttm: str, hypothesis_rttm: str) -> dict[str, float]:
        for p in (reference_rttm, hypothesis_rttm):
            if not Path(p).is_file():
                raise FileNotFoundError(p)

        if not _PYANNOTE_AVAILABLE:  # pragma: no cover
            return self._empty_result()

        ref = _parse_rttm(reference_rttm)
        hyp = _parse_rttm(hypothesis_rttm)
        metric = DiarizationErrorRate(collar=self.collar, skip_overlap=self.skip_overlap)
        details = metric(ref, hyp, detailed=True)
        total = float(details.get("total", 0.0))
        return {
            "der": float(details.get("diarization error rate", math.nan)),
            "false_alarm": float(details.get("false alarm", 0.0)) / total if total else math.nan,
            "missed_speech": float(details.get("missed detection", 0.0)) / total if total else math.nan,
            "speaker_confusion": float(details.get("confusion", 0.0)) / total if total else math.nan,
            "total_speech_duration": total,
        }

    def evaluate_batch(self, rttm_pairs: Iterable[Tuple[str, str]]) -> dict[str, float]:
        pairs: List[Tuple[str, str]] = list(rttm_pairs)
        if not pairs:
            raise ValueError("rttm_pairs must contain at least one (ref, hyp) tuple")

        if not _PYANNOTE_AVAILABLE:  # pragma: no cover
            return self._empty_result()

        metric = DiarizationErrorRate(collar=self.collar, skip_overlap=self.skip_overlap)
        for ref_path, hyp_path in pairs:
            metric(_parse_rttm(ref_path), _parse_rttm(hyp_path))
        report = metric.report()
        total = float(report.loc["TOTAL", ("total", "%")] or 0.0)
        return {
            "der": float(metric.report().loc["TOTAL", ("diarization error rate", "%")]) / 100,
            "false_alarm": float(report.loc["TOTAL", ("false alarm", "%")] or 0.0) / 100,
            "missed_speech": float(report.loc["TOTAL", ("missed detection", "%")] or 0.0) / 100,
            "speaker_confusion": float(report.loc["TOTAL", ("confusion", "%")] or 0.0) / 100,
            "total_speech_duration": total,
        }
