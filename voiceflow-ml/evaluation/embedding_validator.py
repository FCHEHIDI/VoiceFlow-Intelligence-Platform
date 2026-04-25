"""Validate the quality of embeddings produced by an exported ONNX model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore[assignment]


class EmbeddingValidator:
    """Inspect an ONNX model's output node and probe embedding quality."""

    def __init__(
        self,
        intra_speaker_threshold: float = 0.4,
        inter_speaker_threshold: float = 0.7,
        target_dim: int = 512,
    ) -> None:
        self.intra_speaker_threshold = intra_speaker_threshold
        self.inter_speaker_threshold = inter_speaker_threshold
        self.target_dim = target_dim

    def validate_onnx_model(
        self,
        onnx_path: str,
        test_audio_dir: str | None = None,
    ) -> dict[str, Any]:
        if not Path(onnx_path).is_file():
            raise FileNotFoundError(onnx_path)
        if ort is None or np is None:  # pragma: no cover
            return {
                "embedding_dim": -1,
                "is_normalized": False,
                "onnx_output_node_name": "",
                "avg_intra_speaker_distance": float("nan"),
                "avg_inter_speaker_distance": float("nan"),
                "threshold_accuracy": float("nan"),
                "validation_passed": False,
                "warning": "onnxruntime or numpy not installed",
            }

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()
        if not outputs:
            raise RuntimeError(f"ONNX model {onnx_path} has no output nodes")

        output_node = outputs[0]
        output_name = output_node.name
        output_shape = list(output_node.shape)

        embedding_dim = -1
        for dim in reversed(output_shape):
            if isinstance(dim, int) and dim > 0:
                embedding_dim = dim
                break

        report: dict[str, Any] = {
            "onnx_output_node_name": output_name,
            "embedding_dim": embedding_dim,
            "is_normalized": False,
            "avg_intra_speaker_distance": float("nan"),
            "avg_inter_speaker_distance": float("nan"),
            "threshold_accuracy": float("nan"),
            "validation_passed": False,
            "input_names": [i.name for i in session.get_inputs()],
            "input_shapes": [list(i.shape) for i in session.get_inputs()],
        }

        if test_audio_dir and Path(test_audio_dir).is_dir():
            embeddings = self._probe_with_audio(session, test_audio_dir)
            if embeddings is not None and embeddings.shape[0] >= 2:
                report.update(self._embedding_stats(embeddings))

        report["validation_passed"] = (
            embedding_dim == self.target_dim
            and bool(report.get("is_normalized", False))
        )
        return report

    def _probe_with_audio(self, session: "ort.InferenceSession", audio_dir: str) -> "np.ndarray | None":
        try:
            import wave
        except Exception:  # pragma: no cover
            return None

        wavs = sorted(Path(audio_dir).glob("*.wav"))[: 8]
        if not wavs:
            return None

        embeddings = []
        input_meta = session.get_inputs()[0]
        for wav_path in wavs:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if audio.size == 0:
                continue
            audio = audio[: 16000 * 3] if audio.size >= 16000 * 3 else np.pad(audio, (0, 16000 * 3 - audio.size))
            tensor = audio.reshape(1, -1)
            try:
                outputs = session.run(None, {input_meta.name: tensor.astype(np.float32)})
            except Exception:
                return None
            embeddings.append(np.asarray(outputs[0]).reshape(-1))
        if not embeddings:
            return None
        return np.stack(embeddings)

    def _embedding_stats(self, embeddings: "np.ndarray") -> dict[str, Any]:
        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = bool(np.allclose(norms, 1.0, atol=1e-3))
        sims = embeddings @ embeddings.T
        n = sims.shape[0]
        intra = []
        inter = []
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - float(sims[i, j])
                # Without speaker labels, treat the two halves as different speakers as a heuristic.
                (intra if abs(i - j) <= 1 else inter).append(d)
        return {
            "is_normalized": is_normalized,
            "avg_intra_speaker_distance": float(np.mean(intra)) if intra else float("nan"),
            "avg_inter_speaker_distance": float(np.mean(inter)) if inter else float("nan"),
            "threshold_accuracy": float(np.mean(np.array(inter) > self.inter_speaker_threshold))
            if inter
            else float("nan"),
        }
