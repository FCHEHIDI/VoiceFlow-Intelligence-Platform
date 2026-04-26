"""Export a tiny stub ONNX model for end-to-end smoke testing.

The real diarization weights are not committed to the repo, so this stub
produces the input/output node names expected by the Rust engine and the
Agent 3 embedding pipeline. It is *not* a trained model.

Inputs:
    audio: float32, shape [batch, audio_length] — raw 16 kHz PCM samples.

Outputs:
    speaker_probabilities: float32, shape [batch, 2] — softmaxed dummy logits.
    embedding:             float32, shape [batch, 192] — pooled latent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class StubDiarizer(nn.Module):
    def __init__(self, embedding_dim: int = 192, num_speakers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(1, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_speakers)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # audio: [B, T] -> [B, T, 1]
        x = audio.unsqueeze(-1)
        # mean-pool over time -> [B, 1]
        pooled = x.mean(dim=1)
        embedding = self.proj(pooled)
        logits = self.head(embedding)
        speaker_probabilities = torch.softmax(logits, dim=-1)
        return speaker_probabilities, embedding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX file path")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = StubDiarizer().eval()
    dummy = torch.randn(1, 16_000, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        args.out.as_posix(),
        input_names=["audio"],
        output_names=["speaker_probabilities", "embedding"],
        dynamic_axes={
            "audio": {0: "batch", 1: "audio_length"},
            "speaker_probabilities": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Wrote stub ONNX to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
