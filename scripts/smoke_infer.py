"""Live smoke-test for the Rust inference engine.

Generates ~6 s of synthetic 16 kHz PCM audio (two alternating tones to simulate
a two-speaker conversation), POSTs it to /infer, and prints the response.
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request

SAMPLE_RATE = 16_000
DURATION_S = 6.0
BASE_URL = "http://127.0.0.1:3000"


def synth_audio() -> list[float]:
    samples: list[float] = []
    n = int(SAMPLE_RATE * DURATION_S)
    for i in range(n):
        t = i / SAMPLE_RATE
        # Speaker A (440 Hz) for first half, Speaker B (880 Hz) for second half.
        freq = 440.0 if t < DURATION_S / 2 else 880.0
        samples.append(0.3 * math.sin(2 * math.pi * freq * t))
    return samples


def main() -> int:
    payload = {
        "audio": synth_audio(),
        "sample_rate": SAMPLE_RATE,
        "window_secs": 1.0,
        "hop_secs": 0.5,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
