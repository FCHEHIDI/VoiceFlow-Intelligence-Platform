"""Generate canonical audio fixtures for the test suite.

Creates ``tests/fixtures/audio/sample_16k_mono.wav`` — 5 s of silence at
16 kHz, mono, 16-bit PCM — using the stdlib ``wave`` module so there is
no soundfile / librosa / numpy dependency at fixture-build time.

Run this script directly to (re)materialise the file:

    python tests/fixtures/generate_audio.py
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path


SAMPLE_RATE = 16_000
DURATION_SECONDS = 5.0
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def write_silence_wav(path: Path) -> Path:
    """Write a 16 kHz mono 5 s silent WAV at ``path``. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(SAMPLE_RATE)
        # 16-bit silent sample is 0x0000 little-endian.
        wav.writeframes(b"\x00\x00" * n_samples)
    return path


def verify(path: Path) -> dict:
    """Read the file back and return basic header stats."""
    with wave.open(str(path), "rb") as wav:
        info = {
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "sample_rate": wav.getframerate(),
            "n_frames": wav.getnframes(),
        }
    info["file_size_bytes"] = path.stat().st_size
    info["duration_seconds"] = info["n_frames"] / info["sample_rate"]
    return info


def main() -> int:
    here = Path(__file__).resolve().parent
    target = here / "audio" / "sample_16k_mono.wav"
    write_silence_wav(target)
    info = verify(target)

    expected_data_bytes = int(DURATION_SECONDS * SAMPLE_RATE) * SAMPLE_WIDTH_BYTES * CHANNELS
    # WAV header is 44 bytes for the canonical PCM-only RIFF/WAVE layout.
    expected_total = expected_data_bytes + 44

    print(f"Wrote: {target}")
    print(f"  size           = {info['file_size_bytes']} bytes (expected ~{expected_total})")
    print(f"  channels       = {info['channels']}")
    print(f"  sample_rate    = {info['sample_rate']} Hz")
    print(f"  sample_width   = {info['sample_width_bytes'] * 8}-bit")
    print(f"  duration       = {info['duration_seconds']:.3f} s")

    if info["sample_rate"] != SAMPLE_RATE:
        print("ERROR: unexpected sample rate", file=sys.stderr)
        return 1
    if info["channels"] != CHANNELS:
        print("ERROR: unexpected channel count", file=sys.stderr)
        return 1
    if info["sample_width_bytes"] != SAMPLE_WIDTH_BYTES:
        print("ERROR: unexpected sample width", file=sys.stderr)
        return 1
    if abs(info["duration_seconds"] - DURATION_SECONDS) > 1e-6:
        print("ERROR: unexpected duration", file=sys.stderr)
        return 1
    if info["file_size_bytes"] < expected_total - 8:
        print("ERROR: file smaller than expected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
