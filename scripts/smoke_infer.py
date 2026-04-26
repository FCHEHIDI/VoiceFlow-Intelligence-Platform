"""Live smoke-test for the Rust inference engine.

Without ``--audio``: generates ~6 s of synthetic two-tone audio and POSTs to ``/infer``.

With ``--audio``: loads a PCM WAV (16-bit mono **16 kHz** recommended), optionally
trims with ``--max-seconds``, and POSTs to ``/infer``. Use ffmpeg to resample::

    ffmpeg -y -i raw.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
DURATION_S = 6.0
DEFAULT_BASE = "http://127.0.0.1:3000"


def synth_audio() -> list[float]:
    samples: list[float] = []
    n = int(SAMPLE_RATE * DURATION_S)
    for i in range(n):
        t = i / SAMPLE_RATE
        freq = 440.0 if t < DURATION_S / 2 else 880.0
        samples.append(0.3 * math.sin(2 * math.pi * freq * t))
    return samples


def load_wav_f32(path: Path, max_seconds: float | None) -> tuple[list[float], int]:
    """Return (samples as float32-ish list, sample_rate). Only 16-bit PCM supported."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        if sampwidth != 2:
            raise SystemExit(
                f"{path}: need 16-bit PCM WAV (sampwidth={sampwidth}); "
                "convert with: ffmpeg -i in.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav"
            )

        if framerate != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: sample rate is {framerate} Hz; engine expects {SAMPLE_RATE} Hz. "
                f"Run: ffmpeg -y -i {path.name} -ac 1 -ar {SAMPLE_RATE} -sample_fmt s16 fixed.wav"
            )

        if max_seconds is not None:
            max_frames = int(framerate * max_seconds)
            n_frames = min(n_frames, max_frames)

        raw = wf.readframes(n_frames)

    # int16 little-endian stereo/mono
    fmt = "<" + "h" * n_channels
    frame_size = 2 * n_channels
    samples: list[float] = []
    for off in range(0, len(raw), frame_size):
        chunk = raw[off : off + frame_size]
        if len(chunk) < frame_size:
            break
        if n_channels == 1:
            (s,) = struct.unpack("<h", chunk)
            samples.append(s / 32768.0)
        else:
            shorts = struct.unpack("<" + "h" * n_channels, chunk)
            samples.append(sum(shorts) / float(n_channels) / 32768.0)

    return samples, framerate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="16 kHz mono 16-bit PCM WAV (use ffmpeg to convert)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help=f"Inference base URL (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="When using --audio, only send the first N seconds (default: 120). "
        "Use a smaller value for quick iteration on long meetings.",
    )
    parser.add_argument("--window-secs", type=float, default=3.0)
    parser.add_argument("--hop-secs", type=float, default=1.0)
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Send the full WAV (careful: long files produce huge JSON bodies).",
    )
    args = parser.parse_args()

    if args.audio is not None:
        if not args.audio.is_file():
            print(f"File not found: {args.audio}", file=sys.stderr)
            return 1
        max_s = None if args.no_trim else args.max_seconds
        audio, sr = load_wav_f32(args.audio, max_s)
        print(
            f"Loaded {args.audio.name}: {len(audio) / sr:.2f} s @ {sr} Hz "
            f"({len(audio)} samples)",
            file=sys.stderr,
        )
    else:
        audio = synth_audio()
        sr = SAMPLE_RATE

    payload = {
        "audio": audio,
        "sample_rate": sr,
        "window_secs": args.window_secs,
        "hop_secs": args.hop_secs,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        print("Is the Rust engine running with JWT_SECRET_KEY and MODELS_DIR set?", file=sys.stderr)
        return 1

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
