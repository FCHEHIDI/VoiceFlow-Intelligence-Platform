"""Convert a Rust /infer JSON response (or our API shape) to NIST RTTM lines.

Usage::

    python scripts/smoke_infer.py --audio meeting.wav > out.json
    python scripts/segments_to_rttm.py --json out.json --file-id ES2002a --out pred.rttm

Then compute DER with ``evaluation.diarization_evaluator.DiarizationEvaluator``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, help="Path to JSON file (default: stdin)")
    p.add_argument("--file-id", default="unknown", help="RTTM filename / URI field")
    p.add_argument("--out", type=Path, required=True, help="Output .rttm path")
    args = p.parse_args()

    if args.json:
        data = json.loads(args.json.read_text(encoding="utf-8"))
    else:
        data = json.load(sys.stdin)

    segs = data.get("segments") or []
    lines: list[str] = []
    for s in segs:
        spk = str(s.get("speaker_id", "SPEAKER"))
        start = float(s["start"])
        end = float(s["end"])
        dur = max(0.0, end - start)
        # SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
        lines.append(
            f"SPEAKER {args.file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
        )

    args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
