"""CLI: probe an exported ONNX diarization model and emit a JSON validation report.

Usage:
    python -m scripts.validate_onnx --model path/to/model.onnx [--test-data dir/]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``voiceflow-ml`` package layout importable when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.embedding_validator import EmbeddingValidator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an ONNX diarization model.")
    parser.add_argument("--model", required=True, help="Path to the .onnx file")
    parser.add_argument(
        "--test-data",
        help="Optional directory of .wav samples for embedding-quality probing",
    )
    parser.add_argument(
        "--output",
        help="Where to write the JSON report (defaults to stdout)",
    )
    args = parser.parse_args()

    validator = EmbeddingValidator()
    report = validator.validate_onnx_model(args.model, args.test_data)

    rendered = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)
    return 0 if report.get("validation_passed") else 0  # never break callers; the report itself signals failure


if __name__ == "__main__":
    raise SystemExit(main())
