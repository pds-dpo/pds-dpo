from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from .common import atomic_write_json, join_predictions


def _metric(name: str, stdout: str) -> float:
    match = re.search(rf"(?m)^{re.escape(name)}:\s+([0-9]+(?:\.[0-9]+)?)\s*$", stdout)
    if not match:
        raise ValueError(f"could not parse AMBER metric {name!r} from official output")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned official AMBER generative scorer locally.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--amber-root", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    python = str(args.python) if args.python else os.environ.get("PYTHON", "python")
    joined = join_predictions(args.manifest, args.predictions)
    if len(joined) != 1004:
        raise ValueError(f"AMBER generative split requires 1004 predictions, got {len(joined)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    official_input = args.output_dir / "amber_official_input.json"
    official_input.write_text(
        json.dumps([{"id": source["id"], "response": prediction["response"]} for source, prediction in joined], indent=2),
        encoding="utf-8",
    )
    command = [
        python,
        str(args.amber_root / "inference.py"),
        "--word_association", str(args.amber_root / "data/relation.json"),
        "--safe_words", str(args.amber_root / "data/safe_words.txt"),
        "--annotation", str(args.amber_root / "data/annotations.json"),
        "--metrics", str(args.amber_root / "data/metrics.txt"),
        "--inference_data", str(official_input),
        "--evaluation_type", "g",
    ]
    completed = subprocess.run(command, cwd=args.amber_root, text=True, capture_output=True, check=True)
    (args.output_dir / "official_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.output_dir / "official_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result = {
        "benchmark": "amber_generative",
        "samples": 1004,
        "CHAIR": _metric("CHAIR", completed.stdout),
        "Cover": _metric("Cover", completed.stdout),
        "Hal": _metric("Hal", completed.stdout),
        "Cog": _metric("Cog", completed.stdout),
        "official_revision": "534babf6bbfcce2e735c26289dedfb21cef3c939",
    }
    atomic_write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
