from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .common import EXPECTED_COUNTS, atomic_write_json, atomic_write_jsonl, file_record


def _verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.convert("RGB").load()


def _mmhal_rows(suite: Path) -> list[dict]:
    source = suite / "assets/mmhal-bench/response_template.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(records):
        filename = Path(urlparse(record["image_src"]).path).name
        image_path = suite / "assets/mmhal-bench/images" / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(
            {
                "benchmark": "mmhal",
                "id": index,
                "image_id": record["image_id"],
                "image_path": str(image_path),
                "prompt": record["question"],
                "question_type": record["question_type"],
                "question_topic": record["question_topic"],
                "image_content": record["image_content"],
                "gt_answer": record["gt_answer"],
            }
        )
    return rows


def _object_rows(suite: Path) -> list[dict]:
    source = suite / "vendor/rlhf-v/eval/data/obj_halbench_300_with_image.jsonl"
    image_dir = suite / "assets/object_halbench/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            image_path = image_dir / f"{int(record['image_id']):012d}.jpg"
            payload = base64.b64decode(record["image"], validate=True)
            if not image_path.exists():
                temporary = image_path.with_suffix(".jpg.part")
                temporary.write_bytes(payload)
                os.replace(temporary, image_path)
            elif image_path.read_bytes() != payload:
                raise ValueError(f"extracted Object HalBench image mismatch: {image_path}")
            rows.append(
                {
                    "benchmark": "object_halbench",
                    "id": index,
                    "org_idx": int(record["org_idx"]),
                    "image_id": int(record["image_id"]),
                    "image_path": str(image_path),
                    "prompt": record["question"],
                }
            )
    return rows


def _amber_rows(suite: Path) -> list[dict]:
    source = suite / "vendor/amber/data/query/query_generative.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(records):
        image_path = suite / "assets/amber/image" / record["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(
            {
                "benchmark": "amber",
                "id": int(record["id"]),
                "ordinal": index,
                "image_path": str(image_path),
                "prompt": record["query"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--verify-all-images", action="store_true")
    args = parser.parse_args()
    suite = args.suite_root.resolve()
    builders = {
        "mmhal": _mmhal_rows,
        "object_halbench": _object_rows,
        "amber": _amber_rows,
    }
    manifest_dir = suite / "manifests"
    summary = {"schema_version": 1, "suite_root": str(suite), "manifests": {}}
    for benchmark, builder in builders.items():
        rows = builder(suite)
        if len(rows) != EXPECTED_COUNTS[benchmark]:
            raise ValueError(f"{benchmark}: expected {EXPECTED_COUNTS[benchmark]}, got {len(rows)}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{benchmark}: duplicate ids")
        if args.verify_all_images:
            for row in rows:
                _verify_image(Path(row["image_path"]))
        else:
            for row in (rows[0], rows[len(rows) // 2], rows[-1]):
                _verify_image(Path(row["image_path"]))
        path = manifest_dir / f"{benchmark}.jsonl"
        atomic_write_jsonl(path, rows)
        summary["manifests"][benchmark] = {**file_record(path), "rows": len(rows)}
    atomic_write_json(manifest_dir / "manifest_index.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
