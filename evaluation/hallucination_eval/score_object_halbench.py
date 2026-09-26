from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from .common import atomic_write_json, join_predictions, read_jsonl


def _load_official(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rlhfv_object_halbench", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CHAIR stage using pinned RLHF-V Object HalBench code.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    joined = join_predictions(args.manifest, args.predictions)
    if len(joined) != 300:
        raise ValueError("Object HalBench requires 300 predictions")
    extraction_rows = list(read_jsonl(args.extractions))
    extraction_by_id = {row["id"]: row for row in extraction_rows}
    if len(extraction_by_id) != 300 or set(extraction_by_id) != {source["id"] for source, _ in joined}:
        raise ValueError("Object HalBench extraction coverage mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", dir=args.output.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        for source, prediction in joined:
            handle.write(
                json.dumps(
                    {
                        "image_id": source["image_id"],
                        "question": source["prompt"],
                        "text": prediction["response"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    original_cwd = Path.cwd()
    vendor_root = args.official_evaluator.resolve().parents[1]
    try:
        os.chdir(vendor_root)
        official = _load_official(args.official_evaluator.resolve())
        image_ids = {int(source["image_id"]) for source, _ in joined}
        evaluator = official.CHAIR(image_ids, str(args.coco_annotations.resolve()), "")
        evaluator.get_annotations()
        extraction_by_image = {
            int(source["image_id"]): extraction_by_id[source["id"]]["objects"] for source, _ in joined
        }

        def frozen_extractions():
            data = evaluator.caps
            for row in data:
                row["extract_objs"] = extraction_by_image[int(row["image_id"])]
            return data, {}, {}

        evaluator.gpt_caption_processor = frozen_extractions
        metrics = evaluator.compute_chair(str(temporary), -1, gpt_process=True)
    finally:
        os.chdir(original_cwd)
        temporary.unlink(missing_ok=True)
    overall = metrics["overall_metrics"]
    result = {
        "benchmark": "object_halbench",
        "samples": 300,
        "response_hallucination": overall["CHAIRs_refine"] * 100,
        "mention_hallucination": overall["CHAIRi"] * 100,
        "response_correct": overall["correct_rate"] * 100,
        "object_correct": overall["object_correct_rate"] * 100,
        "object_recall": overall["obj_rec"] * 100,
        "average_length": overall["avg_word_len"],
        "official_overall_metrics": overall,
        "judge_models_returned": sorted({row["judge_returned"] for row in extraction_rows}),
        "historical_parity": False,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
