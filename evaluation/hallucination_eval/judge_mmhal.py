from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import time
from pathlib import Path

from openai import OpenAI

from .common import atomic_write_json, join_predictions, read_jsonl
from .rating_parser import rating as _rating, PARSER_VERSION


def _load_template(source: Path) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "template" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"template assignment not found in {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paid MMHal judge stage; requires an environment-only API key.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--judge-model", required=True, help="A fixed, supported model snapshot; never use an alias silently")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY must be set in the environment")
    joined = join_predictions(args.manifest, args.predictions)
    if len(joined) != 96:
        raise ValueError(f"MMHal requires 96 predictions, got {len(joined)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "judge_responses.jsonl"
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(args.output_dir / "summary.json")
    completed = {row["id"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}
    if raw_path.exists() and len(completed)!=len(list(read_jsonl(raw_path))):
        raise ValueError('Duplicate saved judgments')
    if not set(completed) <= {s['id'] for s,_ in joined}:
        raise ValueError('Unexpected saved judgment IDs')
    for row in completed.values():
        if row['judge_returned']!=args.judge_model or _rating(row['content'])!=row['rating']:
            raise ValueError('Saved judgment model/rating mismatch')
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)
    template = _load_template(args.official_evaluator)
    with raw_path.open("a", encoding="utf-8") as output:
        for ordinal, (source, prediction) in enumerate(joined):
            if source["id"] in completed:
                continue
            prompt = template.format(
                ", ".join(source["image_content"]), source["prompt"], source["gt_answer"], prediction["response"]
            )
            last_error = None
            for attempt in range(args.max_retries):
                try:
                    response = client.chat.completions.create(
                        model=args.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    content = response.choices[0].message.content or ""
                    with (args.output_dir/'all_attempts.jsonl').open('a') as audit:
                        audit.write(json.dumps({'id':source['id'],'attempt':attempt,'response':response.model_dump()})+'\n')
                        audit.flush();os.fsync(audit.fileno())
                    if response.model != args.judge_model or response.choices[0].finish_reason != 'stop':
                        raise ValueError('Unexpected judge model or incomplete output')
                    score = _rating(content)
                    record = {
                        "id": source["id"],
                        "ordinal": ordinal,
                        "question_type": source["question_type"],
                        "judge_requested": args.judge_model,
                        "judge_returned": response.model,
                        "rating": score,
                        "parser_version": PARSER_VERSION,
                        "content": content,
                        "response_id": response.id,
                        "usage": response.usage.model_dump() if response.usage else None,
                    }
                    output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    completed[source["id"]] = record
                    break
                except Exception as exc:  # API errors must remain visible after bounded retries.
                    last_error = exc
                    time.sleep(min(2**attempt, 30))
            else:
                raise RuntimeError(f"judge failed for id={source['id']}") from last_error
    ordered = [completed[source["id"]] for source, _ in joined]
    scores = [row["rating"] for row in ordered]
    per_type: dict[str, list[int]] = {}
    for row in ordered:
        per_type.setdefault(row["question_type"], []).append(row["rating"])
    summary = {
        "benchmark": "mmhal",
        "samples": 96,
        "judge_model_requested": args.judge_model,
        "judge_models_returned": sorted({row["judge_returned"] for row in ordered}),
        "average_score": statistics.fmean(scores),
        "hallucination_rate": sum(score < 3 for score in scores) / len(scores),
        "per_type_score": {key: statistics.fmean(values) for key, values in sorted(per_type.items())},
        "historical_parity": False,
        "note": "Historical gpt-4-0314 is retired; compare only outputs re-scored with this same judge.",
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
