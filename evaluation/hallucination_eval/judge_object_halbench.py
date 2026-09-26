from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from .common import join_predictions, read_jsonl


def _load_prompt(source: Path) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "prompt_template" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"prompt_template not found in {source}")


def _parse_objects(content: str) -> list[str]:
    candidates = [content.strip(), content.strip().splitlines()[-1] if content.strip() else ""]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
            return [item.strip() for item in value]
    raise ValueError("judge response is not a JSON list of object names")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paid Object HalBench object-extraction stage.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY must be set in the environment")
    joined = join_predictions(args.manifest, args.predictions)
    if len(joined) != 300:
        raise ValueError(f"Object HalBench requires 300 predictions, got {len(joined)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["id"]: row for row in read_jsonl(args.output)} if args.output.exists() else {}
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)
    template = _load_prompt(args.official_evaluator)
    with args.output.open("a", encoding="utf-8") as output:
        for source, prediction in joined:
            if source["id"] in completed:
                continue
            if len(prediction["response"].strip().split()) <= 3:
                record = {
                    "id": source["id"],
                    "image_id": source["image_id"],
                    "judge_requested": args.judge_model,
                    "judge_returned": "not_called_short_answer",
                    "objects": [],
                    "content": "",
                    "response_id": None,
                    "usage": None,
                }
                output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                completed[source["id"]] = record
                continue
            prompt = template.replace("{question}", source["prompt"]).replace("{answer}", prediction["response"])
            last_error = None
            for attempt in range(args.max_retries):
                try:
                    response = client.chat.completions.create(
                        model=args.judge_model,
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512,
                    )
                    content = response.choices[0].message.content or ""
                    with args.output.with_suffix('.attempts.jsonl').open('a') as audit:
                        audit.write(json.dumps({'id':source['id'],'attempt':attempt,'response':response.model_dump()})+'\n')
                        audit.flush();os.fsync(audit.fileno())
                    if response.model != args.judge_model or response.choices[0].finish_reason != 'stop':
                        raise ValueError('Unexpected judge model or incomplete output')
                    objects = _parse_objects(content)
                    record = {
                        "id": source["id"],
                        "image_id": source["image_id"],
                        "judge_requested": args.judge_model,
                        "judge_returned": response.model,
                        "objects": objects,
                        "content": content,
                        "response_id": response.id,
                        "usage": response.usage.model_dump() if response.usage else None,
                    }
                    output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    completed[source["id"]] = record
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(min(2**attempt, 30))
            else:
                raise RuntimeError(f"judge failed for id={source['id']}") from last_error
    if set(completed) != {source["id"] for source, _ in joined}:
        raise ValueError("incomplete extraction file")
    print(f"complete: {len(completed)} Object HalBench extractions -> {args.output}")


if __name__ == "__main__":
    main()
