from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


EXPECTED_COUNTS = {"mmhal": 96, "object_halbench": 300, "amber": 1004}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_no}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL line {line_no} is not an object: {path}")
            yield value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported protocol schema in {path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def join_predictions(manifest: Path, prediction_paths: list[Path]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    source = list(read_jsonl(manifest))
    predictions: dict[Any, dict[str, Any]] = {}
    for path in prediction_paths:
        for row in read_jsonl(path):
            if row["id"] in predictions:
                raise ValueError(f"duplicate prediction id {row['id']}")
            predictions[row["id"]] = row
    source_ids = {row["id"] for row in source}
    if len(source_ids) != len(source):
        raise ValueError("duplicate ids in manifest")
    if set(predictions) != source_ids:
        raise ValueError(
            f"prediction coverage mismatch: missing={list(source_ids - set(predictions))[:10]} "
            f"extra={list(set(predictions) - source_ids)[:10]}"
        )
    model_ids = {row.get("model_id") for row in predictions.values()}
    if len(model_ids) != 1 or None in model_ids:
        raise ValueError(f"mixed/missing model ids: {model_ids}")
    return [(row, predictions[row["id"]]) for row in source]
