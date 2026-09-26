"""Local task utilities for the pinned Qwen2-VL baseline protocol.

This module is deliberately self-contained. The stock MM-Vet utility initializes
an API-backed judge at import time in lmms-eval v0.7.2, so this task never imports
it. MM-Vet inference and judging are two separate stages in this protocol.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# MME: exact upstream aggregation for complete pairs, graceful limited smokes.


def mme_aggregate_results_complete_or_smoke(
    results: Iterable[dict[str, Any]],
) -> float:
    question_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        question_scores[str(result["category"])][str(result["question_id"])].append(
            float(result["score"])
        )

    total = 0.0
    for questions in question_scores.values():
        category_total = 0.0
        for scores in questions.values():
            if len(scores) == 2:
                # This is exactly lmms-eval v0.7.2's official MME formula:
                # accuracy (0/50/100) + accuracy-plus (0/100).
                category_total += sum(scores) / 2.0 * 100.0
                category_total += float(sum(scores) == 2.0) * 100.0
            elif len(scores) == 1:
                # Only reachable for --limit smoke/interrupted runs. The outer
                # run validator refuses to accept this for a full benchmark.
                category_total += scores[0] * 100.0
            else:
                raise RuntimeError(
                    f"MME question must have one smoke row or two full rows, got {len(scores)}"
                )
        total += category_total / len(questions)
    return total


# ---------------------------------------------------------------------------
# POPE: preserve lmms-eval's strict exact yes/no parser during primary scoring.


def pope_doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    return [doc["image"].convert("RGB")]


def pope_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    return (
        f"{kwargs.get('pre_prompt', '')}"
        f"{doc['question'].strip()}"
        f"{kwargs.get('post_prompt', '')}"
    )


def pope_process_results_strict(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    prediction = results[0].lower().strip()
    ground_truth = doc["answer"].lower().strip()
    if ground_truth not in {"yes", "no"}:
        raise ValueError(f"Unexpected POPE target: {ground_truth!r}")
    item = {
        "question_id": doc["question_id"],
        "category": doc.get("category", "unknown"),
        "score": float(prediction == ground_truth),
        "prediction": prediction,
        "ground_truth": ground_truth,
    }
    return {
        "pope_accuracy": dict(item),
        "pope_precision": dict(item),
        "pope_recall": dict(item),
        "pope_f1_score": dict(item),
    }


def pope_aggregate_accuracy(results: Iterable[dict[str, Any]]) -> float:
    items = list(results)
    return sum(float(item["score"]) for item in items) / len(items)


def pope_aggregate_precision(results: Iterable[dict[str, Any]]) -> float:
    items = list(results)
    true_positive = sum(
        item["prediction"] == "yes" and item["ground_truth"] == "yes"
        for item in items
    )
    false_positive = sum(
        item["prediction"] == "yes" and item["ground_truth"] == "no"
        for item in items
    )
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def pope_aggregate_recall(results: Iterable[dict[str, Any]]) -> float:
    items = list(results)
    true_positive = sum(
        item["prediction"] == "yes" and item["ground_truth"] == "yes"
        for item in items
    )
    false_negative = sum(
        item["prediction"] != "yes" and item["ground_truth"] == "yes"
        for item in items
    )
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def pope_aggregate_f1_score(results: Iterable[dict[str, Any]]) -> float:
    items = list(results)
    precision = pope_aggregate_precision(items)
    recall = pope_aggregate_recall(items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


# ---------------------------------------------------------------------------
# SEED-Bench: freeze an image-only derivative before request construction.


def seed_image_process_docs(dataset: Any) -> Any:
    source_count = len(dataset)
    if source_count != 17_990:
        raise RuntimeError(
            f"Pinned SEED-Bench source must contain 17,990 rows, got {source_count:,}"
        )
    # Restrict the filter input to one scalar column. Passing full rows causes
    # Hugging Face Datasets to decode the 26 GB image column unnecessarily.
    filtered = dataset.filter(
        lambda data_type: str(data_type).lower().strip() == "image",
        input_columns=["data_type"],
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Selecting the frozen SEED-img subset",
    )
    if len(filtered) != 14_233:
        raise RuntimeError(
            f"Pinned SEED-img subset must contain 14,233 rows, got {len(filtered):,}"
        )
    return filtered


def seed_doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    images = doc["image"]
    if not isinstance(images, list):
        images = [images]
    return [image.convert("RGB") for image in images if image is not None]


def seed_doc_to_text(doc: dict[str, Any]) -> str:
    question = doc["question"]
    choices = "\n".join(
        f"{letter}. {doc[f'choice_{letter.lower()}']}" for letter in "ABCD"
    )
    return (
        f"{question}\n{choices}\n"
        "Answer with the option's letter from the given choices directly."
    )


def seed_process_result(
    doc: dict[str, Any], results: list[str]
) -> dict[str, float]:
    prediction = results[0].strip()
    if len(prediction) > 1:
        prediction = prediction[0]
    return {
        "seed_image": float(
            prediction.lower().strip() == str(doc["answer"]).lower().strip()
        )
    }


# ---------------------------------------------------------------------------
# MM-Vet: inference-only.  No network client, API key, or judge is initialized.


def mmvet_doc_to_visual_safe(doc: dict[str, Any]) -> list[Any]:
    image = doc.get("image")
    return [] if image is None else [image.convert("RGB")]


def mmvet_doc_to_text_safe(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    return (
        f"{kwargs.get('pre_prompt', '')}"
        f"{doc['question']}"
        f"{kwargs.get('post_prompt', '')}"
    )


def mmvet_process_results_inference_only(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    return {
        "mmvet_prediction_count": {
            "question_id": str(doc["question_id"]),
            "question": doc["question"],
            "ground_truth": doc["answer"],
            "capability": doc.get("capability", ""),
            "prediction": results[0],
        }
    }


def mmvet_aggregate_predictions_inference_only(
    results: Iterable[dict[str, Any]], args: Any
) -> float:
    # Keep aggregation side-effect-free so limited smoke runs work. The
    # post-inference exporter is the only stage allowed to publish an artifact,
    # and it fail-closes unless all 218 unique predictions are present.
    del args
    return float(len(list(results)))
