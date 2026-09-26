"""Offline MMMU validation helpers pinned to lmms-eval v0.7.2.

This module contains only the deterministic prompting, parsing, and local
aggregation path needed by ``mmmu_val_qwen2vl_refresh12k``.  In particular, it
does not import or initialize lmms-eval's optional LLM-judge server.

The behavior is derived from ``lmms_eval/tasks/mmmu/utils.py`` at revision
``cb45ac4d4a667ea5ef89c7a148bff69b3489b981`` and the official MMMU evaluator
referenced by that file.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict

from lmms_eval.tasks._task_utils.mmmu_mcq_utils import (
    get_multi_choice_info as shared_get_multi_choice_info,
)
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import (
    parse_mmmu_multi_choice_response,
)

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(
        f"{letter}. {option}" for letter, option in zip(option_letters, options)
    )


def construct_prompt(doc, mc_prompt="", open_ended_prompt="", prompt_type="reasoning"):
    del prompt_type  # Retained for parity with the pinned lmms-eval signature.
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        return f"{question}\n{parsed_options}\n\n{mc_prompt}"
    return f"{question}\n\n{open_ended_prompt}"


def mmmu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        return construct_prompt(doc)
    return construct_prompt(
        doc,
        lmms_eval_specific_kwargs["multiple_choice_prompt"],
        lmms_eval_specific_kwargs["open_ended_prompt"],
        lmms_eval_specific_kwargs.get("prompt_type", "format"),
    )


def mmmu_doc_to_visual(doc):
    image_tokens = re.findall(r"<image \d+>", construct_prompt(doc))
    image_fields = sorted(
        {token.strip("<>").replace(" ", "_") for token in image_tokens}
    )
    return [doc[field].convert("RGB") for field in image_fields]


def get_multi_choice_info(options):
    return shared_get_multi_choice_info(options)


def parse_multi_choice_response(response, all_choices, index2ans):
    """Parse an MMMU multiple-choice response using the pinned v0.7.2 rules."""
    return parse_mmmu_multi_choice_response(response, all_choices, index2ans)


def extract_numbers(string):
    numbers_with_commas = re.findall(r"-?\b\d{1,3}(?:,\d{3})+\b", string)
    numbers_scientific = re.findall(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", string)
    numbers_simple = re.findall(
        r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])", string
    )
    return numbers_with_commas + numbers_scientific + numbers_simple


def check_is_number(string):
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    string = string.strip()
    if check_is_number(string):
        return [round(float(string.replace(",", "")), 2)]
    string = string.lower()
    if len(string) == 1:
        return [f" {string}", f"{string} "]
    return [string]


def parse_open_response(response):
    def get_key_subresponses(value):
        value = value.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", value)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, sub_response in enumerate(sub_responses):
            current_indicators = indicators + (["="] if index == len(sub_responses) - 1 else [])
            shortest = None
            for indicator in current_indicators:
                if indicator in sub_response:
                    candidate = sub_response.split(indicator)[-1].strip()
                    if shortest is None or len(candidate) < len(shortest):
                        shortest = candidate
            if shortest and shortest.strip() not in {":", ",", ".", "!", "?", ";", "'"}:
                key_responses.append(shortest)
        return key_responses or [value]

    key_responses = get_key_subresponses(response)
    predictions = list(key_responses)
    for item in key_responses:
        predictions.extend(extract_numbers(item))

    normalized = []
    for item in predictions:
        normalized.extend(normalize_str(item))
    # Preserve the exact pinned evaluator behavior.  PYTHONHASHSEED is fixed
    # by the runner, so this legacy set-based de-duplication is reproducible.
    return list(set(normalized))


def mmmu_process_results(doc, results):
    parsed_predictions = []
    for prediction in results:
        if doc["question_type"] == "multiple-choice":
            index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["options"]))
            parsed = parse_multi_choice_response(prediction, all_choices, index2ans)
        else:
            parsed_open = parse_open_response(prediction)
            parsed = str(parsed_open[0]) if parsed_open else ""
        parsed_predictions.append(parsed)

    exact_record = {
        "id": doc["id"],
        "subdomain": extract_subset_name(doc["id"]),
        "question_type": doc["question_type"],
        "answer": doc["answer"],
        "parsed_pred": parsed_predictions,
    }
    return {
        "mmmu_acc": exact_record,
        "mmmu_acc_pass_at_k": exact_record,
        "submission": {doc["id"]: parsed_predictions[0]},
    }


def extract_subset_name(identifier):
    split = identifier.split("_")[0]
    match = re.search(rf"^{re.escape(split)}_(.+?)_\d+$", identifier)
    if not match:
        raise ValueError(f'No MMMU subset found in "{identifier}"')
    return match.group(1)


def eval_multi_choice(gold, prediction):
    if isinstance(gold, list):
        return prediction in gold
    return gold == prediction


def eval_open(gold, predictions):
    normalized_answers = []
    if isinstance(gold, list):
        for answer in gold:
            normalized_answers.extend(normalize_str(answer))
    else:
        normalized_answers = normalize_str(gold)

    for prediction in predictions:
        if isinstance(prediction, str):
            if any(
                isinstance(answer, str) and answer in prediction
                for answer in normalized_answers
            ):
                return True
        elif prediction in normalized_answers:
            return True
    return False


def evaluate_mmmu(samples):
    correct_count = 0
    judgments = {}
    for sample in samples:
        correct = False
        for prediction in sample["parsed_pred"]:
            if sample["question_type"] == "multiple-choice":
                correct = eval_multi_choice(sample["answer"], prediction)
            else:
                correct = eval_open(sample["answer"], prediction)
            if correct:
                correct_count += 1
                break
        judgments[sample["id"]] = "Correct" if correct else "Wrong"
    accuracy = correct_count / len(samples) if samples else 0.0
    return judgments, {"acc": accuracy}


def calculate_ins_level_acc(results):
    correct_fraction_sum = 0.0
    sample_count = 0
    for category in results.values():
        correct_fraction_sum += category["acc"] * category["num_example"]
        sample_count += category["num_example"]
    return correct_fraction_sum / sample_count if sample_count else 0.0


def mmmu_aggregate_results(results):
    subset_samples = defaultdict(list)
    for result in results:
        subset_samples[result["subdomain"]].append(result)

    subset_results = {}
    for subset, samples in subset_samples.items():
        _, metric = evaluate_mmmu(samples)
        metric["num_example"] = len(samples)
        subset_results[subset] = metric

    # Construct the same domain summary as the pinned harness, while returning
    # only its overall instance-level accuracy to lmms-eval.
    printable_results = {}
    for domain, categories in DOMAIN_CAT2SUB_CAT.items():
        present = {name: subset_results[name] for name in categories if name in subset_results}
        printable_results[f"Overall-{domain}"] = {
            "num": sum(item["num_example"] for item in present.values()),
            "acc": round(calculate_ins_level_acc(present), 5),
        }
        for name, item in present.items():
            printable_results[name] = {
                "num": int(item["num_example"]),
                "acc": round(item["acc"], 5),
            }

    printable_results["Overall"] = {
        "num": sum(item["num_example"] for item in subset_results.values()),
        "acc": round(calculate_ins_level_acc(subset_results), 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]
