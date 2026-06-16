import argparse
import gc
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, pipeline


DEFAULT_RESPONSE_FILES = [
    "responses/mistral-7b.txt",
    "responses/vicuna-13b.txt",
    "responses/vicuna-7b.txt",
]
DEFAULT_MODEL_IDS = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool_env(name):
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and rank VLM responses for ranked images.")
    parser.add_argument("--image-folder", default=os.getenv("PDS_DPO_STEP2_IMAGE_FOLDER", "images-ranked"))
    parser.add_argument("--output-json-file", default=os.getenv("PDS_DPO_OUTPUT_JSON", "output.json"))
    parser.add_argument("--prompt-file", default=os.getenv("PDS_DPO_STEP2_PROMPT_FILE", "instruction-prompts/sample.txt"))
    parser.add_argument(
        "--response-files",
        default=os.getenv("PDS_DPO_RESPONSE_FILES", ",".join(DEFAULT_RESPONSE_FILES)),
        help="Comma-separated output files, one per model.",
    )
    parser.add_argument(
        "--model-ids",
        default=os.getenv("PDS_DPO_MODEL_IDS", ",".join(DEFAULT_MODEL_IDS)),
        help="Comma-separated image-to-text model IDs.",
    )
    parser.add_argument("--ranking-model-path", default=os.getenv("PDS_DPO_RANKING_MODEL", "RLHFlow/ArmoRM-Llama3-8B-v0.1"))
    parser.add_argument("--device-map", default=os.getenv("PDS_DPO_DEVICE_MAP", "auto"))
    parser.add_argument(
        "--ranker-device-map",
        default=os.getenv("PDS_DPO_RANKER_DEVICE_MAP", "single"),
        help="Use 'single' to keep the reward model on one GPU, or pass a Transformers device_map value.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("PDS_DPO_MAX_NEW_TOKENS", "1000")))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--json-image-prefix", default=os.getenv("PDS_DPO_JSON_IMAGE_PREFIX", "images"))
    parser.add_argument(
        "--reuse-responses",
        action="store_true",
        default=parse_bool_env("PDS_DPO_REUSE_RESPONSES"),
        help="Skip response generation for response files that already exist.",
    )
    return parser.parse_args()


def image_sort_key(path):
    match = re.search(r"(\d+)$", path.stem)
    number = int(match.group(1)) if match else float("inf")
    return number, path.stem


def load_prompts(prompt_file, max_items):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if max_items is not None:
        prompts = prompts[:max_items]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}.")
    return prompts


def load_image_files(image_folder, max_items):
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder '{image_folder}' not found.")
    image_files = sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=image_sort_key,
    )
    if max_items is not None:
        image_files = image_files[:max_items]
    if not image_files:
        raise ValueError(f"No image files found in {image_folder}.")
    return image_files


def clean_response(prompt, response):
    prompt_variants = [
        re.escape(prompt),
        re.escape(prompt.strip()),
        re.escape(prompt.strip().replace("<image>", "").strip()),
    ]
    for variant in prompt_variants:
        response = re.sub(variant + r"\s*", "", response, flags=re.IGNORECASE).strip()
    return response


def resolve_ranker_device_map(value):
    if not torch.cuda.is_available():
        return None
    if value.lower() in {"none", "false", "null"}:
        return None
    if value.lower() == "single":
        return {"": 0}
    return value


def generate_responses(args, prompts, image_files, model_ids, response_files):
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    for model_id, response_file in zip(model_ids, response_files):
        response_path = Path(response_file)
        if args.reuse_responses and response_path.exists():
            print(f"Reusing existing responses from '{response_file}'.", flush=True)
            continue

        print(f"Generating responses using model: {model_id}", flush=True)
        response_path.parent.mkdir(parents=True, exist_ok=True)

        model_kwargs = {"quantization_config": quantization_config} if quantization_config else {}
        pipe = pipeline(
            "image-to-text",
            model=model_id,
            model_kwargs=model_kwargs,
            device_map=args.device_map if torch.cuda.is_available() else None,
        )

        responses = []
        for idx, image_path in enumerate(image_files):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                print(f"Failed to open image '{image_path.name}': {exc}", flush=True)
                responses.append("")
                continue

            prompt = prompts[idx]
            try:
                outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": args.max_new_tokens})
                generated_text = outputs[0].get("generated_text", "") if outputs else ""
            except Exception as exc:
                print(f"Failed to generate text for image '{image_path.name}': {exc}", flush=True)
                generated_text = ""

            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            responses.append(generated_text.strip())
            print(f"Generated response {idx + 1}/{len(image_files)} for {image_path.name}", flush=True)

        with open(response_path, "w", encoding="utf-8") as f:
            for response in responses:
                f.write(response.replace("\n", " ") + "\n")
        print(f"Responses saved to '{response_file}'.", flush=True)

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def read_responses(response_files):
    responses_from_files = []
    for response_file in response_files:
        with open(response_file, "r", encoding="utf-8") as f:
            responses_from_files.append([line.strip() for line in f])
    return responses_from_files


def rank_responses(args, prompts, image_files, response_files):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ranking_model = AutoModelForSequenceClassification.from_pretrained(
        args.ranking_model_path,
        device_map=resolve_ranker_device_map(args.ranker_device_map),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    ranking_tokenizer = AutoTokenizer.from_pretrained(args.ranking_model_path, use_fast=True)
    rank_device = next(ranking_model.parameters()).device if torch.cuda.is_available() else torch.device(device)
    responses_from_files = read_responses(response_files)

    output_data = []
    for idx, (prompt, source_image) in enumerate(zip(prompts, image_files)):
        image_id = source_image.stem
        image_path = f"{args.json_image_prefix.rstrip('/')}/{source_image.name}"
        responses = [responses[idx] for responses in responses_from_files if len(responses) > idx]
        if not responses:
            raise ValueError(f"No responses found for {source_image.name}.")

        scores = []
        for response in responses:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            input_ids = ranking_tokenizer.apply_chat_template(messages, return_tensors="pt").to(rank_device)
            with torch.no_grad():
                output = ranking_model(input_ids)
                scores.append(output.score.cpu().float().item())

        max_score_idx = scores.index(max(scores))
        min_score_idx = scores.index(min(scores))
        chosen_response = clean_response(prompt, responses[max_score_idx])
        rejected_response = clean_response(prompt, responses[min_score_idx])

        output_data.append(
            {
                "id": image_id,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": prompt.replace("<image> <image>", "<image>").strip()},
                    {"from": "gpt", "value": chosen_response},
                ],
                "rejected_conversations": [
                    {"from": "human", "value": prompt.replace("<image> <image>", "<image>").strip()},
                    {"from": "gpt", "value": rejected_response},
                ],
            }
        )
        print(f"Ranked responses for {image_id}: scores={scores}", flush=True)

    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Final output saved to '{args.output_json_file}'.", flush=True)


def main():
    args = parse_args()
    model_ids = split_csv(args.model_ids)
    response_files = split_csv(args.response_files)
    if len(model_ids) != len(response_files):
        raise ValueError(f"Expected one response file per model, got {len(model_ids)} models and {len(response_files)} files.")

    prompts = load_prompts(args.prompt_file, args.max_items)
    image_files = load_image_files(args.image_folder, args.max_items)

    if len(prompts) < len(image_files):
        raise ValueError(f"Number of prompts ({len(prompts)}) is less than number of images ({len(image_files)}).")
    if len(prompts) > len(image_files):
        print(f"Warning: Number of prompts ({len(prompts)}) exceeds number of images ({len(image_files)}). Extra prompts will be ignored.", flush=True)
        prompts = prompts[:len(image_files)]

    print(f"Using {len(image_files)} images from '{args.image_folder}'.", flush=True)
    generate_responses(args, prompts, image_files, model_ids, response_files)
    rank_responses(args, prompts, image_files, response_files)


if __name__ == "__main__":
    main()

