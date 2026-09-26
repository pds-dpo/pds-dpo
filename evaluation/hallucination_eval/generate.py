from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
from pathlib import Path
from typing import Callable

from PIL import Image

from .common import atomic_write_json, file_record, load_protocol, read_jsonl, sha256


def _existing_ids(path: Path) -> set[object]:
    if not path.exists():
        return set()
    rows = list(read_jsonl(path))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in existing output: {path}")
    return set(ids)


def _checkpoint_records(value: str | None) -> list[dict]:
    if not value:
        return []
    path = Path(value).expanduser()
    if not path.exists():
        return [{"path_or_hub_id": value, "local": False}]
    if path.is_file():
        return [file_record(path.resolve())]
    names = (
        "config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "model.safetensors.index.json",
        "non_lora_trainables.bin",
        "run_manifest.json",
        "trainer_state.json",
    )
    records = [file_record((path / name).resolve()) for name in names if (path / name).is_file()]
    if not records:
        raise FileNotFoundError(f"checkpoint directory has none of the expected identity files: {path}")
    return records


def _qwen_generator(args: argparse.Namespace, protocol: dict) -> Callable[[Image.Image, str, int], str]:
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    cfg = protocol["qwen2_vl"]
    model_source = args.model_base or args.model
    processor = AutoProcessor.from_pretrained(
        model_source,
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
        use_fast=False,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": args.device},
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": protocol["generation"]["system_prompt"]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
        ]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[rendered], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    return generate


def _llava_generator(args: argparse.Namespace, protocol: dict) -> Callable[[Image.Image, str, int], str]:
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_path = args.adapter or args.model
    model_base = args.model_base if args.adapter else None
    model_name = args.llava_model_name or ("llava-v1.5-7b-lora" if args.adapter else Path(model_path).name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map=args.device,
        device=args.device,
    )
    model.eval()
    conv_mode = protocol["llava_1_5"]["conversation"]

    def generate(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        if model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conversation = conv_templates[conv_mode].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        rendered = conversation.get_prompt()
        input_ids = tokenizer_image_token(rendered, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        pixels = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().to(args.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=pixels,
                image_sizes=[image.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hallucination benchmark predictions; never calls a judge API.")
    parser.add_argument("--backend", choices=["qwen2_vl", "llava_1_5"], required=True)
    parser.add_argument("--benchmark", choices=["mmhal", "object_halbench", "amber"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Full checkpoint, or base checkpoint when --adapter is used")
    parser.add_argument("--model-base", help="Explicit base checkpoint for an adapter")
    parser.add_argument("--adapter", help="PEFT/LLaVA LoRA adapter path")
    parser.add_argument("--llava-model-name", help="Force a LLaVA builder model name; LoRA names must contain 'lora'")
    parser.add_argument("--model-id", required=True, help="Stable human-readable result label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard parameters")
    if args.adapter and not args.model_base:
        parser.error("--adapter requires --model-base")
    if os.getenv("OPENAI_API_KEY"):
        print("warning: OPENAI_API_KEY is present but generation never reads or transmits it", flush=True)

    # Isolate a single physical GPU before importing torch/model libraries. This
    # also avoids the legacy LLaVA builder passing a dict as a torch device.
    if args.device.startswith("cuda:"):
        physical_gpu = args.device.split(":", 1)[1]
        existing_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_visibility is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
        elif "," in existing_visibility:
            raise ValueError("generation requires exactly one visible GPU per worker")
        args.device = "cuda:0"

    protocol = load_protocol(args.protocol)
    benchmark_cfg = protocol["benchmarks"][args.benchmark]
    rows = [row for ordinal, row in enumerate(read_jsonl(args.manifest)) if ordinal % args.num_shards == args.shard_index]
    if any(row["benchmark"] != args.benchmark for row in rows):
        raise ValueError("manifest benchmark does not match --benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer holds {lock_path}") from exc
        completed = _existing_ids(args.output)
        pending = [row for row in rows if row["id"] not in completed]
        run_manifest = {
            "schema_version": 1,
            "evaluation_stage": "prediction",
            "benchmark": args.benchmark,
            "backend": args.backend,
            "model_id": args.model_id,
            "model": _checkpoint_records(args.model),
            "model_base": _checkpoint_records(args.model_base),
            "adapter": _checkpoint_records(args.adapter),
            "protocol_path": str(args.protocol.resolve()),
            "protocol_sha256": sha256(args.protocol),
            "input_manifest_path": str(args.manifest.resolve()),
            "input_manifest_sha256": sha256(args.manifest),
            "output_path": str(args.output.resolve()),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "generation": {**protocol["generation"], **benchmark_cfg},
        }
        sidecar = args.output.with_suffix(args.output.suffix + ".run.json")
        if sidecar.exists():
            if json.loads(sidecar.read_text(encoding="utf-8")) != run_manifest:
                raise ValueError(f"existing run sidecar does not match requested run: {sidecar}")
        else:
            atomic_write_json(sidecar, run_manifest)
        if not pending:
            print(f"complete: {len(rows)}/{len(rows)} rows already exist in {args.output}")
            return
        random.seed(protocol["generation"]["seed"])
        generator = _qwen_generator(args, protocol) if args.backend == "qwen2_vl" else _llava_generator(args, protocol)
        with args.output.open("a", encoding="utf-8") as output:
            for offset, row in enumerate(pending, 1):
                with Image.open(row["image_path"]) as opened:
                    image = opened.convert("RGB")
                response = generator(image, row["prompt"], int(benchmark_cfg["max_new_tokens"]))
                if not response:
                    raise ValueError(f"empty response for {row['id']}")
                record = {
                    "schema_version": 1,
                    "benchmark": args.benchmark,
                    "id": row["id"],
                    "response": response,
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "protocol": protocol["suite_name"],
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(f"{offset}/{len(pending)} id={row['id']}", flush=True)


if __name__ == "__main__":
    main()
