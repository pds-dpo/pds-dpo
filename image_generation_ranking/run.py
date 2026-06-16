import argparse
import os

import ImageReward as RM
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import get_token, login
from PIL import Image


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_GUIDANCE_SCALES = [5.0, 7.0, 9.0, 11.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate image candidates and rank them with ImageReward.")
    parser.add_argument("--model-id", default=os.getenv("PDS_DPO_IMAGE_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--ranker-id", default=os.getenv("PDS_DPO_IMAGE_RANKER_ID", "ImageReward-v1.0"))
    parser.add_argument("--prompt-file", default=os.getenv("PDS_DPO_PROMPT_FILE", "prompts/sample.txt"))
    parser.add_argument("--img-dir", default=os.getenv("PDS_DPO_IMG_DIR", "images/sample"))
    parser.add_argument("--output-dir", default=os.getenv("PDS_DPO_OUTPUT_DIR", "images/sample-ranked"))
    parser.add_argument("--device", default=os.getenv("PDS_DPO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-images", type=int, default=int(os.getenv("PDS_DPO_NUM_IMAGES", "4")))
    parser.add_argument("--num-inference-steps", type=int, default=int(os.getenv("PDS_DPO_NUM_INFERENCE_STEPS", "28")))
    parser.add_argument("--seed-base", type=int, default=int(os.getenv("PDS_DPO_SEED_BASE", "1000")))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.getenv("PDS_DPO_RESUME", "").lower() in {"1", "true", "yes"},
        help="Skip prompts whose ranked output image already exists.",
    )
    parser.add_argument(
        "--guidance-scales",
        default=os.getenv("PDS_DPO_GUIDANCE_SCALES", ",".join(str(scale) for scale in DEFAULT_GUIDANCE_SCALES)),
        help="Comma-separated guidance scales. Values are cycled if --num-images is larger than the list.",
    )
    return parser.parse_args()


def get_hf_token_arg():
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    if token:
        login(token=token, add_to_git_credential=False)
        return token
    return True if get_token() else None


def load_prompts(prompt_file, max_prompts):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}.")
    return prompts


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    guidance_scales = [float(scale.strip()) for scale in args.guidance_scales.split(",") if scale.strip()]
    if not guidance_scales:
        raise ValueError("At least one guidance scale is required.")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    token_arg = get_hf_token_arg()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        token=token_arg,
    )
    pipe = pipe.to(args.device)

    model = RM.load(args.ranker_id, device=args.device)

    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args.prompt_file, args.max_prompts)

    for prompt_idx, prompt in enumerate(prompts, start=1):
        output_path = os.path.join(args.output_dir, f"art-{prompt_idx}.jpg")
        if args.resume and os.path.exists(output_path):
            print(f"Skipping prompt {prompt_idx}/{len(prompts)} because {output_path} already exists.", flush=True)
            continue

        print(f"Processing prompt {prompt_idx}/{len(prompts)}: {prompt}", flush=True)

        images = []
        for img_num in range(1, args.num_images + 1):
            seed = prompt_idx * args.seed_base + img_num
            generator = torch.Generator(device=args.device).manual_seed(seed)
            guidance_scale = guidance_scales[(img_num - 1) % len(guidance_scales)]

            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            img_path = os.path.join(args.img_dir, f"{prompt_idx}-{img_num}.png")
            image.save(img_path)
            images.append(img_path)
            print(f"Saved: {img_path}", flush=True)

        with torch.no_grad():
            scores = []
            for img_path in images:
                score = model.score(prompt, img_path)
                scores.append(score)
                print(f"{os.path.basename(img_path)}: {score:.2f}", flush=True)

            max_index = scores.index(max(scores))
            best_image_path = images[max_index]
            best_score = scores[max_index]

            with Image.open(best_image_path) as img:
                img.convert("RGB").save(output_path, "JPEG")
            print(f"Best image for prompt {prompt_idx} saved: {output_path} (Score: {best_score:.2f})\n", flush=True)


if __name__ == "__main__":
    main()
