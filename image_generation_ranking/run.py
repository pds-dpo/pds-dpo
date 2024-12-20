import os
import torch
from diffusers import StableDiffusion3Pipeline
import ImageReward as RM
from PIL import Image
from shutil import copyfile

# Hugging Face login
from huggingface_hub import login
login(token="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")

def main():
    # Load Stable Diffusion pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Load ImageReward model
    model = RM.load("ImageReward-v1.0")

    # Input prompt file and output directories
    prompt_file = "prompts/sample.txt"
    img_dir = "images/sample"
    output_dir = "images/sample-ranked"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Read prompts
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Define guidance scales
    guidance_scales = [5.0, 7.0, 9.0, 11.0]

    # Generate and rank images
    for prompt_idx, prompt in enumerate(prompts, start=1):
        print(f"Processing prompt {prompt_idx}/{len(prompts)}: {prompt}")

        # Generate images
        images = []
        for img_num in range(1, 5):
            seed = prompt_idx * 1000 + img_num
            generator = torch.manual_seed(seed)

            guidance_scale = guidance_scales[img_num - 1]
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            # Save image
            img_path = os.path.join(img_dir, f"{prompt_idx}-{img_num}.png")
            image.save(img_path)
            images.append(img_path)
            print(f"Saved: {img_path}")

        # Rank images using ImageReward
        with torch.no_grad():
            scores = []
            for img_path in images:
                score = model.score(prompt, img_path)
                scores.append(score)
                print(f"{os.path.basename(img_path)}: {score:.2f}")

            # Find the best image
            max_index = scores.index(max(scores))
            best_image_path = images[max_index]
            best_score = scores[max_index]

            # Save the best image
            output_path = os.path.join(output_dir, f"art-{prompt_idx}.jpg")
            with Image.open(best_image_path) as img:
                img.convert("RGB").save(output_path, "JPEG")
            print(f"Best image for prompt {prompt_idx} saved: {output_path} (Score: {best_score:.2f})\n")

if __name__ == "__main__":
    main()
