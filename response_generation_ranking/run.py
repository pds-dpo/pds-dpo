import os
import json
import re
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Paths and configuration
image_folder = "images-ranked/"
output_json_file = "output.json"
prompt_file = "instruction-prompts/sample.txt"
response_files = [
    'responses/mistral-7b.txt',
    'responses/vicuna-13b.txt',
    'responses/vicuna-7b.txt'
]
model_ids = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf"
]
ranking_model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Initialize the ranking model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
ranking_model = AutoModelForSequenceClassification.from_pretrained(
    ranking_model_path,
    device_map=device,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
ranking_tokenizer = AutoTokenizer.from_pretrained(ranking_model_path, use_fast=True)

# Read prompts
try:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}.")
except FileNotFoundError:
    print(f"Prompt file '{prompt_file}' not found.")
    exit(1)

# Sort image files numerically
image_files = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else float('inf'))

if len(prompts) < len(image_files):
    raise ValueError(f"Number of prompts ({len(prompts)}) is less than number of images ({len(image_files)}). Provide more prompts.")
elif len(prompts) > len(image_files):
    print(f"Warning: Number of prompts ({len(prompts)}) exceeds number of images ({len(image_files)}). Extra prompts will be ignored.")

# Generate responses with all models and save them in separate files
for model_id, response_file in zip(model_ids, response_files):
    print(f"Generating responses using model: {model_id}")
    
    # Ensure the directory exists for the response file
    response_dir = os.path.dirname(response_file)
    os.makedirs(response_dir, exist_ok=True)

    # Initialize the image-to-text pipeline
    pipe = pipeline(
        "image-to-text",
        model=model_id,
        model_kwargs={"quantization_config": quantization_config},
        device_map="auto"
    )

    responses = []
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image '{image_file}': {e}")
            responses.append("")
            continue

        prompt = prompts[idx]

        try:
            outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 1000})
            generated_text = outputs[0]["generated_text"] if outputs and "generated_text" in outputs[0] else ""
        except Exception as e:
            print(f"Failed to generate text for image '{image_file}': {e}")
            generated_text = ""

        # Ensure the generated response does not include the prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        responses.append(generated_text.strip())

    # Save the responses
    try:
        with open(response_file, "w", encoding="utf-8") as f:
            for response in responses:
                f.write(response.replace("\n", " ") + "\n")
        print(f"Responses saved to '{response_file}'.")
    except Exception as e:
        print(f"Failed to save responses to '{response_file}': {e}")

# Read responses from all models and rank them
output_data = []
responses_from_files = []
for response_file in response_files:
    try:
        with open(response_file, "r", encoding="utf-8") as f:
            responses_from_files.append([line.strip() for line in f])
    except Exception as e:
        print(f"Failed to read responses from '{response_file}': {e}")
        responses_from_files.append([])

def clean_response(prompt, response):
    # Remove the exact prompt and variations (e.g., with extra spaces)
    prompt_variants = [
        re.escape(prompt),  # Exact match
        re.escape(prompt.strip()),  # Stripped version
        re.escape(prompt.strip().replace("<image>", "").strip()),  # Without <image>
    ]
    for variant in prompt_variants:
        response = re.sub(variant + r"\s*", "", response, flags=re.IGNORECASE).strip()
    return response

# Rank responses 
for idx, prompt in enumerate(prompts):
    image_id = f"art-{idx + 1}"
    image_path = f"images/{image_id}.jpg"

    # Collect responses for this prompt
    responses = [responses[idx] for responses in responses_from_files if len(responses) > idx]
    scores = []

    for response in responses:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_ids = ranking_tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        with torch.no_grad():
            output = ranking_model(input_ids)
            scores.append(output.score.cpu().float().item())

    max_score_idx = scores.index(max(scores))
    min_score_idx = scores.index(min(scores))

    chosen_response = clean_response(prompt, responses[max_score_idx])
    rejected_response = clean_response(prompt, responses[min_score_idx])

    # Prepare data in the required format
    data = {
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
    output_data.append(data)

# Write output to a JSON file
try:
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Final output saved to '{output_json_file}'.")
except Exception as e:
    print(f"Failed to write output to '{output_json_file}': {e}")

