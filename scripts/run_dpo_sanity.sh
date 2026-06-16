#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

deepspeed --num_gpus "${PDS_DPO_NUM_GPUS:-2}" llava/train/train_dpo.py \
    --lora_enable True --lora_r "${PDS_DPO_LORA_R:-8}" --lora_alpha "${PDS_DPO_LORA_ALPHA:-16}" --mm_projector_lr 2e-5 \
    --deepspeed "${PDS_DPO_DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
    --model_name_or_path "${PDS_DPO_MODEL_NAME_OR_PATH:-liuhaotian/llava-v1.5-7b}" \
    --version v1 \
    --data_path "${PDS_DPO_DATA_PATH:-./data/step3_sanity/step3_sanity.json}" \
    --image_folder "${PDS_DPO_IMAGE_FOLDER:-./data/step3_sanity}" \
    --vision_tower "${PDS_DPO_VISION_TOWER:-openai/clip-vit-large-patch14-336}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${PDS_DPO_OUTPUT_DIR:-./checkpoints/step3-sanity}" \
    --num_train_epochs 1 \
    --max_steps "${PDS_DPO_MAX_STEPS:--1}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${PDS_DPO_MODEL_MAX_LENGTH:-1024}" \
    --bits "${PDS_DPO_BITS:-4}" \
    --gradient_checkpointing "${PDS_DPO_GRADIENT_CHECKPOINTING:-False}" \
    --dataloader_num_workers 0 \
    --lazy_preprocess True
