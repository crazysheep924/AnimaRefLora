#!/usr/bin/env bash
# Host launcher: overnight f1_anti_copy resume training on the local RTX 5090.
# Mounts the local storage/caches/dataset and resumes from the from0-diffw 145k
# checkpoint. Close VRAM-hungry Windows apps (Photos/Chrome) before starting —
# the host side already steals ~6GB of the 32GB.
# Usage:
#   bash scripts/launch_f1anticopy_local5090.sh
# Override knobs by exporting env first, e.g. STEPS=200000 F1_ANTI_COPY_WEIGHT=0.075.
set -euo pipefail

REPO="/path/to/anima-reflora"
STORAGE_HOST="/path/to/storage"
IMAGES_HOST="/path/to/dataset/images"
# merged 138k-image latcache (old 55k + 83k expansion) on D:
LATCACHE_HOST="${LATCACHE_HOST:-/path/to/latcache}"
IMAGE_TAG="${IMAGE_TAG:-anima-reflora-runpod:latest}"

exec docker run --gpus all --rm \
  --ulimit nofile=1048576:1048576 \
  --name f1anticopy-5090 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ANIMA_REFLORA_STORAGE=/workspace/storage \
  -e ANIMA_REFLORA_MODEL_DIT=/workspace/storage/anima_models/diffusion_models/anima-base-v1.0.safetensors \
  -e ANIMA_REFLORA_MODEL_TE=/workspace/storage/anima_models/text_encoders \
  -e ANIMA_REFLORA_MODEL_VAE=/workspace/storage/anima_models/vae/qwen_image_vae.safetensors \
  -e ANIMA_REFLORA_LATCACHE="${ANIMA_REFLORA_LATCACHE:-/workspace/latcache}" \
  -v "${LATCACHE_HOST}:/workspace/latcache:ro" \
  -e OUT=/work/RunpodTraining \
  -e RUN_NAME="${RUN_NAME:-headroi-rope-cpm-f1anticopy-idinject}" \
  -e SOURCE_CKPT="${SOURCE_CKPT:-}" \
  -e RESUME_CKPT="${RESUME_CKPT:-/work/RunpodTraining/experiments/headroi-rope-cpm-f1anticopy-200k/checkpoints/lora_step_150000.safetensors}" \
  -e IDENTITY_INJECT_MAP="${IDENTITY_INJECT_MAP:-/workspace/storage/runs/identity_inject_map.json}" \
  -e IDENTITY_INJECT_PROB="${IDENTITY_INJECT_PROB:-0.8}" \
  -e STEPS="${STEPS:-200000}" \
  -e CKPT_EVERY="${CKPT_EVERY:-2500}" \
  -e CCIP_CACHE="${CCIP_CACHE:-}" \
  -e ANIMA_REFLORA_HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-}" \
  -e PAIR_DHASH_CACHE="${PAIR_DHASH_CACHE:-}" \
  -e PAIR_MIN_DHASH="${PAIR_MIN_DHASH:-25}" \
  -e LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}" \
  -e DIFF_LOSS_LAMBDA="${DIFF_LOSS_LAMBDA:-0.5}" \
  -e DIFF_WEIGHT_MIN="${DIFF_WEIGHT_MIN:-0.2}" \
  -e HEAD_LOSS_WEIGHT="${HEAD_LOSS_WEIGHT:-2.0}" \
  -e F1_ANTI_COPY_WEIGHT="${F1_ANTI_COPY_WEIGHT:-0.05}" \
  -e F1_ANTI_COPY_MARGIN="${F1_ANTI_COPY_MARGIN:-0.35}" \
  -e F1_ANTI_COPY_SIGMA_CUTOFF="${F1_ANTI_COPY_SIGMA_CUTOFF:-0.8}" \
  -e REF_DROPOUT_PROB="${REF_DROPOUT_PROB:-0.25}" \
  -e SINGLETON_REF_MODE="${SINGLETON_REF_MODE:-blank}" \
  -e CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}" \
  -e TAG_DROPOUT_PROB="${TAG_DROPOUT_PROB:-0.5}" \
  -e TAG_KEEP_PROB="${TAG_KEEP_PROB:-0.5}" \
  -e TAG_KEEP_MIN="${TAG_KEEP_MIN:-3}" \
  -e LR="${LR:-1e-5}" \
  -e WARMUP_STEPS="${WARMUP_STEPS:-0}" \
  -v "${REPO}/anima_reflora:/opt/AnimaRefLora/anima_reflora" \
  -v "${REPO}/scripts:/opt/AnimaRefLora/scripts" \
  -v "${STORAGE_HOST}:/workspace/storage" \
  -v "${IMAGES_HOST}:/workspace/storage/dataset/images:ro" \
  -v "${REPO}/RunpodTraining:/work/RunpodTraining" \
  "${IMAGE_TAG}" \
  bash /opt/AnimaRefLora/scripts/run_from0_f1anticopy_resume.sh
