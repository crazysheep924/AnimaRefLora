#!/usr/bin/env bash
# Self-service correct-condition generator (local 5090, validated flags).
# Usage:
#   bash scripts/gen_correct.sh <STEP> [PROMPT] [REF_ROOT_HOST] [OUT_NAME]
# Examples:
#   bash scripts/gen_correct.sh 135000
#   bash scripts/gen_correct.sh 135000 "1girl, <your tags here>"
#   bash scripts/gen_correct.sh 135000 "1girl, ..." /path/to/storage/8AFC_refs my_test
# Notes:
#   - PROMPT empty -> the built-in default eval prompt.
#   - REF_ROOT_HOST must contain <name>/pick/0sample.<ext> per character.
#   - LIMIT: how many refs to use (default: all found, capped 22).
#   - Output: RunpodTraining/<RUN>/generated/<OUT_NAME>/correct_XX_<name>.png
set -euo pipefail

STEP="${1:?usage: gen_correct.sh <STEP> [PROMPT] [REF_ROOT_HOST] [OUT_NAME]}"
PROMPT="${2:-}"
REF_ROOT_HOST="${3:-/path/to/eval_refs}"
OUT_NAME="${4:-custom_step${STEP}_correct}"
LIMIT="${LIMIT:-$(find "${REF_ROOT_HOST}" -mindepth 3 -maxdepth 3 -path '*/pick/0sample.*' ! -name '*.txt' 2>/dev/null | wc -l)}"
LIMIT=$(( LIMIT > 22 ? 22 : LIMIT ))
SEED="${SEED:-0}"

REPO="/path/to/anima-reflora"
RUN_DIR="${RUN_DIR:-headroi-rope-cpm-from0-diffw-20260703-060452}"

PROMPT_ARGS=()
if [ -n "${PROMPT}" ]; then
  PROMPT_ARGS=(--prompt "${PROMPT}")
fi

exec docker run --gpus all --rm \
  --ulimit nofile=1048576:1048576 \
  -e ANIMA_REFLORA_STORAGE=/workspace/storage \
  -e ANIMA_REFLORA_MODEL_DIT=/workspace/storage/anima_models/diffusion_models/anima-base-v1.0.safetensors \
  -e ANIMA_REFLORA_MODEL_TE=/workspace/storage/anima_models/text_encoders \
  -e ANIMA_REFLORA_MODEL_VAE=/workspace/storage/anima_models/vae/qwen_image_vae.safetensors \
  -v "${REPO}/anima_reflora:/opt/AnimaRefLora/anima_reflora" \
  -v /path/to/storage:/workspace/storage \
  -v "${REF_ROOT_HOST}:/refs:ro" \
  -v "${REPO}/RunpodTraining:/work/RunpodTraining" \
  anima-reflora-runpod:latest \
  python -m anima_reflora.local_ref_ab_infer \
  --checkpoint "/work/RunpodTraining/${RUN_DIR}/checkpoints/lora_step_${STEP}.safetensors" \
  --ref-root /refs \
  --wrong-ref /work/RunpodTraining/wrong.jpg \
  --output-dir "/work/RunpodTraining/${RUN_DIR}/generated/${OUT_NAME}" \
  --storage /workspace/storage \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --steps 24 \
  --guidance-scale 4.5 \
  --bucket-short 1024 \
  --bucket-long-max 1024 \
  --device cuda \
  --decode-device cuda \
  --vae-chunk-size 8 \
  --conditions correct \
  "${PROMPT_ARGS[@]}"
