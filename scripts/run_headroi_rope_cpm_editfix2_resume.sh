#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  RESUME/CONTINUE from an existing checkpoint with tag-level caption dropout.
#  Forks an existing run (e.g. editfix step 40000, which already follows prompts
#  well) and continues to --steps with:
#    * --caption-dropout-prob 0          (drop the "empty caption -> copy" reflex)
#    * --tag-dropout-prob 0.5 / keep 0.5 / min 3   (short-but-non-empty captions)
#    * --ref-dropout-prob 0.25           (unchanged)
#  This applies the short-prompt fix as a fine-tune instead of retraining from 0.
#
#  Set RESUME_CKPT to the lora_step_N.safetensors to fork from. Its sidecars
#  (optimizer_step_N.pt, ref_conditioner_step_N, crepa_projector_step_N,
#  feature_config_step_N.json) must sit beside it. start-step auto-parses N.
#
#  Usage (override RESUME_CKPT for your pod layout):
#    RESUME_CKPT=/workspace/Persistent/runs/experiments/headroi-rope-cpm-editfix-100k-20260630-040642/checkpoints/lora_step_40000.safetensors \
#      bash /opt/AnimaRefLora/scripts/run_headroi_rope_cpm_editfix2_resume.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace}"
OUT="${ANIMA_REFLORA_OUT:-/workspace/Persistent/runs}"
IMAGE_ROOT="${ANIMA_REFLORA_IMAGES:-${STORAGE}/dataset/images}"
IMAGE_PREFIX="${ANIMA_REFLORA_IMAGE_PREFIX:-/path/to/dataset/images/}"
CCIP_CACHE="${CCIP_CACHE:-${STORAGE}/runs/ccip_ref_head_emb_cache.pt}"
HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-${STORAGE}/runs/head_roi_cache.pt}"
LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}"

REF_DROPOUT_PROB="${REF_DROPOUT_PROB:-0.25}"
CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}"
TAG_DROPOUT_PROB="${TAG_DROPOUT_PROB:-0.5}"
TAG_KEEP_PROB="${TAG_KEEP_PROB:-0.5}"
TAG_KEEP_MIN="${TAG_KEEP_MIN:-3}"
STEPS="${STEPS:-100000}"

# Checkpoint to fork from (override for your pod). Default = the editfix 40k run.
RESUME_CKPT="${RESUME_CKPT:-${OUT}/experiments/headroi-rope-cpm-editfix-100k-20260630-040642/checkpoints/lora_step_40000.safetensors}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="headroi-rope-cpm-editfix2-resume-${RUN_TAG}"

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: RESUME_CKPT not found: ${RESUME_CKPT}" >&2
  echo "Set RESUME_CKPT=<...>/checkpoints/lora_step_N.safetensors (sidecars must sit beside it)." >&2
  exit 2
fi

echo "=== Resuming ${RUN_NAME} from ${RESUME_CKPT} ==="
echo "    (ref-dropout=${REF_DROPOUT_PROB}, caption-dropout=${CAPTION_DROPOUT_PROB}, tag-dropout=${TAG_DROPOUT_PROB}/keep=${TAG_KEEP_PROB}/min=${TAG_KEEP_MIN}) -> ${STEPS} steps"
echo ""

exec python -m anima_reflora.train \
  --stage from0-headroi-rope-cpm \
  --run-name "${RUN_NAME}" \
  --allow-existing-run \
  --resume "${RESUME_CKPT}" \
  --steps "${STEPS}" \
  --batch 1 \
  --lr 1e-5 \
  --optimizer came \
  --warmup-steps 500 \
  --frames 3 \
  \
  --rope-refpos \
  --rope-layout disjoint \
  --cpm \
  --crepa \
  --crepa-pool head_roi \
  --crepa-lambda 0.1 \
  --crepa-block 8 \
  --latent-recon-loss-weight "${LATENT_RECON_LOSS_WEIGHT}" \
  --head-loss-weight 4.0 \
  --head-sigma-cutoff 0.6 \
  \
  --no-build-missing-head-cache \
  --high-sigma-mix-prob 0.20 \
  --ref-dropout-prob "${REF_DROPOUT_PROB}" \
  --ref-dropout-mode blank \
  --ref-dropout-t3-mode structured \
  --caption-dropout-prob "${CAPTION_DROPOUT_PROB}" \
  --tag-dropout-prob "${TAG_DROPOUT_PROB}" \
  --tag-keep-prob "${TAG_KEEP_PROB}" \
  --tag-keep-min "${TAG_KEEP_MIN}" \
  \
  --weighting-scheme none \
  --prompt-mode change_only \
  \
  --ckpt-every 5000 \
  --ref-eval-every 10000 \
  --log-every 10 \
  \
  --ccip-cache "${CCIP_CACHE}" \
  --head-roi-cache "${HEAD_ROI_CACHE}" \
  --storage "${STORAGE}" \
  --out-dir "${OUT}" \
  --image-root "${IMAGE_ROOT}" \
  --image-source-prefix "${IMAGE_PREFIX}" \
  \
  --network lokr \
  --network-dim 512 \
  --network-alpha 512 \
  --no-grad-checkpoint \
  --tf32 \
  --dtype bf16
