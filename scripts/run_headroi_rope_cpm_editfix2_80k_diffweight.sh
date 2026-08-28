#!/usr/bin/env bash
# Continue the editfix2 resume run from step 80000 with diff-weighted target MSE.
# Rationale (2026-07-02 factorial annotation + 差分 scan):
#   - copy pathway is dominated by the F0 head frame, not F1 -> pathway-agnostic
#     diff weighting replaces the F1-only hinge anti-copy loss (weight stays 0).
#   - head_loss_weight lowered 4.0 -> 2.0 to stop over-strengthening the F0
#     identity pathway (CREPA head_roi + CPM keep carrying identity).
# Usage:
#   bash /opt/AnimaRefLora/scripts/run_headroi_rope_cpm_editfix2_80k_diffweight.sh
set -euo pipefail

STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace}"
# POLICY: do NOT write training outputs to S3 / the network volume (/workspace).
# The network volume backs the S3 bucket and its quota is largely consumed by
# _latcache; writing checkpoints there can abort a long run mid-training with
# "Disk quota exceeded". Outputs go to the CONTAINER disk instead — ephemeral
# by design, download wanted checkpoints manually.
# Deliberately ignores the image env ANIMA_REFLORA_OUT
# (/workspace/Persistent/runs); override with OUT=... only if you know better.
OUT="${OUT:-/opt/AnimaRefLora/runs}"
IMAGE_ROOT="${ANIMA_REFLORA_IMAGES:-${STORAGE}/dataset/images}"
IMAGE_PREFIX="${ANIMA_REFLORA_IMAGE_PREFIX:-/path/to/dataset/images/}"
CCIP_CACHE="${CCIP_CACHE:-${STORAGE}/runs/ccip_ref_head_emb_cache.pt}"
HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-${STORAGE}/runs/head_roi_cache.pt}"

# Default resume point matches where the user unpacks checkpoints on the pod:
# /opt/AnimaRefLora/runs/experiments/80K/checkpoints/
SOURCE_RUN="${SOURCE_RUN:-80K}"
RESUME_CKPT="${RESUME_CKPT:-${OUT}/experiments/${SOURCE_RUN}/checkpoints/lora_step_80000.safetensors}"
STEPS="${STEPS:-100000}"

LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}"
DIFF_LOSS_LAMBDA="${DIFF_LOSS_LAMBDA:-0.5}"
DIFF_WEIGHT_MIN="${DIFF_WEIGHT_MIN:-0.2}"
HEAD_LOSS_WEIGHT="${HEAD_LOSS_WEIGHT:-2.0}"

# 差分-aware ref pairing: excludes near-duplicate refs (hamming < PAIR_MIN_DHASH
# of 256) so (ref, target) pairs are real edits, not effective self-pairs.
# Missing cache file -> graceful fallback to random pairing (a log line says so).
PAIR_DHASH_CACHE="${PAIR_DHASH_CACHE:-${STORAGE}/runs/pair_dhash_cache.pt}"
PAIR_MIN_DHASH="${PAIR_MIN_DHASH:-25}"

REF_DROPOUT_PROB="${REF_DROPOUT_PROB:-0.25}"
CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}"
TAG_DROPOUT_PROB="${TAG_DROPOUT_PROB:-0.5}"
TAG_KEEP_PROB="${TAG_KEEP_PROB:-0.5}"
TAG_KEEP_MIN="${TAG_KEEP_MIN:-3}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-headroi-rope-cpm-editfix2-80k-diffw-${RUN_TAG}}"

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: RESUME_CKPT not found: ${RESUME_CKPT}" >&2
  echo "Set RESUME_CKPT=<...>/checkpoints/lora_step_80000.safetensors" >&2
  exit 2
fi

echo "=== Continuing ${RUN_NAME} ==="
echo "resume: ${RESUME_CKPT}"
echo "steps:  ${STEPS}"
echo "diff-weight: lambda=${DIFF_LOSS_LAMBDA}, min=${DIFF_WEIGHT_MIN}, head_loss_weight=${HEAD_LOSS_WEIGHT}"
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
  --diff-loss-lambda "${DIFF_LOSS_LAMBDA}" \
  --diff-weight-min "${DIFF_WEIGHT_MIN}" \
  --pair-dhash-cache "${PAIR_DHASH_CACHE}" \
  --pair-min-dhash "${PAIR_MIN_DHASH}" \
  --head-loss-weight "${HEAD_LOSS_WEIGHT}" \
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
