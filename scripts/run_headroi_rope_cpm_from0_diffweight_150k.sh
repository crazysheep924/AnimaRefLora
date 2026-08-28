#!/usr/bin/env bash
# FROM-0 full-stack anti-copy recipe (150k steps, ~3.5 days on a 4090-class GPU).
# Everything the 80k-resume experiments validated, active from step 0 so the
# "weak caption -> copy ref" shortcut never becomes the dominant solution:
#   - diff-weighted target MSE (lambda 0.5, head-ROI exempt)   [8AFC eval: copy 41% -> 5%]
#   - 差分-aware ref pairing (pair_dhash_cache, min hamming 25)
#   - tag-level caption dropout (NO plain caption dropout - it caused the reflex)
#   - head_loss_weight 2.0 (not 4.0: don't over-strengthen the F0 pathway)
# POLICY: outputs go to the CONTAINER disk, never the network volume/S3
# (see run_headroi_rope_cpm_editfix2_80k_diffweight.sh for the rationale).
# Usage:
#   bash /opt/AnimaRefLora/scripts/run_headroi_rope_cpm_from0_diffweight_150k.sh
set -euo pipefail

STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace}"
OUT="${OUT:-/opt/AnimaRefLora/runs}"
IMAGE_ROOT="${ANIMA_REFLORA_IMAGES:-${STORAGE}/dataset/images}"
IMAGE_PREFIX="${ANIMA_REFLORA_IMAGE_PREFIX:-/path/to/dataset/images/}"
CCIP_CACHE="${CCIP_CACHE:-${STORAGE}/runs/ccip_ref_head_emb_cache.pt}"
HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-${STORAGE}/runs/head_roi_cache.pt}"

STEPS="${STEPS:-150000}"

LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}"
DIFF_LOSS_LAMBDA="${DIFF_LOSS_LAMBDA:-0.5}"
DIFF_WEIGHT_MIN="${DIFF_WEIGHT_MIN:-0.2}"
HEAD_LOSS_WEIGHT="${HEAD_LOSS_WEIGHT:-2.0}"

PAIR_DHASH_CACHE="${PAIR_DHASH_CACHE:-${STORAGE}/runs/pair_dhash_cache.pt}"
PAIR_MIN_DHASH="${PAIR_MIN_DHASH:-25}"

REF_DROPOUT_PROB="${REF_DROPOUT_PROB:-0.25}"
CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}"
TAG_DROPOUT_PROB="${TAG_DROPOUT_PROB:-0.5}"
TAG_KEEP_PROB="${TAG_KEEP_PROB:-0.5}"
TAG_KEEP_MIN="${TAG_KEEP_MIN:-3}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-headroi-rope-cpm-from0-diffw-${RUN_TAG}}"

if [ ! -f "${PAIR_DHASH_CACHE}" ]; then
  echo "ERROR: pair-dhash cache missing: ${PAIR_DHASH_CACHE}" >&2
  echo "Build/upload pair_dhash_cache.pt first, or set PAIR_DHASH_CACHE=<path>." >&2
  exit 2
fi

echo "=== FROM-0 ${RUN_NAME} ==="
echo "steps: ${STEPS}  out: ${OUT}"
echo "diff-weight: lambda=${DIFF_LOSS_LAMBDA}, min=${DIFF_WEIGHT_MIN}, head_loss_weight=${HEAD_LOSS_WEIGHT}"
echo "pair-dhash: ${PAIR_DHASH_CACHE} (min ${PAIR_MIN_DHASH})"
echo ""

exec python -m anima_reflora.train \
  --stage from0-headroi-rope-cpm \
  --run-name "${RUN_NAME}" \
  --from-scratch \
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
