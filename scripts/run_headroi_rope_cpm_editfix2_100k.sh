#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  RunPod 100k-step training: head-ROI + RoPE + CPM + CREPA  (EDITFIX2 variant)
#  From scratch, batch=1, T=3.
#
#  Difference vs run_headroi_rope_cpm_editfix_100k.sh:
#    * --caption-dropout-prob 0   (was 0.10) — remove whole-caption CFG dropout; it
#      instilled a "weak/empty caption -> copy the reference" reflex that made short
#      prompts collapse to ref-copy as early as step 10k.
#    * tag-LEVEL dropout added:
#        --tag-dropout-prob 0.5  --tag-keep-prob 0.5  --tag-keep-min 3
#      Each triggered step re-encodes the caption after randomly dropping a subset of
#      its delta tags (structural prefix score/quality/recency/rating/year + subject
#      anchors 1girl/solo/... always kept). This trains the model on short-but-non-empty
#      captions so short user prompts stay in-distribution and the model fills unstated
#      attributes instead of copying the reference.
#    * --ref-dropout-prob 0.25 kept (same as editfix).
#    * new RUN_NAME "headroi-rope-cpm-editfix2-100k-*".
#  Goal: make SHORT prompts follow the prompt instead of copying the reference.
#  See conversation analysis + .claude/plans/fuzzy-finding-firefly.md.
#
#  Usage:
#    bash /opt/AnimaRefLora/scripts/run_headroi_rope_cpm_editfix2_100k.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── paths (all overridable via env) ──
STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace}"
OUT="${ANIMA_REFLORA_OUT:-/workspace/Persistent/runs}"
IMAGE_ROOT="${ANIMA_REFLORA_IMAGES:-${STORAGE}/dataset/images}"
IMAGE_PREFIX="${ANIMA_REFLORA_IMAGE_PREFIX:-/path/to/dataset/images/}"
CCIP_CACHE="${CCIP_CACHE:-${STORAGE}/runs/ccip_ref_head_emb_cache.pt}"
HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-${STORAGE}/runs/head_roi_cache.pt}"
LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}"

# editability knobs (overridable)
REF_DROPOUT_PROB="${REF_DROPOUT_PROB:-0.25}"
CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}"
TAG_DROPOUT_PROB="${TAG_DROPOUT_PROB:-0.5}"
TAG_KEEP_PROB="${TAG_KEEP_PROB:-0.5}"
TAG_KEEP_MIN="${TAG_KEEP_MIN:-3}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="headroi-rope-cpm-editfix2-100k-${RUN_TAG}"

# ── step 0: preflight ──
echo "=== Preflight ==="
python -m anima_reflora.preflight \
  --stage from0-headroi-rope-cpm \
  --from-scratch \
  --frames 3 \
  --rope-refpos \
  --cpm \
  --crepa \
  --crepa-pool head_roi \
  --latent-recon-loss-weight "${LATENT_RECON_LOSS_WEIGHT}" \
  --head-loss-weight 4.0 \
  --ccip-cache "${CCIP_CACHE}" \
  --head-roi-cache "${HEAD_ROI_CACHE}" \
  --storage "${STORAGE}" \
  --out-dir "${OUT}" \
  --image-root "${IMAGE_ROOT}" \
  --image-source-prefix "${IMAGE_PREFIX}" \
  --run-name "${RUN_NAME}" \
  --json

echo ""
echo "=== Starting training: ${RUN_NAME} (ref-dropout=${REF_DROPOUT_PROB}, caption-dropout=${CAPTION_DROPOUT_PROB}, tag-dropout=${TAG_DROPOUT_PROB}/keep=${TAG_KEEP_PROB}/min=${TAG_KEEP_MIN}) ==="
echo ""

# ── step 1: train ──
exec python -m anima_reflora.train \
  --stage from0-headroi-rope-cpm \
  --run-name "${RUN_NAME}" \
  --from-scratch \
  --steps 100000 \
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
