#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  RunPod 100k-step training: head-ROI + RoPE + CPM + CREPA
#  Batch=1, T=3 frames, from scratch
#
#  Usage:
#    bash /opt/AnimaRefLora/scripts/run_headroi_rope_cpm_100k.sh
#
#  Prerequisites:
#    1. Storage synced  (anima-reflora-entrypoint prepare)
#    2. dataset.tar extracted
#    3. Head-crop CCIP cache built  (anima-reflora-entrypoint head-ccip-cache ...)
#    4. Head ROI cache built  (anima-reflora-entrypoint head-roi-cache ...)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── paths (all overridable via env) ──
STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace}"
OUT="${ANIMA_REFLORA_OUT:-/workspace/Persistent/runs}"
IMAGE_ROOT="${ANIMA_REFLORA_IMAGES:-${STORAGE}/dataset/images}"
IMAGE_PREFIX="${ANIMA_REFLORA_IMAGE_PREFIX:-/path/to/dataset/images/}"
# CPM/CREPA use the HEAD-CROP CCIP prototypes (consistent with --crepa-pool head_roi).
# Override with CCIP_CACHE=... to use a different cache (e.g. full-image ccip_ref_emb_cache.pt).
CCIP_CACHE="${CCIP_CACHE:-${STORAGE}/runs/ccip_ref_head_emb_cache.pt}"
HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-${STORAGE}/runs/head_roi_cache.pt}"
LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="headroi-rope-cpm-100k-${RUN_TAG}"

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
echo "=== Starting training: ${RUN_NAME} ==="
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
  --ref-dropout-prob 0.10 \
  --ref-dropout-mode blank \
  --ref-dropout-t3-mode structured \
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
