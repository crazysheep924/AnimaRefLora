#!/usr/bin/env bash
# RESUME the from0-diffw run with f1_anti_copy enabled — single-knob probe.
# Rationale (2026-07-06): NCC factorial re-verification proved F1 is the pixel
# content channel for copying, so the margin penalty targets the right frame.
# 145k from0-diffw still shows recidivist-ref copies; everything
# else in the recipe stays IDENTICAL to run_headroi_rope_cpm_from0_diffweight_150k.sh
# so any copy-rate change attributes to f1_anti_copy alone.
# Designed for the local RTX 5090 (32GB): --grad-checkpoint ON, batch 1.
# Usage: see scripts/launch_f1anticopy_local5090.sh
set -euo pipefail

STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace}"
OUT="${OUT:-/opt/AnimaRefLora/runs}"
IMAGE_ROOT="${ANIMA_REFLORA_IMAGES:-${STORAGE}/dataset/images}"
IMAGE_PREFIX="${ANIMA_REFLORA_IMAGE_PREFIX:-/path/to/dataset/images/}"
CCIP_CACHE="${CCIP_CACHE:-${STORAGE}/runs/ccip_ref_head_emb_cache.pt}"
HEAD_ROI_CACHE="${ANIMA_REFLORA_HEAD_ROI_CACHE:-${STORAGE}/runs/head_roi_cache.pt}"

STEPS="${STEPS:-200000}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"

LATENT_RECON_LOSS_WEIGHT="${LATENT_RECON_LOSS_WEIGHT:-0.1}"
DIFF_LOSS_LAMBDA="${DIFF_LOSS_LAMBDA:-0.5}"
DIFF_WEIGHT_MIN="${DIFF_WEIGHT_MIN:-0.2}"
HEAD_LOSS_WEIGHT="${HEAD_LOSS_WEIGHT:-2.0}"

F1_ANTI_COPY_WEIGHT="${F1_ANTI_COPY_WEIGHT:-0.05}"
F1_ANTI_COPY_MARGIN="${F1_ANTI_COPY_MARGIN:-0.35}"
F1_ANTI_COPY_SIGMA_CUTOFF="${F1_ANTI_COPY_SIGMA_CUTOFF:-0.8}"

PAIR_DHASH_CACHE="${PAIR_DHASH_CACHE:-${STORAGE}/runs/pair_dhash_cache.pt}"
PAIR_MIN_DHASH="${PAIR_MIN_DHASH:-25}"
# singleton (char,bucket) cells: 'blank' = train unconditional branch instead of
# the self-pair copy signal (~32% of the expanded data would otherwise self-pair).
SINGLETON_REF_MODE="${SINGLETON_REF_MODE:-blank}"

REF_DROPOUT_PROB="${REF_DROPOUT_PROB:-0.25}"
CAPTION_DROPOUT_PROB="${CAPTION_DROPOUT_PROB:-0.0}"
TAG_DROPOUT_PROB="${TAG_DROPOUT_PROB:-0.5}"
TAG_KEEP_PROB="${TAG_KEEP_PROB:-0.5}"
TAG_KEEP_MIN="${TAG_KEEP_MIN:-3}"
# Identity-accessory injection: re-insert per-image signature accessory words
# (sig_subtract stripped them; anti-copy then dropped them from the output).
# inject-prob is a KEEP rate applied when the GT image has the accessory; 0 = off.
IDENTITY_INJECT_MAP="${IDENTITY_INJECT_MAP:-${STORAGE}/runs/identity_inject_map.json}"
IDENTITY_INJECT_PROB="${IDENTITY_INJECT_PROB:-0.0}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-headroi-rope-cpm-f1anticopy-200k}"
# SOURCE_CKPT: the 145K checkpoint from run_headroi_rope_cpm_from0_diffweight_150k.sh.
SOURCE_CKPT="${SOURCE_CKPT:-}"
if [ -z "${RESUME_CKPT:-}" ]; then
  latest="$(find "${OUT}/experiments/${RUN_NAME}/checkpoints" -maxdepth 1 -name 'lora_step_*.safetensors' 2>/dev/null | sort -V | tail -n 1 || true)"
  RESUME_CKPT="${latest:-${SOURCE_CKPT}}"
fi

if [ -z "${RESUME_CKPT}" ] || [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: RESUME_CKPT not found: '${RESUME_CKPT}'" >&2
  echo "Set SOURCE_CKPT=<...>/checkpoints/lora_step_145000.safetensors (or RESUME_CKPT=...)" >&2
  exit 2
fi
if [ ! -f "${PAIR_DHASH_CACHE}" ]; then
  echo "ERROR: pair-dhash cache missing: ${PAIR_DHASH_CACHE}" >&2
  exit 2
fi

echo "=== RESUME+f1_anti_copy ${RUN_NAME} ==="
echo "resume: ${RESUME_CKPT}"
echo "steps: ${STEPS}  ckpt-every: ${CKPT_EVERY}  out: ${OUT}"
echo "f1-anti-copy: weight=${F1_ANTI_COPY_WEIGHT}, margin=${F1_ANTI_COPY_MARGIN}, sigma-cutoff=${F1_ANTI_COPY_SIGMA_CUTOFF}"
echo "diff-weight: lambda=${DIFF_LOSS_LAMBDA}, min=${DIFF_WEIGHT_MIN}, head_loss_weight=${HEAD_LOSS_WEIGHT}"
echo ""

exec python -m anima_reflora.train \
  --stage from0-headroi-rope-cpm \
  --run-name "${RUN_NAME}" \
  --allow-existing-run \
  --resume "${RESUME_CKPT}" \
  --no-resume-data-skip \
  --steps "${STEPS}" \
  --batch 1 \
  --lr "${LR}" \
  --optimizer came \
  --warmup-steps "${WARMUP_STEPS}" \
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
  --f1-anti-copy-weight "${F1_ANTI_COPY_WEIGHT}" \
  --f1-anti-copy-margin "${F1_ANTI_COPY_MARGIN}" \
  --f1-anti-copy-sigma-cutoff "${F1_ANTI_COPY_SIGMA_CUTOFF}" \
  --pair-dhash-cache "${PAIR_DHASH_CACHE}" \
  --pair-min-dhash "${PAIR_MIN_DHASH}" \
  --singleton-ref-mode "${SINGLETON_REF_MODE}" \
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
  --identity-inject-map "${IDENTITY_INJECT_MAP}" \
  --identity-inject-prob "${IDENTITY_INJECT_PROB}" \
  \
  --weighting-scheme none \
  --prompt-mode change_only \
  \
  --ckpt-every "${CKPT_EVERY}" \
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
  --grad-checkpoint \
  --tf32 \
  --dtype bf16
