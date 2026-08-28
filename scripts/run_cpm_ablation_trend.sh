#!/usr/bin/env bash
# CPM inference-contribution trend: baseline vs --no-cpm-component at several
# checkpoints, same seed/prompt, oc_bunny (the ref whose CCIP embedding is valid).
set -euo pipefail
REPO="/path/to/anima-reflora"
STORAGE="/path/to/storage"
RUN="experiments/headroi-rope-cpm-f1anticopy-idinject"
IMAGE="anima-reflora-runpod:latest"
REFS="${REPO}/PaperSample/eval_refs/single_oc_bunny"
OUT="${REPO}/PaperSample/cpm_ablation_trend"
STEPS_LIST=(160000 275000 400000 500000)
PROMPT="1girl, solo, standing, white dress, simple background, looking at viewer"

gen() {
  local step="$1" variant="$2"; shift 2
  local extra_args=("$@")
  local output="${OUT}/step${step}/${variant}"
  if [[ -f "${output}/manifest.json" ]]; then echo "[skip] ${output}"; return; fi
  mkdir -p "${output}"
  echo "[gen] ${output}"
  docker run --gpus all --rm \
    --ulimit nofile=1048576:1048576 \
    -e ANIMA_REFLORA_STORAGE=/workspace/storage \
    -e ANIMA_REFLORA_MODEL_DIT=/workspace/storage/anima_models/diffusion_models/anima-base-v1.0.safetensors \
    -e ANIMA_REFLORA_MODEL_TE=/workspace/storage/anima_models/text_encoders \
    -e ANIMA_REFLORA_MODEL_VAE=/workspace/storage/anima_models/vae/qwen_image_vae.safetensors \
    -v "${REPO}/anima_reflora:/opt/AnimaRefLora/anima_reflora" \
    -v "${STORAGE}:/workspace/storage" \
    -v "${REFS}:/refs:ro" \
    -v "${REPO}:/work" \
    "${IMAGE}" python -m anima_reflora.local_ref_ab_infer \
      --checkpoint "/work/RunpodTraining/${RUN}/checkpoints/lora_step_${step}.safetensors" \
      --ref-root /refs --wrong-ref /work/RunpodTraining/wrong.jpg \
      --output-dir "/work/${output#"${REPO}/"}" \
      --storage /workspace/storage \
      --ccip-cache /workspace/storage/runs/ccip_ref_head_emb_cache.pt \
      --head-roi-cache /workspace/storage/runs/head_roi_cache.pt \
      --prompt "${PROMPT}" --limit 1 --seed 4000 \
      --steps 24 --flow-shift 3.0 --guidance-scale 4.5 \
      --bucket-short 1024 --bucket-long-max 1024 \
      --device cuda --decode-device cuda --vae-chunk-size 8 \
      --conditions correct --same-condition-seed --skip-grid \
      "${extra_args[@]}"
}

for step in "${STEPS_LIST[@]}"; do
  echo "=== step ${step} ==="
  gen "${step}" baseline
  gen "${step}" cpm_off --no-cpm-component
done
echo "=== TREND GEN DONE ==="
