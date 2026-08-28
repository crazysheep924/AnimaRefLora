#!/usr/bin/env bash
# 2-image preview for the forked-horn OC (scene + kimono finger-licking).
set -euo pipefail
REPO="/path/to/anima-reflora"
STORAGE="/path/to/storage"
RUN="experiments/headroi-rope-cpm-f1anticopy-idinject"
STEP="${STEP:-500000}"
IMAGE="anima-reflora-runpod:latest"
REFS="${REPO}/PaperSample/eval_refs/single_oc_horns"
OUT="${REPO}/PaperSample/preview"

appearance="adult woman, tall slender, deep wine-purple hair with blue-violet highlights, neat layered bangs, long sidelocks, two thick braids rising from the sides in symmetrical arcs and wrapping around the horn prongs, braided loop hair, short ivory rhinoceros-beetle horn on the forehead splitting into two short prongs, silver hair clasps on the braids, heterochromia, icy cyan left eye, amber-gold right eye, pale skin, calm cold aristocratic expression, small silver earrings"
kimono_outfit="traditional japanese kimono, floral pattern, wide sleeves, obi sash, elegant"
scene="1girl, solo, sitting at an outdoor cafe table, white sleeveless sundress, sandals, sunny Mediterranean seaside, blue ocean, bougainvillea, warm sunlight, three-quarter view"
lick="1girl, solo, full body, kneeling on the floor, from above, high angle shot, looking up at viewer, seductive light smile, licking finger, index finger to lips, head tilt"

gen() {
  local output="$1" prompt="$2" seed="$3"
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
      --checkpoint "/work/RunpodTraining/${RUN}/checkpoints/lora_step_${STEP}.safetensors" \
      --ref-root /refs --wrong-ref /work/RunpodTraining/wrong.jpg \
      --output-dir "/work/${output#"${REPO}/"}" \
      --storage /workspace/storage \
      --ccip-cache /workspace/storage/runs/ccip_ref_head_emb_cache.pt \
      --head-roi-cache /workspace/storage/runs/head_roi_cache.pt \
      --prompt "${prompt}" --limit 1 --seed "${seed}" \
      --steps 24 --flow-shift 3.0 --guidance-scale 4.5 \
      --bucket-short 1024 --bucket-long-max 1024 \
      --device cuda --decode-device cuda --vae-chunk-size 8 \
      --conditions correct --same-condition-seed --skip-grid
}

gen "${OUT}/scene_oc_horns_v3" "${scene}, ${appearance}" 514
gen "${OUT}/lick_kimono_oc_horns_v3" "${lick}, ${appearance}, ${kimono_outfit}, simple background" 3200
echo "=== PREVIEW DONE ==="
