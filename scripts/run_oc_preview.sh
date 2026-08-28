#!/usr/bin/env bash
# Quick 4-image preview of the new OC prompts (does not touch blog assets).
set -euo pipefail
REPO="/path/to/anima-reflora"
STORAGE="/path/to/storage"
RUN="experiments/headroi-rope-cpm-f1anticopy-idinject"
STEP="${STEP:-500000}"
IMAGE="anima-reflora-runpod:latest"
REFS="${REPO}/PaperSample/eval_refs"
OUT="${REPO}/PaperSample/preview"

characters=(oc_bunny oc_unicorn)
appearance=(
  "long vibrant red hair with black underlayers and neat bangs, glowing pink eyes, fair skin, slightly flushed cheeks, soft mischievous smile, mismatched bunny ears, one solid black, one soft pinkish-white, small pink flowers and silver rings on the ears, sparkling butterfly hair clips, star-shaped hairpins, blue crystal ornaments"
  "long dark blue-black hair in high twin tails, soft purple-blue gradient tips, neat bangs, spiral unicorn horn on forehead glowing with purple and blue magical light and sparkles, pale skin, light blue eyes, calm neutral expression, silver triangle hair ornaments on twin tails, star earring"
)
kimono_outfit="traditional japanese kimono, floral pattern, wide sleeves, obi sash, elegant"
scene="1girl, solo, sitting at an outdoor cafe table, white sleeveless sundress, straw hat, sandals, sunny Mediterranean seaside, blue ocean, bougainvillea, warm sunlight, three-quarter view"
lick="1girl, solo, full body, kneeling on the floor, from above, high angle shot, looking up at viewer, seductive light smile, licking finger, index finger to lips, head tilt"

gen() {
  local ref_mount="$1" output="$2" prompt="$3" seed="$4"
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
    -v "${ref_mount}:/refs:ro" \
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

for char_i in "${!characters[@]}"; do
  character="${characters[$char_i]}"
  gen "${REFS}/single_${character}" "${OUT}/scene_${character}" \
    "${scene}, ${appearance[$char_i]}" $((314 + char_i * 100))
  gen "${REFS}/single_${character}" "${OUT}/lick_kimono_${character}" \
    "${lick}, ${appearance[$char_i]}, ${kimono_outfit}, simple background" $((3000 + char_i * 100))
done
echo "=== PREVIEW DONE ==="
