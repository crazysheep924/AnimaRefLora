#!/usr/bin/env bash
# Base-model swap probe: rerun the blog's Pose x Outfit edit (stage 3 of
# run_oc_blog_eval.sh — finger-licking kneeling, original outfit vs kimono)
# with the WAI Anima community finetune as the DiT base instead of
# anima-base-v1.0, same 500K AnimaRefLoRA checkpoint / seeds / prompts.
# Outputs land in a separate dir so blog assets stay untouched.
set -euo pipefail

REPO="${REPO:-/path/to/anima-reflora}"
STORAGE="${STORAGE:-/path/to/storage}"
RUN="${RUN:-experiments/headroi-rope-cpm-f1anticopy-idinject}"
STEP="${STEP:-500000}"
IMAGE="${IMAGE:-anima-reflora-runpod:latest}"
REFS="${REPO}/PaperSample/eval_refs"
DIT_FILE="${DIT_FILE:-waiANIMA_v10Base10.safetensors}"
RESULTS="${RESULTS:-${REPO}/PaperSample/waibase_finger_licking}"

characters=(oc_bunny oc_horns)
appearance=(
  "long vibrant red hair with black underlayers and neat bangs, a bold black X-shaped marking on the front bangs, glowing pink eyes, fair skin, slightly flushed cheeks, soft mischievous smile, mismatched bunny ears, one solid black, one soft pinkish-white, small pink flowers and silver rings on the ears, sparkling butterfly hair clips, star-shaped hairpins, blue crystal ornaments"
  "adult woman, tall slender, deep wine-purple hair with blue-violet highlights, neat layered bangs, long sidelocks, two thick braids rising from the sides in symmetrical arcs and wrapping around the horn prongs, braided loop hair, short ivory rhinoceros-beetle horn on the forehead splitting into two short prongs, silver hair clasps on the braids, heterochromia, icy cyan left eye, amber-gold right eye, pale skin, calm cold aristocratic expression, small silver earrings"
)
original_outfit=(
  "thick pink choker with heart pendant, pink heart-themed bralette-style camisole top with metallic heart charms and silver buckles, crisscrossing pink leather harness straps across chest and shoulders, sheer iridescent light-pink and lavender off-shoulder jacket hanging loosely, embroidered butterflies and gem details on the jacket, dangling pearl chains"
  "fitted black high-collar gothic bodice, amethyst gemstone brooch at the collar, symmetrical silver piping and geometric seam lines, structured angular puff sleeves, fitted black gloves, silver-trimmed wrist cuffs, wide black corset belt with purple gemstone ornament, charcoal bell-shaped skirt with geometric panels and silver piping, dark purple pleated underskirt, opaque black stockings, black knee-high boots with silver cross-lacing and buckle straps"
)
kimono_outfit="traditional japanese kimono, floral pattern, wide sleeves, obi sash, elegant"

infer() {
  local ref_mount="$1" output="$2" prompt="$3" seed="$4"
  if [[ -f "${output}/manifest.json" ]]; then
    echo "[skip] ${output}"
    return
  fi
  mkdir -p "${output}"
  echo "[gen] ${output}"
  docker run --gpus all --rm \
    --ulimit nofile=1048576:1048576 \
    -e ANIMA_REFLORA_STORAGE=/workspace/storage \
    -e ANIMA_REFLORA_MODEL_DIT="/workspace/storage/anima_models/diffusion_models/${DIT_FILE}" \
    -e ANIMA_REFLORA_MODEL_TE=/workspace/storage/anima_models/text_encoders \
    -e ANIMA_REFLORA_MODEL_VAE=/workspace/storage/anima_models/vae/qwen_image_vae.safetensors \
    -v "${REPO}/anima_reflora:/opt/AnimaRefLora/anima_reflora" \
    -v "${REPO}/sd-scripts/library/anima_utils.py:/opt/AnimaRefLora/sd-scripts/library/anima_utils.py:ro" \
    -v "${STORAGE}:/workspace/storage" \
    -v "${ref_mount}:/refs:ro" \
    -v "${REPO}:/work" \
    "${IMAGE}" python -m anima_reflora.local_ref_ab_infer \
      --checkpoint "/work/RunpodTraining/${RUN}/checkpoints/lora_step_${STEP}.safetensors" \
      --ref-root /refs \
      --wrong-ref /work/RunpodTraining/wrong.jpg \
      --output-dir "/work/${output#"${REPO}/"}" \
      --storage /workspace/storage \
      --ccip-cache /workspace/storage/runs/ccip_ref_head_emb_cache.pt \
      --head-roi-cache /workspace/storage/runs/head_roi_cache.pt \
      --prompt "${prompt}" \
      --limit 1 \
      --seed "${seed}" \
      --steps 24 \
      --flow-shift 3.0 \
      --guidance-scale 4.5 \
      --bucket-short 1024 \
      --bucket-long-max 1024 \
      --device cuda \
      --decode-device cuda \
      --vae-chunk-size 8 \
      --conditions correct \
      --same-condition-seed \
      --skip-grid
}

lick_pose="full body, kneeling on the floor, from above, high angle shot, looking up at viewer, seductive light smile, licking finger, index finger to lips, head tilt"
for char_i in "${!characters[@]}"; do
  character="${characters[$char_i]}"
  seed=$((3000 + char_i * 100))
  infer "${REFS}/single_${character}" "${RESULTS}/original/${character}" \
    "1girl, solo, ${lick_pose}, ${appearance[$char_i]}, ${original_outfit[$char_i]}, simple background" "${seed}"
  infer "${REFS}/single_${character}" "${RESULTS}/kimono/${character}" \
    "1girl, solo, ${lick_pose}, ${appearance[$char_i]}, ${kimono_outfit}, simple background" "${seed}"
done

echo "=== WAI-base pose x outfit DONE -> ${RESULTS} ==="
