#!/usr/bin/env bash
# Generate the OC-character project-page assets with the 500K checkpoint:
#   stage 1  four 2x2 scene grids (no_appearance / with_appearance)
#   stage 2  original-outfit pose transfer (wave, crossed_arms)
#   stage 3  finger-licking kneeling from-above grid (original outfit vs kimono)
#   stage 4  component-ablation grid (baseline / F0-only / F1-only / blank / CPM-off / LoRA-off)
#   stage 5  val-5 ref A/B (5SAMPLE) + 8AFC ref A/B via gen_ref_ab_all.py
set -euo pipefail

REPO="${REPO:-/path/to/anima-reflora}"
STORAGE="${STORAGE:-/path/to/storage}"
RUN="${RUN:-experiments/headroi-rope-cpm-f1anticopy-idinject}"
STEP="${STEP:-500000}"
IMAGE="${IMAGE:-anima-reflora-runpod:latest}"
REFS="${REPO}/PaperSample/eval_refs"
RESULTS="${RESULTS:-${REPO}/blog/assets/results}"

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

scenes=(seaside_cafe winter_closeup petite_studio curvy_studio)
scene_seeds=(314 808 1600 1600)
prompts=(
  "1girl, solo, pov, feeding the viewer, holding out a spoonful of dessert toward the viewer, incoming food, leaning forward, looking at viewer, gentle smile, cowboy shot, hips-up framing, white sleeveless sundress, seated at an outdoor seaside cafe table, sunny Mediterranean terrace, blue ocean and bougainvillea in background, warm sunlight, detailed face and hair"
  "1girl, solo, pov, holding hands with the viewer, pov hand, walking ahead and looking back at viewer over her shoulder, gentle happy smile, cowboy shot, warm winter coat, wool scarf, snowy christmas market street at night, glowing wooden market stalls, warm string lights, falling snow, warm bokeh, detailed face and hair"
  "1girl, solo, adult woman, petite body, small breasts, slim waist, cowboy shot, fitted navy turtleneck sweater, high-waisted beige trousers, modern photo studio, neutral gray backdrop, standing, hands on hips, looking at viewer, softbox lighting"
  "1girl, solo, adult woman, curvy body, large breasts, wide hips, cowboy shot, fitted navy turtleneck sweater, high-waisted beige trousers, modern photo studio, neutral gray backdrop, standing, hands on hips, looking at viewer, softbox lighting"
)

infer() {
  local ref_mount="$1" output="$2" prompt="$3" seed="$4" limit="$5" conditions="${6:-correct,blank}"
  local extra_args=()
  if (( $# > 6 )); then extra_args=("${@:7}"); fi
  if [[ -f "${output}/manifest.json" && "${FORCE_OUTPUT:-}" != "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
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
      --ref-root /refs \
      --wrong-ref /work/RunpodTraining/wrong.jpg \
      --output-dir "/work/${output#"${REPO}/"}" \
      --storage /workspace/storage \
      --ccip-cache /workspace/storage/runs/ccip_ref_head_emb_cache.pt \
      --head-roi-cache /workspace/storage/runs/head_roi_cache.pt \
      --prompt "${prompt}" \
      --limit "${limit}" \
      --seed "${seed}" \
      --steps 24 \
      --flow-shift 3.0 \
      --guidance-scale 4.5 \
      --bucket-short 1024 \
      --bucket-long-max 1024 \
      --device cuda \
      --decode-device cuda \
      --vae-chunk-size 8 \
      --conditions "${conditions}" \
      --same-condition-seed \
      --skip-grid \
      "${extra_args[@]}"
}

echo "=== stage 1: scene 2x2 grids ==="
for scene_i in "${!scenes[@]}"; do
  scene="${scenes[$scene_i]}"
  base_seed="${scene_seeds[$scene_i]}"
  prompt="${prompts[$scene_i]}"
  infer "${REFS}" "${RESULTS}/${scene}/no_appearance" "${prompt}" "${base_seed}" 2
  for char_i in "${!characters[@]}"; do
    character="${characters[$char_i]}"
    seed=$((base_seed + char_i * 100))
    infer "${REFS}/single_${character}" "${RESULTS}/${scene}/with_appearance/${character}" \
      "${prompt}, ${appearance[$char_i]}" "${seed}" 1
  done
done

echo "=== stage 2: original-outfit poses ==="
pose_names=(wave crossed_arms)
pose_prompts=(
  "upper body portrait, waist up, cropped legs, close framing, waving at viewer, one arm raised, cheerful smile, looking at viewer"
  "upper body portrait, waist up, cropped legs, close framing, arms crossed, three-quarter view, confident smile, looking at viewer"
)
for pose_i in "${!pose_names[@]}"; do
  pose="${pose_names[$pose_i]}"
  for char_i in "${!characters[@]}"; do
    character="${characters[$char_i]}"
    seed=$((2600 + char_i * 100))
    prompt="1girl, solo, ${pose_prompts[$pose_i]}, ${appearance[$char_i]}, ${original_outfit[$char_i]}"
    infer "${REFS}/single_${character}" "${RESULTS}/original_outfit/${pose}/${character}" \
      "${prompt}" "${seed}" 1 correct
  done
done

echo "=== stage 3: finger-licking kneeling from-above (original vs kimono) ==="
lick_pose="full body, kneeling on the floor, from above, high angle shot, looking up at viewer, seductive light smile, licking finger, index finger to lips, head tilt"
for char_i in "${!characters[@]}"; do
  character="${characters[$char_i]}"
  seed=$((3000 + char_i * 100))
  infer "${REFS}/single_${character}" "${RESULTS}/finger_licking/original/${character}" \
    "1girl, solo, ${lick_pose}, ${appearance[$char_i]}, ${original_outfit[$char_i]}, simple background" "${seed}" 1 correct
  infer "${REFS}/single_${character}" "${RESULTS}/finger_licking/kimono/${character}" \
    "1girl, solo, ${lick_pose}, ${appearance[$char_i]}, ${kimono_outfit}, simple background" "${seed}" 1 correct
done

echo "=== stage 4: component-ablation grid ==="
abl_prompt="1girl, solo, standing, white dress, simple background, looking at viewer"
for char_i in "${!characters[@]}"; do
  character="${characters[$char_i]}"
  seed=$((4000 + char_i * 100))
  root="${REFS}/single_${character}"
  infer "${root}" "${RESULTS}/ablation/baseline/${character}"  "${abl_prompt}" "${seed}" 1 correct
  infer "${root}" "${RESULTS}/ablation/f0_only/${character}"   "${abl_prompt}" "${seed}" 1 correct --ref-frame-mode head_only
  infer "${root}" "${RESULTS}/ablation/f1_only/${character}"   "${abl_prompt}" "${seed}" 1 correct --ref-frame-mode full_only
  infer "${root}" "${RESULTS}/ablation/no_ref/${character}"    "${abl_prompt}" "${seed}" 1 correct --ref-frame-mode blank
  infer "${root}" "${RESULTS}/ablation/cpm_off/${character}"   "${abl_prompt}" "${seed}" 1 correct --no-cpm-component
  infer "${root}" "${RESULTS}/ablation/lora_off/${character}"  "${abl_prompt}" "${seed}" 1 correct --zero-lora
done

echo "=== stage 5: 5SAMPLE + 8AFC ref A/B ==="
GEN_DIR="${REPO}/RunpodTraining/${RUN}/generated"
if [[ ! -d "${GEN_DIR}/val5_ref_ab_step${STEP}_1024" ]]; then
  python3 scripts/gen_ref_ab_all.py --run-subdir "${RUN}" --only "${STEP}" \
    --refs /path/to/eval_refs --limit 5 --image "${IMAGE}"
  mv "${GEN_DIR}/ref_ab_step${STEP}_1024" "${GEN_DIR}/val5_ref_ab_step${STEP}_1024"
fi
if [[ ! -d "${GEN_DIR}/8afc_ref_ab_step${STEP}_1024" ]]; then
  python3 scripts/gen_ref_ab_all.py --run-subdir "${RUN}" --only "${STEP}" \
    --refs "${STORAGE}/8AFC_refs" --limit 11 --image "${IMAGE}"
  mv "${GEN_DIR}/ref_ab_step${STEP}_1024" "${GEN_DIR}/8afc_ref_ab_step${STEP}_1024"
fi

echo "=== ALL STAGES DONE ==="
