#!/usr/bin/env bash
# Build the minimal open-source distributable of ComfyUI-AnimaRefLora.
#
# Carries ONLY the inference closure (traced by running Loader->Encode->Sampler
# end-to-end and recording loaded modules) + a small safety margin:
#   - anima_reflora/: 16 of 38 modules (no train.py / noise.py / build caches)
#   - sd-scripts/: 24 library files + 4 networks files + LICENSE (Apache-2.0)
# Trainer entry points (anima_train_network.py etc.) are NOT shipped; the
# sd-scripts presence marker is library/anima_utils.py (see sd_scripts_bridge).
#
# Usage: scripts/build_comfyui_plugin_dist.sh [OUT_DIR]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN="${REPO}/comfyui/ComfyUI-AnimaRefLora"
ANIMA_REFLORA="${REPO}/anima_reflora"
SD="${REPO}/sd-scripts"
OUT="${1:-${REPO}/dist/ComfyUI-AnimaRefLora}"

ROOT_FILES=(
  __init__.py
  bootstrap.py
  nodes.py
  README.md
  requirements.txt
)

# Traced inference closure of the standalone nodes (see docstring above).
ANIMA_REFLORA_FILES=(
  __init__.py
  anima_caption.py
  bundle.py
  build_training_cache.py
  cache.py
  ccip_head_cache.py
  checkpoints.py
  config.py
  crepa.py
  features.py
  head_cache.py
  local_ref_ab_infer.py
  models.py
  paths.py
  ref_conditioning.py
  rope_refpos.py
  sd_scripts_bridge.py
)

# NOTE: anima_train_utils / checkpoint_io / model_util (and their IO deps) are
# deliberately absent — build_training_cache.load_vae imports
# qwen_image_autoencoder_kl directly, keeping diffusers' SD pipelines out of
# the inference import chain (they break on some transformers/diffusers combos).
SD_LIBRARY_FILES=(
  __init__.py
  accelerator_setup.py
  anima_models.py
  anima_utils.py
  attention.py
  caching.py
  custom_offloading_utils.py
  deepspeed_utils.py
  device_utils.py
  fp8_optimization_utils.py
  lora_utils.py
  qwen_image_autoencoder_kl.py
  safetensors_utils.py
  sdxl_original_unet.py
  strategy_anima.py
  strategy_base.py
  utils.py
)

# lokr/loha traced; lora_anima added so network=lora checkpoints load too.
SD_NETWORKS_FILES=(
  lokr.py
  loha.py
  lora_anima.py
  network_base.py
)

rm -rf "${OUT}"
mkdir -p "${OUT}/anima_reflora" "${OUT}/sd-scripts/library" "${OUT}/sd-scripts/networks" "${OUT}/sd-scripts/configs"

for f in "${ROOT_FILES[@]}"; do cp "${PLUGIN}/${f}" "${OUT}/${f}"; done
for f in "${ANIMA_REFLORA_FILES[@]}"; do cp "${ANIMA_REFLORA}/${f}" "${OUT}/anima_reflora/${f}"; done
for f in "${SD_LIBRARY_FILES[@]}"; do cp "${SD}/library/${f}" "${OUT}/sd-scripts/library/${f}"; done
for f in "${SD_NETWORKS_FILES[@]}"; do cp "${SD}/networks/${f}" "${OUT}/sd-scripts/networks/${f}"; done
cp "${SD}/LICENSE.md" "${OUT}/sd-scripts/LICENSE.md"
# Tokenizer/config assets — required by the Anima text-encoder strategy
# (qwen3_06b) and by load_t5_tokenizer for target/source attention ids (t5_old).
cp -r "${SD}/configs/qwen3_06b" "${OUT}/sd-scripts/configs/qwen3_06b"
cp -r "${SD}/configs/t5_old" "${OUT}/sd-scripts/configs/t5_old"

echo "built: ${OUT}"
du -sh "${OUT}"
