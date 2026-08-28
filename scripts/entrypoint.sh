#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-jupyter}"
if [[ $# -gt 0 ]]; then
  shift
fi

export ANIMA_REFLORA_REPO="${ANIMA_REFLORA_REPO:-/opt/AnimaRefLora}"
export ANIMA_REFLORA_STORAGE="${ANIMA_REFLORA_STORAGE:-/workspace/storage}"
export ANIMA_REFLORA_PERSISTENT="${ANIMA_REFLORA_PERSISTENT:-/workspace/Persistent}"
export ANIMA_REFLORA_OUT="${ANIMA_REFLORA_OUT:-${ANIMA_REFLORA_PERSISTENT}/runs}"
export ANIMA_REFLORA_MODEL_TE="${ANIMA_REFLORA_MODEL_TE:-${ANIMA_REFLORA_STORAGE}/anima_models/text_encoders}"

mkdir -p "${ANIMA_REFLORA_STORAGE}" "${ANIMA_REFLORA_PERSISTENT}" "${ANIMA_REFLORA_OUT}"

if [[ -n "${JUPYTER_STORAGE_LINK:-}" ]]; then
  mkdir -p "$(dirname "${JUPYTER_STORAGE_LINK}")"
  ln -sfn "${ANIMA_REFLORA_STORAGE}" "${JUPYTER_STORAGE_LINK}"
fi

if [[ -n "${JUPYTER_PERSISTENT_LINK:-}" ]]; then
  mkdir -p "$(dirname "${JUPYTER_PERSISTENT_LINK}")"
  ln -sfn "${ANIMA_REFLORA_PERSISTENT}" "${JUPYTER_PERSISTENT_LINK}"
fi

prepare_text_encoder_symlink() {
  local te_dir="${ANIMA_REFLORA_MODEL_TE}"
  local src="${te_dir}/qwen_3_06b_base.safetensors"
  local dst="${te_dir}/model.safetensors"
  if [[ -f "${src}" && ! -e "${dst}" ]]; then
    ln -s "qwen_3_06b_base.safetensors" "${dst}"
  fi
}

auto_sync_storage() {
  if [[ "${AUTO_SYNC_S3:-0}" == "1" ]]; then
    sync_storage.sh pull "${S3_URI:-}" "${ANIMA_REFLORA_STORAGE}"
  fi
}

prepare_text_encoder_symlink

case "${MODE}" in
  jupyter)
    auto_sync_storage
    export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
    export JUPYTER_ROOT_DIR="${JUPYTER_ROOT_DIR:-/opt}"
    export JUPYTER_ALLOW_ORIGIN_PAT="${JUPYTER_ALLOW_ORIGIN_PAT:-https://.*[.]proxy[.]runpod[.]net}"
    mkdir -p "${JUPYTER_ROOT_DIR}"
    if [[ -n "${JUPYTER_PASSWORD:-}" ]]; then
      PASSWORD_HASH="$(python -c 'from jupyter_server.auth import passwd; import os; print(passwd(os.environ["JUPYTER_PASSWORD"]))')"
      exec jupyter lab \
        --ip=0.0.0.0 \
        --port="${JUPYTER_PORT}" \
        --no-browser \
        --allow-root \
        --ServerApp.root_dir="${JUPYTER_ROOT_DIR}" \
        --ServerApp.allow_origin_pat="${JUPYTER_ALLOW_ORIGIN_PAT}" \
        --ServerApp.password="${PASSWORD_HASH}" \
        "$@"
    else
      if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
        export JUPYTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
        echo "Generated JUPYTER_TOKEN=${JUPYTER_TOKEN}"
      fi
      exec jupyter lab \
        --ip=0.0.0.0 \
        --port="${JUPYTER_PORT}" \
        --no-browser \
        --allow-root \
        --ServerApp.root_dir="${JUPYTER_ROOT_DIR}" \
        --ServerApp.allow_origin_pat="${JUPYTER_ALLOW_ORIGIN_PAT}" \
        --ServerApp.token="${JUPYTER_TOKEN}" \
        "$@"
    fi
    ;;
  bash|shell)
    exec /bin/bash "$@"
    ;;
  train)
    auto_sync_storage
    # Optional fail-safe: on SIGTERM/SIGINT (e.g. RunPod preemption) push the latest
    # checkpoints of this run to S3 before teardown. Off by default (AUTO_PUSH_S3=1
    # to enable); results otherwise stay on the ephemeral OUT and are fetched manually.
    RUN_NAME=""
    _args=("$@")
    for ((i = 0; i < ${#_args[@]}; i++)); do
      case "${_args[i]}" in
        --run-name) RUN_NAME="${_args[i + 1]:-}" ;;
        --run-name=*) RUN_NAME="${_args[i]#--run-name=}" ;;
      esac
    done
    push_on_term() {
      if [[ "${AUTO_PUSH_S3:-0}" == "1" && -n "${RUN_NAME}" ]]; then
        echo "[entrypoint] signal received -> pushing run '${RUN_NAME}' to S3 ..."
        sync_storage.sh push-runs "${RUN_NAME}" || echo "  (push-runs failed)"
      fi
    }
    trap push_on_term TERM INT
    train_plan.sh "$@" &
    TRAIN_PID=$!
    wait "${TRAIN_PID}"
    ;;
  preflight)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.preflight "$@"
    ;;
  head-roi-cache)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.build_head_roi_cache "$@"
    ;;
  ccip-cache)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.build_ccip_ref_cache "$@"
    ;;
  training-cache)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.build_training_cache "$@"
    ;;
  head-cache)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.head_cache "$@"
    ;;
  head-ccip-cache)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.ccip_head_cache "$@"
    ;;
  ref-use-eval)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.ref_use_eval "$@"
    ;;
  ref-ab)
    auto_sync_storage
    exec "${PYTHON_BIN:-python}" -m anima_reflora.local_ref_ab_infer "$@"
    ;;
  extract-dataset)
    exec sync_storage.sh extract-dataset "$@"
    ;;
  pull-caches)
    exec sync_storage.sh pull-caches "$@"
    ;;
  prepare)
    # Full setup: sync storage, pull caches to local, extract dataset, symlink TE
    echo "=== prepare: syncing storage ==="
    auto_sync_storage
    echo "=== prepare: pulling caches to ${ANIMA_REFLORA_CACHE_DIR:-${ANIMA_REFLORA_STORAGE}/runs} ==="
    sync_storage.sh pull-caches || echo "  (pull-caches skipped or failed)"
    echo "=== prepare: extracting dataset ==="
    sync_storage.sh extract-dataset || echo "  (extract-dataset skipped or failed)"
    echo "=== prepare: ensuring text encoder symlink ==="
    prepare_text_encoder_symlink
    echo "=== prepare: done ==="
    ;;
  tensorboard)
    exec tensorboard --host 0.0.0.0 --port "${TENSORBOARD_PORT:-6006}" --logdir "${TENSORBOARD_LOGDIR:-${ANIMA_REFLORA_OUT}}" "$@"
    ;;
  *)
    exec "${MODE}" "$@"
    ;;
esac
