#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
# No default bucket: set S3_URI (or pass an s3-uri argument) explicitly.
DEFAULT_S3_URI="${S3_URI:-}"
DEFAULT_LOCAL="${ANIMA_REFLORA_STORAGE:-/workspace}"
DEFAULT_OUT="${ANIMA_REFLORA_OUT:-/opt/AnimaRefLora/runs}"
# Caches live on the persistent storage mount (see paths.py), NOT the ephemeral OUT.
DEFAULT_CACHE_DIR="${ANIMA_REFLORA_CACHE_DIR:-${DEFAULT_LOCAL}/runs}"
DEFAULT_DATASET="${ANIMA_REFLORA_DATASET:-${DEFAULT_LOCAL}/dataset}"
ENDPOINT="${S3_ENDPOINT_URL:-https://s3api-us-mo-1.runpod.io}"
REGION="${S3_REGION:-us-mo-1}"

aws_base() {
  aws --endpoint-url "${ENDPOINT}" --region "${REGION}" "$@"
}

usage() {
  cat <<'USAGE'
sync_storage.sh ls [s3-uri]
sync_storage.sh pull [s3-uri] [local-dir]
sync_storage.sh push [local-dir] [s3-uri]
sync_storage.sh push-runs [run-name-or-runs-dir] [s3-uri]
sync_storage.sh pull-caches [s3-uri]
sync_storage.sh extract-dataset [tar-path] [dest-dir]
USAGE
}

case "${ACTION}" in
  ls)
    S3_URI_ARG="${2:-${DEFAULT_S3_URI}}"
    aws_base s3 ls "${S3_URI_ARG}"
    ;;
  pull)
    S3_URI_ARG="${2:-${DEFAULT_S3_URI}}"
    LOCAL_DIR="${3:-${DEFAULT_LOCAL}}"
    mkdir -p "${LOCAL_DIR}"
    aws_base s3 sync "${S3_URI_ARG}" "${LOCAL_DIR}" --only-show-errors
    ;;
  push)
    LOCAL_DIR="${2:-${DEFAULT_LOCAL}}"
    S3_URI_ARG="${3:-${DEFAULT_S3_URI}}"
    aws_base s3 sync "${LOCAL_DIR}" "${S3_URI_ARG}" --only-show-errors
    ;;
  push-runs)
    RUN_ARG="${2:-}"
    S3_URI_ARG="${3:-${DEFAULT_S3_URI}}"
    if [[ -z "${RUN_ARG}" ]]; then
      echo "push-runs requires a run name or run directory" >&2
      exit 2
    fi
    if [[ -d "${RUN_ARG}" ]]; then
      RUN_DIR="${RUN_ARG}"
    elif [[ -d "${DEFAULT_OUT}/experiments/${RUN_ARG}" ]]; then
      RUN_DIR="${DEFAULT_OUT}/experiments/${RUN_ARG}"
    else
      echo "Run not found: ${RUN_ARG}" >&2
      exit 2
    fi
    RUN_NAME="$(basename "${RUN_DIR}")"
    DEST="${S3_URI_ARG%/}/runs/experiments/${RUN_NAME}/"
    aws_base s3 sync "${RUN_DIR}" "${DEST}" --only-show-errors
    ;;
  pull-caches)
    # Pull cache .pt files from S3 runs/ into the persistent cache dir (matches paths.py)
    S3_URI_ARG="${2:-${DEFAULT_S3_URI}}"
    S3_RUNS="${S3_URI_ARG%/}/runs/"
    mkdir -p "${DEFAULT_CACHE_DIR}"
    echo "Pulling caches from ${S3_RUNS} to ${DEFAULT_CACHE_DIR} ..."
    aws_base s3 sync "${S3_RUNS}" "${DEFAULT_CACHE_DIR}" \
      --exclude "*" \
      --include "*.pt" \
      --include "*.json" \
      --only-show-errors
    echo "Cache pull done. Contents:"
    ls -lh "${DEFAULT_CACHE_DIR}"/*.pt 2>/dev/null || echo "  (no .pt files found)"
    ;;
  extract-dataset)
    TAR_PATH="${2:-${DEFAULT_LOCAL}/dataset.tar}"
    DEST_DIR="${3:-${DEFAULT_DATASET}}"
    if [[ ! -f "${TAR_PATH}" ]]; then
      echo "dataset.tar not found: ${TAR_PATH}" >&2
      echo "Pull storage first: sync_storage.sh pull" >&2
      exit 2
    fi
    if [[ -d "${DEST_DIR}" && -f "${DEST_DIR}/index.parquet" ]]; then
      echo "Dataset already extracted at ${DEST_DIR}"
      echo "  $(find "${DEST_DIR}" -type f | wc -l) files"
      exit 0
    fi
    mkdir -p "${DEST_DIR}"
    echo "Extracting ${TAR_PATH} -> ${DEST_DIR} ..."
    tar xf "${TAR_PATH}" -C "${DEST_DIR}" --strip-components=0
    echo "Done. $(find "${DEST_DIR}" -type f | wc -l) files extracted."
    ;;
  *)
    usage
    exit 2
    ;;
esac
