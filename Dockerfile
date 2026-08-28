# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG APP_HOME=/opt/AnimaRefLora
# RunPod network volumes mount at /workspace, so the volume root IS the storage root.
# Data lives directly at /workspace/_latcache, /workspace/runs, /workspace/anima_models, ...
ARG STORAGE_ROOT=/workspace
ARG PERSISTENT_ROOT=/workspace/Persistent
ARG SD_SCRIPTS_REPO=https://github.com/kohya-ss/sd-scripts.git
ARG SD_SCRIPTS_REF=c162e9039cc3228057b3fc6ae64ce770b87ab9bf
ARG PYTHONUNBUFFERED=1
ARG PIP_NO_CACHE_DIR=1
ARG S3_ENDPOINT_URL=https://s3api-us-ca-2.runpod.io
ARG S3_REGION=us-ca-2
ARG S3_URI=

ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
    PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR} \
    SD_SCRIPTS_COMMIT=${SD_SCRIPTS_REF} \
    ANIMA_REFLORA_REPO=${APP_HOME} \
    ANIMA_REFLORA_SD_SCRIPTS=${APP_HOME}/sd-scripts \
    ANIMA_REFLORA_STORAGE=${STORAGE_ROOT} \
    ANIMA_REFLORA_PERSISTENT=${PERSISTENT_ROOT} \
    ANIMA_REFLORA_LATCACHE=${STORAGE_ROOT}/_latcache \
    ANIMA_REFLORA_LATCACHE_512=${STORAGE_ROOT}/_latcache-512 \
    ANIMA_REFLORA_DATASET=${STORAGE_ROOT}/dataset \
    ANIMA_REFLORA_PARQUET=${STORAGE_ROOT}/dataset/index.parquet \
    ANIMA_REFLORA_IMAGES=${STORAGE_ROOT}/dataset/images \
    ANIMA_REFLORA_VAL=${STORAGE_ROOT}/val/test \
    ANIMA_REFLORA_VAL_TAGS=${STORAGE_ROOT}/val/txt_backup \
    ANIMA_REFLORA_AFC=${STORAGE_ROOT}/8AFC \
    ANIMA_REFLORA_MODEL_DIT=${STORAGE_ROOT}/anima_models/diffusion_models/anima-base-v1.0.safetensors \
    ANIMA_REFLORA_MODEL_TE=${STORAGE_ROOT}/anima_models/text_encoders \
    ANIMA_REFLORA_MODEL_VAE=${STORAGE_ROOT}/anima_models/vae/qwen_image_vae.safetensors \
    ANIMA_REFLORA_OUT=${PERSISTENT_ROOT}/runs \
    ANIMA_REFLORA_CCIP_EMB_CACHE=${STORAGE_ROOT}/runs/ccip_ref_emb_cache.pt \
    ANIMA_REFLORA_HEAD_ROI_CACHE=${STORAGE_ROOT}/runs/head_roi_cache.pt \
    JUPYTER_PORT=8888 \
    JUPYTER_ROOT_DIR=/opt \
    JUPYTER_STORAGE_LINK=/opt/storage \
    JUPYTER_PERSISTENT_LINK=/opt/Persistent \
    JUPYTER_ALLOW_ORIGIN_PAT=https://.*[.]proxy[.]runpod[.]net \
    TENSORBOARD_LOGDIR=${PERSISTENT_ROOT}/runs \
    S3_ENDPOINT_URL=${S3_ENDPOINT_URL} \
    S3_REGION=${S3_REGION} \
    S3_URI=${S3_URI}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        less \
        rsync \
        tini \
        tmux \
        vim \
        wget \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

COPY requirements.txt pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=false python -m pip install --upgrade pip setuptools wheel \
    && PIP_NO_CACHE_DIR=false python -m pip install -r requirements.txt \
    && git clone "${SD_SCRIPTS_REPO}" "${APP_HOME}/sd-scripts" \
    && cd "${APP_HOME}/sd-scripts" \
    && git checkout "${SD_SCRIPTS_REF}" \
    && python -m pip install -e . \
    && mkdir -p "${STORAGE_ROOT}" "${STORAGE_ROOT}/dataset" "${PERSISTENT_ROOT}/runs"

COPY anima_reflora ./anima_reflora
COPY scripts ./scripts

RUN python -m pip install -e . --no-deps \
    && chmod +x "${APP_HOME}"/scripts/*.sh \
    && ln -sfn "${APP_HOME}/scripts/entrypoint.sh" /usr/local/bin/anima-reflora-entrypoint \
    && ln -sfn "${APP_HOME}/scripts/train_plan.sh" /usr/local/bin/train_plan.sh \
    && ln -sfn "${APP_HOME}/scripts/sync_storage.sh" /usr/local/bin/sync_storage.sh

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/anima-reflora-entrypoint"]
CMD ["jupyter"]
