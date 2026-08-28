from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULTS = {
    "ANIMA_REFLORA_REPO": "/opt/AnimaRefLora",
    "ANIMA_REFLORA_SD_SCRIPTS": "/opt/AnimaRefLora/sd-scripts",
    "ANIMA_REFLORA_STORAGE": "/workspace/storage",
    "ANIMA_REFLORA_LATCACHE": "/workspace/storage/_latcache",
    "ANIMA_REFLORA_LATCACHE_512": "/workspace/storage/_latcache-512",
    "ANIMA_REFLORA_DATASET": "/workspace/storage/dataset",
    "ANIMA_REFLORA_PARQUET": "/workspace/storage/dataset/index.parquet",
    "ANIMA_REFLORA_IMAGES": "/workspace/storage/dataset/images",
    "ANIMA_REFLORA_VAL": "/workspace/storage/val/test",
    "ANIMA_REFLORA_VAL_TAGS": "/workspace/storage/val/txt_backup",
    "ANIMA_REFLORA_AFC": "/workspace/storage/8AFC",
    "ANIMA_REFLORA_MODEL_DIT": "/workspace/storage/anima_models/diffusion_models/anima-base-v1.0.safetensors",
    "ANIMA_REFLORA_MODEL_TE": "/workspace/storage/anima_models/text_encoders",
    "ANIMA_REFLORA_MODEL_VAE": "/workspace/storage/anima_models/vae/qwen_image_vae.safetensors",
    "ANIMA_REFLORA_OUT": "/workspace/Persistent/runs",
    # Caches are small but expensive to rebuild (CPU onnxruntime). Keep them on the
    # persistent /workspace mount so a new pod doesn't re-pull/re-build every time.
    # (OUT stays ephemeral by design — results are fetched manually before teardown.)
    "ANIMA_REFLORA_CCIP_EMB_CACHE": "/workspace/storage/runs/ccip_ref_emb_cache.pt",
    "ANIMA_REFLORA_HEAD_ROI_CACHE": "/workspace/storage/runs/head_roi_cache.pt",
}


@dataclass(frozen=True)
class AnimaPaths:
    repo: Path
    sd_scripts: Path
    storage: Path
    latcache: Path
    latcache_512: Path
    dataset: Path
    parquet: Path
    images: Path
    val: Path
    val_tags: Path
    afc: Path
    model_dit: Path
    model_te: Path
    model_vae: Path
    out: Path
    ccip_cache: Path
    head_roi_cache: Path

    @classmethod
    def from_env(cls, storage: str | None = None, out_dir: str | None = None) -> "AnimaPaths":
        env = {key: os.environ.get(key, default) for key, default in DEFAULTS.items()}
        if storage:
            env["ANIMA_REFLORA_STORAGE"] = storage
            env.setdefault("ANIMA_REFLORA_LATCACHE", str(Path(storage) / "_latcache"))
        if out_dir:
            env["ANIMA_REFLORA_OUT"] = out_dir

        storage_root = Path(env["ANIMA_REFLORA_STORAGE"])
        repo_root = Path(env["ANIMA_REFLORA_REPO"])

        def path_for(key: str, fallback: Path | None = None) -> Path:
            value = os.environ.get(key)
            if key == "ANIMA_REFLORA_STORAGE" and storage:
                return Path(storage)
            if key == "ANIMA_REFLORA_OUT" and out_dir:
                return Path(out_dir)
            if value:
                return Path(value)
            if fallback is not None:
                return fallback
            return Path(env[key])

        return cls(
            repo=path_for("ANIMA_REFLORA_REPO"),
            sd_scripts=path_for("ANIMA_REFLORA_SD_SCRIPTS"),
            storage=storage_root,
            latcache=path_for("ANIMA_REFLORA_LATCACHE", storage_root / "_latcache"),
            latcache_512=path_for("ANIMA_REFLORA_LATCACHE_512", storage_root / "_latcache-512"),
            dataset=path_for("ANIMA_REFLORA_DATASET", storage_root / "dataset"),
            parquet=path_for("ANIMA_REFLORA_PARQUET", storage_root / "dataset" / "index.parquet"),
            images=path_for("ANIMA_REFLORA_IMAGES", storage_root / "dataset" / "images"),
            val=path_for("ANIMA_REFLORA_VAL", storage_root / "val" / "test"),
            val_tags=path_for("ANIMA_REFLORA_VAL_TAGS", storage_root / "val" / "txt_backup"),
            afc=path_for("ANIMA_REFLORA_AFC", storage_root / "8AFC"),
            model_dit=path_for(
                "ANIMA_REFLORA_MODEL_DIT",
                storage_root / "anima_models" / "diffusion_models" / "anima-base-v1.0.safetensors",
            ),
            model_te=path_for("ANIMA_REFLORA_MODEL_TE", storage_root / "anima_models" / "text_encoders"),
            model_vae=path_for("ANIMA_REFLORA_MODEL_VAE", storage_root / "anima_models" / "vae" / "qwen_image_vae.safetensors"),
            out=path_for("ANIMA_REFLORA_OUT", repo_root / "runs"),
            ccip_cache=path_for("ANIMA_REFLORA_CCIP_EMB_CACHE", repo_root / "runs" / "ccip_ref_emb_cache.pt"),
            head_roi_cache=path_for("ANIMA_REFLORA_HEAD_ROI_CACHE", repo_root / "runs" / "head_roi_cache.pt"),
        )

    def ensure_text_encoder_symlink(self) -> bool:
        src = self.model_te / "qwen_3_06b_base.safetensors"
        dst = self.model_te / "model.safetensors"
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.name)
                return True
            except OSError:
                return False
        return dst.exists()


def run_dir(out_root: Path, run_name: str) -> Path:
    return out_root / "experiments" / run_name
