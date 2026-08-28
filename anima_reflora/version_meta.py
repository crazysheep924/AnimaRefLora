"""Collect runtime version metadata for preflight reports and training runs."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def _pkg_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def collect_runtime_versions(sd_scripts_path: str | None = None) -> dict:
    import torch

    sd_commit = os.environ.get("SD_SCRIPTS_COMMIT")
    if not sd_commit and sd_scripts_path:
        sd_commit = _read_git_head(Path(sd_scripts_path))

    cuda_device = None
    try:
        if torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": cuda_device,
        "sd_scripts_path": sd_scripts_path,
        "sd_scripts_commit": sd_commit,
        "lycoris": _pkg_version("lycoris-lora"),
        "safetensors": _pkg_version("safetensors"),
        "transformers": _pkg_version("transformers"),
        "dghs_imgutils": _pkg_version("dghs-imgutils"),
        "anima_reflora": _pkg_version("anima-reflora"),
        "docker_image_tag": os.environ.get("ANIMA_REFLORA_IMAGE_TAG"),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
    }
