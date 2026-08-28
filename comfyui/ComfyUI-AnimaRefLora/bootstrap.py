from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import folder_paths
except Exception:  # pragma: no cover - outside ComfyUI
    folder_paths = None


ROOT = Path(__file__).resolve().parent


def _add_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _is_sd_scripts(path: Path) -> bool:
    # full checkout or the plugin's trimmed inference subset
    return (path / "anima_train_network.py").exists() or (path / "library" / "anima_utils.py").exists()


def bootstrap_paths() -> None:
    # An explicit env override wins only if it actually exists — stale or
    # container-baked values pointing at missing paths fall back to detection.
    env_repo = os.environ.get("ANIMA_REFLORA_REPO", "")
    repo_candidates = [
        Path(env_repo) if env_repo else None,
        ROOT,
        ROOT / "repo",
        ROOT.parent.parent,
    ]
    for repo in repo_candidates:
        if repo and (repo / "anima_reflora").exists():
            os.environ["ANIMA_REFLORA_REPO"] = str(repo)
            _add_path(repo)
            break

    env_sd = os.environ.get("ANIMA_REFLORA_SD_SCRIPTS", "")
    sd_candidates = [
        Path(env_sd) if env_sd else None,
        ROOT / "sd-scripts",
        ROOT / "repo" / "sd-scripts",
        Path(os.environ.get("ANIMA_REFLORA_REPO", "")) / "sd-scripts",
    ]
    for sd_scripts in sd_candidates:
        if sd_scripts and _is_sd_scripts(sd_scripts):
            os.environ["ANIMA_REFLORA_SD_SCRIPTS"] = str(sd_scripts)
            _add_path(sd_scripts)
            break


BUNDLE_FOLDER = "anima_reflora"
_BUNDLE_SUFFIX = ".safetensors"


def register_bundle_folder() -> None:
    """Register ComfyUI/models/anima_reflora/ as a model folder for bundles.

    The same directory doubles as the plugin's default storage root (runs/
    caches live under it), so listing filters those out.
    """
    if folder_paths is None:
        return
    path = Path(folder_paths.models_dir) / BUNDLE_FOLDER
    path.mkdir(parents=True, exist_ok=True)
    try:
        folder_paths.add_model_folder_path(BUNDLE_FOLDER, str(path))
    except Exception:
        pass


def bundle_names() -> list[str]:
    """Single-file .animaref bundles under models/anima_reflora/ (newest first)."""
    if folder_paths is None:
        return []
    try:
        names = folder_paths.get_filename_list(BUNDLE_FOLDER)
    except Exception:
        return []
    out = []
    for name in names:
        clean = name.replace("\\", "/")
        # skip the storage caches that share this directory
        if clean.startswith("runs/") or not clean.endswith(_BUNDLE_SUFFIX):
            continue
        out.append(name)
    return out


def checkpoint_names() -> list[str]:
    """Loader dropdown: bundles first; legacy multi-file loras as fallback."""
    bundles = bundle_names()
    if bundles:
        return bundles
    return lora_names()


def resolve_checkpoint(name: str) -> Path:
    if folder_paths is not None:
        try:
            full = folder_paths.get_full_path(BUNDLE_FOLDER, name)
        except Exception:
            full = None
        if full:
            return Path(full)
    return resolve_lora(name)


def lora_names() -> list[str]:
    if folder_paths is None:
        return ["lora_step_145000.safetensors"]
    names = folder_paths.get_filename_list("loras")
    return names or ["lora_step_145000.safetensors"]


def _names(kind: str, fallback: str) -> list[str]:
    if folder_paths is None:
        return [fallback]
    try:
        names = folder_paths.get_filename_list(kind)
    except Exception:
        names = []
    return names or [fallback]


def diffusion_model_names() -> list[str]:
    return _names("diffusion_models", "anima-base-v1.0.safetensors")


def text_encoder_names() -> list[str]:
    return _names("text_encoders", "model.safetensors")


def vae_names() -> list[str]:
    return _names("vae", "qwen_image_vae.safetensors")


def resolve_lora(checkpoint: str) -> Path:
    return resolve_model_path("loras", checkpoint)


def resolve_model_path(kind: str, name: str) -> Path:
    if name in {"", "none", "None", None}:
        return Path("")
    path = Path(name)
    if path.exists() or path.is_absolute():
        return path
    if folder_paths is not None:
        full = folder_paths.get_full_path(kind, name)
        if full:
            return Path(full)
    return path


def resolve_text_encoder_dir(name: str) -> Path:
    path = resolve_model_path("text_encoders", name)
    return path if path.is_dir() else path.parent


def default_storage() -> str:
    value = os.environ.get("ANIMA_REFLORA_STORAGE")
    if value:
        return value
    if folder_paths is not None:
        return str(Path(folder_paths.models_dir) / "anima_reflora")
    return str(ROOT / "models")
