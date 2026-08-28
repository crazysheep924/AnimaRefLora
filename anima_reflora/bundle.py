"""Single-file checkpoint bundles for distribution/inference.

A bundle packs everything the ComfyUI loader (or local inference) needs into
ONE safetensors file, so users pick a single dropdown entry instead of
assembling five step-matched files:

  tensors:  lora.<key>, ref_conditioner.<key>, crepa_projector.<key>, ...
  metadata: format="animaref-bundle-v1", step, name,
            feature_config=<json>, rope_refpos=<json>, sidecars=<json list>

Plain safetensors — no zip, mmap-friendly, inspectable with standard tools.
Multi-file training checkpoints remain the training-side format; bundles are
produced from them by scripts/pack_animaref_bundle.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

BUNDLE_FORMAT = "animaref-bundle-v1"
BUNDLE_SUFFIX = ".animaref.safetensors"
# tensor-group prefixes: the main checkpoint plus per-module sidecars
MAIN_PREFIX = "lora"


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """Return the safetensors __metadata__ dict ({} if absent or not safetensors)."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as fh:
            return dict(fh.metadata() or {})
    except Exception:
        return {}


def is_bundle(path: str | Path) -> bool:
    path = Path(path)
    if not path.suffix == ".safetensors" or not path.exists():
        return False
    return read_safetensors_metadata(path).get("format") == BUNDLE_FORMAT


def load_bundle_group(path: str | Path, prefix: str) -> dict[str, torch.Tensor]:
    """Load one tensor group (e.g. 'lora', 'ref_conditioner') with the prefix stripped."""
    from safetensors import safe_open

    want = f"{prefix}."
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        for key in fh.keys():
            if key.startswith(want):
                out[key[len(want):]] = fh.get_tensor(key)
    return out


def bundle_sidecar_names(path: str | Path) -> list[str]:
    meta = read_safetensors_metadata(path)
    try:
        return list(json.loads(meta.get("sidecars", "[]")))
    except Exception:
        return []


def bundle_features(path: str | Path) -> dict[str, Any]:
    meta = read_safetensors_metadata(path)
    try:
        return dict(json.loads(meta.get("feature_config", "{}")))
    except Exception:
        return {}


def bundle_rope_payload(path: str | Path) -> dict[str, Any] | None:
    meta = read_safetensors_metadata(path)
    raw = meta.get("rope_refpos")
    if not raw:
        return None
    try:
        return dict(json.loads(raw))
    except Exception:
        return None


def pack_bundle(
    checkpoints_dir: str | Path,
    step: int,
    out_path: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Pack lora_step_<step> + all its sidecars + JSON configs into one bundle file.

    Discovers sidecars by globbing `*_step_<step>.safetensors` (optimizer state is
    training-only and deliberately excluded).
    """
    from safetensors.torch import load_file, save_file

    ckpt_dir = Path(checkpoints_dir)
    lora = ckpt_dir / f"lora_step_{step}.safetensors"
    if not lora.exists():
        raise FileNotFoundError(f"main checkpoint missing: {lora}")

    tensors: dict[str, torch.Tensor] = {}
    for key, value in load_file(str(lora), device="cpu").items():
        tensors[f"{MAIN_PREFIX}.{key}"] = value

    sidecars: list[str] = []
    for sidecar in sorted(ckpt_dir.glob(f"*_step_{step}.safetensors")):
        group = sidecar.name[: -len(f"_step_{step}.safetensors")]
        if group in {MAIN_PREFIX, "optimizer"}:
            continue
        sidecars.append(group)
        for key, value in load_file(str(sidecar), device="cpu").items():
            bundle_key = f"{group}.{key}"
            if bundle_key in tensors:
                raise ValueError(f"tensor key collision while packing: {bundle_key}")
            tensors[bundle_key] = value

    features_path = ckpt_dir / f"feature_config_step_{step}.json"
    if not features_path.exists():
        raise FileNotFoundError(f"feature config missing: {features_path}")
    features = json.loads(features_path.read_text(encoding="utf-8"))

    metadata: dict[str, str] = {
        "format": BUNDLE_FORMAT,
        "modelspec.architecture": "anima-reference-lora",
        "step": str(int(step)),
        "name": name or f"animaref_{step}",
        "sidecars": json.dumps(sidecars),
        "feature_config": json.dumps(features, sort_keys=True),
    }
    rope_path = ckpt_dir / f"rope_refpos_step_{step}.json"
    if rope_path.exists():
        metadata["rope_refpos"] = json.dumps(
            json.loads(rope_path.read_text(encoding="utf-8")), sort_keys=True
        )
    elif features.get("rope_refpos"):
        raise FileNotFoundError(f"features declare rope_refpos but sidecar missing: {rope_path}")

    # carry the trainer's original metadata (config snapshot etc.) under a prefix
    for key, value in read_safetensors_metadata(lora).items():
        if key not in metadata:
            metadata[f"orig.{key}"] = value

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    return out


def verify_bundle(checkpoints_dir: str | Path, step: int, bundle_path: str | Path) -> None:
    """Raise if the bundle does not reproduce the multi-file checkpoint bit-exactly."""
    from safetensors.torch import load_file

    ckpt_dir = Path(checkpoints_dir)
    groups = {MAIN_PREFIX: ckpt_dir / f"lora_step_{step}.safetensors"}
    for group in bundle_sidecar_names(bundle_path):
        groups[group] = ckpt_dir / f"{group}_step_{step}.safetensors"
    for group, source in groups.items():
        expected = load_file(str(source), device="cpu")
        got = load_bundle_group(bundle_path, group)
        if set(expected) != set(got):
            raise AssertionError(f"{group}: key sets differ")
        for key in expected:
            if not torch.equal(expected[key], got[key]):
                raise AssertionError(f"{group}.{key}: tensor mismatch")
    features = json.loads((ckpt_dir / f"feature_config_step_{step}.json").read_text(encoding="utf-8"))
    if bundle_features(bundle_path) != features:
        raise AssertionError("feature_config mismatch")
    rope_path = ckpt_dir / f"rope_refpos_step_{step}.json"
    if rope_path.exists():
        if bundle_rope_payload(bundle_path) != json.loads(rope_path.read_text(encoding="utf-8")):
            raise AssertionError("rope_refpos mismatch")
