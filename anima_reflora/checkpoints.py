from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn

from .features import write_json
from .models import sidecar_modules, trainable_state_dict
from .rope_refpos import RefPosScheme, save_refpos_scheme


def parse_step(path: str | Path) -> int | None:
    match = re.search(r"step_(\d+)", str(path))
    return int(match.group(1)) if match else None


def save_tensor_file(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(tensors, str(path), metadata=metadata)
    except Exception:
        torch.save({"state_dict": tensors, "metadata": metadata or {}}, path)


def load_tensor_file(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        return obj


def save_checkpoint(model: nn.Module, run_dir: Path, step: int, config_dict: dict, extra_sidecars: dict[str, nn.Module] | None = None) -> Path:
    ckpt = run_dir / "checkpoints" / f"lora_step_{step}.safetensors"
    features = config_dict.get("_feature_snapshot", {})
    metadata = {
        "format": "pt",
        "modelspec.architecture": "anima-reference-lora",
        "anima_reflora_step": str(step),
        "anima_reflora_config": json.dumps({k: str(v) for k, v in config_dict.items()})[:60000],
        "anima_reflora_features": json.dumps(features)[:60000],
    }
    save_tensor_file(trainable_state_dict(model), ckpt, metadata=metadata)
    sidecars = sidecar_modules(model)
    if extra_sidecars:
        sidecars.update(extra_sidecars)
    for name, module in sidecars.items():
        sidecar = run_dir / "checkpoints" / f"{name}_step_{step}.safetensors"
        save_tensor_file({k: v.detach().cpu() for k, v in module.state_dict().items()}, sidecar, metadata=metadata)
    if features:
        write_json(run_dir / "checkpoints" / f"feature_config_step_{step}.json", features)
        if features.get("rope_refpos"):
            save_refpos_scheme(
                RefPosScheme.for_layout(
                    str(features.get("rope_layout", "disjoint")),
                    int(features.get("frames", 3)),
                    shift=float(features.get("rope_shift", 1.0)),
                ),
                run_dir / "checkpoints" / f"rope_refpos_step_{step}.json",
                frames=int(features.get("frames", 3)),
            )
    return ckpt


def optimizer_state_path(checkpoint_path: str | Path) -> Path:
    step = parse_step(checkpoint_path)
    if step is None:
        raise ValueError(f"Cannot derive optimizer step from checkpoint path: {checkpoint_path}")
    return Path(checkpoint_path).with_name(f"optimizer_step_{step}.pt")


def save_optimizer_state(optimizer: torch.optim.Optimizer, checkpoint_path: str | Path) -> Path:
    path = optimizer_state_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"optimizer": optimizer.state_dict()}, path)
    return path


def load_optimizer_state(optimizer: torch.optim.Optimizer, checkpoint_path: str | Path, device: torch.device) -> Path:
    path = optimizer_state_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Required optimizer state missing for resume: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(obj["optimizer"] if isinstance(obj, dict) and "optimizer" in obj else obj)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)
    return path


def load_checkpoint_into(model: nn.Module, path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    from .bundle import MAIN_PREFIX, is_bundle, load_bundle_group

    state = load_bundle_group(path, MAIN_PREFIX) if is_bundle(path) else load_tensor_file(path)
    if hasattr(model, "load_trainable_state_dict"):
        result = model.load_trainable_state_dict(state, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
    result = model.load_state_dict(state, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_sidecar_into(module: nn.Module, checkpoint_path: str | Path, name: str, strict: bool = True) -> tuple[list[str], list[str]]:
    from .bundle import is_bundle, load_bundle_group

    if is_bundle(checkpoint_path):
        state = load_bundle_group(checkpoint_path, name)
        if not state:
            raise FileNotFoundError(f"Bundle {checkpoint_path} carries no '{name}' tensor group")
        result = module.load_state_dict(state, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
    step = parse_step(checkpoint_path)
    if step is None:
        raise ValueError(f"Cannot derive sidecar step from checkpoint path: {checkpoint_path}")
    sidecar = Path(checkpoint_path).with_name(f"{name}_step_{step}.safetensors")
    if not sidecar.exists():
        raise FileNotFoundError(f"Required sidecar missing for resume: {sidecar}")
    result = module.load_state_dict(load_tensor_file(sidecar), strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
