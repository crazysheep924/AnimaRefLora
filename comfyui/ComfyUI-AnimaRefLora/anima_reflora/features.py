from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import TrainConfig


def _path_keys(path: str) -> list[str]:
    p = Path(path)
    keys = [path, str(p), p.as_posix(), p.name]
    stem = p.stem
    if stem:
        keys.append(stem)
    return list(dict.fromkeys(keys))


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().float().flatten()
    if isinstance(value, dict):
        for key in ("embedding", "emb", "ccip", "ccip_emb", "identity", "prototype"):
            tensor = _first_tensor(value.get(key))
            if tensor is not None:
                return tensor
    return None


def _mask_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    if isinstance(value, dict):
        for key in ("mask", "head_mask", "roi", "head_roi"):
            tensor = _mask_tensor(value.get(key))
            if tensor is not None:
                return tensor
    return None


def _bbox_mask(value: Any, latent_size: Any) -> torch.Tensor | None:
    if not isinstance(latent_size, (list, tuple)) or len(latent_size) != 2:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        lat_h, lat_w = int(latent_size[0]), int(latent_size[1])
        y0, y1, x0, x1 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if lat_h <= 0 or lat_w <= 0:
        return None
    y0 = max(0, min(lat_h, y0))
    y1 = max(0, min(lat_h, y1))
    x0 = max(0, min(lat_w, x0))
    x1 = max(0, min(lat_w, x1))
    if y1 <= y0 or x1 <= x0:
        return None
    mask = torch.zeros(lat_h, lat_w, dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def _path_from_record(value: dict[str, Any]) -> str | None:
    for key in ("ref_path", "reference_path", "path", "image_path", "filename", "file"):
        found = value.get(key)
        if found:
            return str(found)
    return None


class CcipEmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        obj = torch.load(self.path, map_location="cpu", weights_only=False)
        self.embeddings = self._parse(obj)
        if not self.embeddings:
            raise ValueError(f"CCIP cache has no usable path->embedding entries: {self.path}")
        dims = {int(t.numel()) for t in self.embeddings.values()}
        if len(dims) != 1:
            raise ValueError(f"CCIP cache has mixed embedding dimensions: {sorted(dims)}")
        self.dim = dims.pop()

    @staticmethod
    def _parse(obj: Any) -> dict[str, torch.Tensor]:
        entries: dict[str, torch.Tensor] = {}
        if isinstance(obj, dict):
            if isinstance(obj.get("path_to_emb"), dict):
                for path, value in obj["path_to_emb"].items():
                    tensor = _first_tensor(value)
                    if tensor is not None:
                        entries[str(path)] = tensor
            if "paths" in obj and "embeddings" in obj:
                paths = list(obj["paths"])
                tensors = obj["embeddings"]
                if isinstance(tensors, torch.Tensor):
                    for path, tensor in zip(paths, tensors):
                        entries[str(path)] = tensor.detach().float().flatten()
            for path, value in obj.items():
                tensor = _first_tensor(value)
                if tensor is not None:
                    entries[str(path)] = tensor
        elif isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                path = _path_from_record(item)
                tensor = _first_tensor(item)
                if path and tensor is not None:
                    entries[path] = tensor
        return entries

    def lookup(self, path: str) -> torch.Tensor | None:
        for key in _path_keys(path):
            if key in self.embeddings:
                return self.embeddings[key]
        return None

    def gather(self, paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        values = []
        valid = []
        for path in paths:
            tensor = self.lookup(path)
            if tensor is None:
                values.append(torch.zeros(self.dim, dtype=torch.float32))
                valid.append(False)
            else:
                values.append(tensor.float())
                valid.append(True)
        return torch.stack(values, dim=0), torch.tensor(valid, dtype=torch.bool)

    def coverage(self, paths: list[str]) -> dict[str, Any]:
        missing = [path for path in paths if self.lookup(path) is None]
        total = len(paths)
        return {
            "cache": str(self.path),
            "entries": len(self.embeddings),
            "embedding_dim": self.dim,
            "requested": total,
            "covered": total - len(missing),
            "missing": len(missing),
            "missing_examples": missing[:10],
        }


class HeadRoiCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        obj = torch.load(self.path, map_location="cpu", weights_only=False)
        self.masks = self._parse(obj)
        if not self.masks:
            raise ValueError(f"Head ROI cache has no usable path->mask entries: {self.path}")

    @staticmethod
    def _parse(obj: Any) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        if isinstance(obj, dict):
            source = obj.get("masks", obj)
            if isinstance(source, dict):
                for key, value in source.items():
                    path = key[0] if isinstance(key, tuple) and key else key
                    latent_size = key[1] if isinstance(key, tuple) and len(key) > 1 else None
                    tensor = _mask_tensor(value)
                    if tensor is None:
                        tensor = _bbox_mask(value, latent_size)
                    if tensor is not None:
                        masks[str(path)] = tensor.float()
        elif isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                path = _path_from_record(item)
                tensor = _mask_tensor(item.get("mask", item))
                if path and tensor is not None:
                    masks[path] = tensor.float()
        return masks

    def lookup(self, path: str) -> torch.Tensor | None:
        for key in _path_keys(path):
            if key in self.masks:
                mask = self.masks[key]
                while mask.ndim > 2:
                    mask = mask.squeeze(0)
                return mask.clamp(0.0, 1.0)
        return None

    def gather(self, paths: list[str], height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        masks = []
        valid = []
        for path in paths:
            mask = self.lookup(path)
            if mask is None:
                masks.append(torch.zeros(1, height, width, dtype=torch.float32))
                valid.append(False)
                continue
            mask = mask.view(1, *mask.shape[-2:])
            if mask.shape[-2:] != (height, width):
                mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)
            masks.append(mask.float())
            valid.append(bool(mask.sum().item() > 0))
        return torch.stack(masks, dim=0).to(device=device, dtype=dtype), torch.tensor(valid, device=device, dtype=torch.bool)

    def coverage(self, paths: list[str]) -> dict[str, Any]:
        missing = [path for path in paths if self.lookup(path) is None]
        total = len(paths)
        return {
            "cache": str(self.path),
            "entries": len(self.masks),
            "requested": total,
            "covered": total - len(missing),
            "missing": len(missing),
            "missing_examples": missing[:10],
        }


class CpmAdapter(nn.Module):
    def __init__(self, embedding_dim: int, channels: int = 16, train_embeddings: bool = True):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, channels)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.train_embeddings = train_embeddings

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor | None, valid: torch.Tensor | None) -> torch.Tensor:
        if embeddings is None or valid is None or not valid.any():
            return x
        if not self.train_embeddings:
            embeddings = embeddings.detach()
        proj_dtype = self.proj.weight.dtype
        bias = self.proj(embeddings.to(device=x.device, dtype=proj_dtype)).to(dtype=x.dtype)
        bias = bias * valid.to(device=x.device, dtype=x.dtype).view(-1, 1)
        out = x.clone()
        out[:, :, -1] = out[:, :, -1] + self.gate.tanh() * bias.view(x.shape[0], x.shape[1], 1, 1)
        return out


class CrepaProjector(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features.float())


def pooled_target_features(prediction: torch.Tensor, head_mask: torch.Tensor | None = None, pool: str = "global") -> torch.Tensor:
    target = prediction[:, :, -1].float()
    if pool == "head_roi" and head_mask is not None and bool(head_mask.any()):
        mask = head_mask.float()
        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask, size=target.shape[-2:], mode="nearest")
        denom = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        return (target * mask).sum(dim=(-2, -1)) / denom
    return target.mean(dim=(-2, -1))


def crepa_loss(
    projector: CrepaProjector,
    prediction: torch.Tensor,
    embeddings: torch.Tensor,
    valid: torch.Tensor,
    head_mask: torch.Tensor | None,
    pool: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not valid.any():
        zero = prediction.sum() * 0.0
        return zero, {"crepa_valid_fraction": 0.0, "crepa_cosine": 0.0}
    features = pooled_target_features(prediction, head_mask=head_mask, pool=pool)
    projected = projector(features)
    wanted = embeddings.to(device=projected.device, dtype=torch.float32)
    valid_f = valid.to(device=projected.device)
    cosine = F.cosine_similarity(projected[valid_f], wanted[valid_f], dim=-1)
    loss = 1.0 - cosine.mean()
    return loss, {
        "crepa_valid_fraction": float(valid_f.float().mean().detach().cpu()),
        "crepa_cosine": float(cosine.mean().detach().cpu()),
    }


class RopeRefPositioner(nn.Module):
    def __init__(self, frames: int, layout: str, shift: float):
        super().__init__()
        if layout not in {"disjoint", "shifted", "packed"}:
            raise ValueError(f"Unsupported RoPE layout: {layout}")
        positions = torch.arange(frames, dtype=torch.float32)
        if frames > 1:
            if layout == "disjoint":
                positions[:-1] -= float(shift)
            elif layout == "shifted":
                positions[:-1] += float(shift)
            elif layout == "packed":
                positions = positions * (1.0 + float(shift))
        self.layout = layout
        self.shift = float(shift)
        self.register_buffer("frame_positions", positions, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.frame_positions.numel():
            return x
        marker = self.frame_positions.to(device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        return x + marker * x.detach().abs().mean().clamp_min(1e-6) * 1e-4

    def sidecar(self) -> dict[str, Any]:
        return {"enabled": True, "layout": self.layout, "shift": self.shift, "frame_positions": self.frame_positions.cpu().tolist()}


@dataclass(frozen=True)
class FeatureConfigSnapshot:
    frames: int
    rope_refpos: bool
    rope_layout: str
    rope_shift: float
    cpm: bool
    crepa: bool
    crepa_pool: str
    head_loss_weight: float


def feature_snapshot(config: TrainConfig) -> FeatureConfigSnapshot:
    return FeatureConfigSnapshot(
        frames=config.frames,
        rope_refpos=config.rope_refpos,
        rope_layout=config.rope_layout,
        rope_shift=config.rope_shift,
        cpm=config.cpm,
        crepa=config.crepa,
        crepa_pool=config.crepa_pool,
        head_loss_weight=config.head_loss_weight,
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def feature_sidecar_path(checkpoint: str | Path, name: str = "feature_config") -> Path:
    path = Path(checkpoint)
    stem = path.stem
    if stem.startswith("lora_"):
        stem = stem[len("lora_") :]
    return path.with_name(f"{name}_{stem}.json")


def check_resume_feature_compatibility(config: TrainConfig) -> None:
    if not config.resume:
        return
    sidecar = feature_sidecar_path(config.resume)
    if not sidecar.exists():
        raise FileNotFoundError(f"Required feature sidecar missing for resume: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as fh:
        saved = json.load(fh)
    current = asdict(feature_snapshot(config))
    for key in ("frames", "rope_refpos", "rope_layout", "rope_shift"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Resume feature mismatch for {key}: checkpoint={saved.get(key)!r} current={current.get(key)!r}")
