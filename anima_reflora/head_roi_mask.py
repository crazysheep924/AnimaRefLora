from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .features import HeadRoiCache


def load_roi_cache(path: str | Path) -> HeadRoiCache:
    return HeadRoiCache(path)


def save_roi_cache(path: str | Path, masks: dict[str, torch.Tensor], meta: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # store binary masks as uint8 (4x smaller than float32); HeadRoiCache._parse casts
    # back to float on load, so this is transparent to consumers.
    torch.save({"masks": {str(k): (v.detach().cpu() > 0.5).to(torch.uint8) for k, v in masks.items()}, "meta": meta or {}}, path)
    return path


def build_head_weight_map(
    cache: HeadRoiCache | None,
    paths: list[str],
    latent_size: tuple[int, int],
    sigmas: torch.Tensor,
    *,
    weight: float,
    sigma_cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cache is None or weight == 1.0:
        return torch.ones(len(paths), 1, *latent_size, device=device, dtype=dtype)
    masks, valid = cache.gather(paths, latent_size[0], latent_size[1], device=device, dtype=dtype)
    masks = masks.float()
    sig = sigmas.to(device=device).float().view(-1, 1, 1, 1)
    if sigma_cutoff > 0:
        decay = (1.0 - sig / sigma_cutoff).clamp_min(0.0)
    else:
        decay = torch.ones_like(sig)
    w_eff = 1.0 + (float(weight) - 1.0) * decay
    valid = valid.float().view(-1, 1, 1, 1)
    weight_map = 1.0 + valid * (w_eff - 1.0) * masks
    mean = weight_map.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (weight_map / mean).to(dtype=dtype)


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.view(1, 1, *mask.shape)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="nearest")
    return mask.squeeze(0).squeeze(0).clamp(0, 1)


__all__ = ["build_head_weight_map", "load_roi_cache", "resize_mask", "save_roi_cache"]
