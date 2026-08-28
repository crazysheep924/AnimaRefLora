from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CrepaProjector(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int, hidden_dim: int | None = None, block_index: int = 8):
        super().__init__()
        hidden = int(hidden_dim or max(in_dim, embedding_dim))
        self.in_dim = int(in_dim)
        self.embedding_dim = int(embedding_dim)
        self.block_index = int(block_index)
        self.net = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.embedding_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


class CrepaHiddenCapture:
    def __init__(self, module: nn.Module, block_index: int):
        blocks = getattr(module, "blocks", None)
        if blocks is None:
            raise AttributeError("CREPA hidden capture requires a model with .blocks")
        if block_index < 0 or block_index >= len(blocks):
            raise ValueError(f"CREPA block index out of range for {len(blocks)} blocks: {block_index}")
        self.hidden: torch.Tensor | None = None
        self.handle = blocks[block_index].register_forward_hook(self._hook)

    def _hook(self, _module, _args, output) -> None:
        tensor = first_tensor(output)
        if tensor is not None:
            self.hidden = tensor

    def clear(self) -> None:
        self.hidden = None

    def close(self) -> None:
        self.handle.remove()


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def normalize_hidden(hidden: torch.Tensor, frames: int) -> torch.Tensor:
    """Return hidden as (B,T,H,W,D) when common Anima shapes allow it."""

    if hidden.ndim == 5:
        return hidden
    if hidden.ndim == 3:
        batch, tokens, dim = hidden.shape
        if tokens % frames:
            raise ValueError(f"Cannot split hidden tokens={tokens} into frames={frames}")
        spatial = tokens // frames
        side = int(spatial**0.5)
        if side * side != spatial:
            raise ValueError(f"Cannot infer square hidden grid from {spatial} tokens per frame")
        return hidden.view(batch, frames, side, side, dim)
    raise ValueError(f"Unsupported CREPA hidden shape: {tuple(hidden.shape)}")


def pool_target_hidden(hidden_b_t_h_w_d: torch.Tensor, head_mask: torch.Tensor | None = None, pool: str = "global") -> torch.Tensor:
    target = hidden_b_t_h_w_d[:, -1].float()
    if pool == "head_roi" and head_mask is not None and bool(head_mask.any()):
        mask = head_mask.float()
        if mask.shape[-2:] != target.shape[1:3]:
            mask = F.interpolate(mask, size=target.shape[1:3], mode="nearest")
        mask = mask.squeeze(1).unsqueeze(-1)
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return (target * mask).sum(dim=(1, 2)) / denom
    return target.mean(dim=(1, 2))


def crepa_hidden_loss(
    projector: CrepaProjector,
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
    valid: torch.Tensor,
    *,
    frames: int,
    sigmas: torch.Tensor | None = None,
    sigma_cutoff: float = 0.0,
    head_mask: torch.Tensor | None = None,
    pool: str = "global",
) -> tuple[torch.Tensor, dict[str, float]]:
    if not valid.any():
        return hidden.sum() * 0.0, {"crepa_valid_fraction": 0.0, "crepa_cosine": 0.0}
    normalized = normalize_hidden(hidden, frames=frames)
    pooled = pool_target_hidden(normalized, head_mask=head_mask, pool=pool)
    projected = F.normalize(projector(pooled), dim=-1)
    wanted = F.normalize(embeddings.to(device=projected.device, dtype=torch.float32), dim=-1)
    valid_mask = valid.to(device=projected.device)
    if sigmas is not None and sigma_cutoff > 0:
        valid_mask = valid_mask & (sigmas.to(device=projected.device) <= sigma_cutoff)
    if not valid_mask.any():
        return hidden.sum() * 0.0, {"crepa_valid_fraction": 0.0, "crepa_cosine": 0.0}
    cosine = F.cosine_similarity(projected[valid_mask], wanted[valid_mask], dim=-1)
    loss = 1.0 - cosine.mean()
    return loss, {
        "crepa_valid_fraction": float(valid_mask.float().mean().detach().cpu()),
        "crepa_cosine": float(cosine.mean().detach().cpu()),
        "crepa_pool_head_roi": 1.0 if pool == "head_roi" else 0.0,
    }


def crepa_state_path(lora_path: str | Path) -> Path:
    path = Path(lora_path)
    if path.name.startswith("lora_step_"):
        return path.with_name(path.name.replace("lora_step_", "crepa_projector_step_", 1))
    return path.with_name(f"{path.stem}.crepa_projector{path.suffix}")


__all__ = [
    "CrepaHiddenCapture",
    "CrepaProjector",
    "crepa_hidden_loss",
    "crepa_state_path",
    "first_tensor",
    "normalize_hidden",
    "pool_target_hidden",
]
