from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class TokenCrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, inner_dim: int, num_heads: int) -> None:
        super().__init__()
        if inner_dim % num_heads:
            raise ValueError(f"inner_dim must be divisible by num_heads: {inner_dim}/{num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(inner_dim) // int(num_heads)
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-4)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        bsz, tokens, _ = tensor.shape
        return tensor.view(bsz, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.q_proj(self.query_norm(query)))
        k = self._heads(self.k_proj(self.context_norm(context)))
        v = self._heads(self.v_proj(self.context_norm(context)))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(out)


class IdentityPrototypeCPM(nn.Module):
    def __init__(self, identity_dim: int, num_tokens: int, token_dim: int, train_embeddings: bool = True) -> None:
        super().__init__()
        self.identity_dim = int(identity_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.train_embeddings = bool(train_embeddings)
        self.input_norm = nn.LayerNorm(identity_dim)
        self.token_proj = nn.Linear(identity_dim, num_tokens * token_dim)
        self.base_tokens = nn.Parameter(torch.zeros(num_tokens, token_dim))

    def forward(self, identity_emb: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if identity_emb.ndim != 2 or identity_emb.shape[-1] != self.identity_dim:
            raise ValueError(f"identity_emb must be (B,{self.identity_dim}), got {tuple(identity_emb.shape)}")
        emb = identity_emb if self.train_embeddings else identity_emb.detach()
        emb = F.normalize(emb.float(), dim=-1).to(device=device, dtype=self.input_norm.weight.dtype)
        tokens = self.token_proj(self.input_norm(emb)).view(identity_emb.shape[0], self.num_tokens, self.token_dim)
        return (tokens + self.base_tokens.unsqueeze(0)).to(dtype=dtype)


class AdapterGateBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int,
        num_heads: int,
        *,
        adapter_enabled: bool = True,
        cpm_enabled: bool = False,
        identity_dim: int = 768,
        cpm_tokens: int = 4,
        cpm_train_embeddings: bool = True,
        target_frame_idx: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.target_frame_idx = int(target_frame_idx)
        self.frame_adapter = (
            TokenCrossAttention(hidden_dim, hidden_dim, adapter_dim, num_heads) if adapter_enabled else None
        )
        self.adapter_gate = nn.Parameter(torch.tensor(1e-3)) if adapter_enabled else None
        self.cpm = (
            IdentityPrototypeCPM(identity_dim, cpm_tokens, adapter_dim, train_embeddings=cpm_train_embeddings)
            if cpm_enabled
            else None
        )
        self.cpm_adapter = (
            TokenCrossAttention(hidden_dim, adapter_dim, adapter_dim, num_heads) if cpm_enabled else None
        )
        self.cpm_gate = nn.Parameter(torch.tensor(1e-3)) if cpm_enabled else None

    def _target_index(self, frames: int) -> int:
        idx = self.target_frame_idx if self.target_frame_idx >= 0 else frames + self.target_frame_idx
        if idx < 0 or idx >= frames:
            raise ValueError(f"target_frame_idx={self.target_frame_idx} is out of range for {frames} frames")
        return idx

    def forward(self, x_b_t_h_w_d: torch.Tensor, identity_emb: torch.Tensor | None = None) -> torch.Tensor:
        if x_b_t_h_w_d.ndim != 5:
            raise ValueError(f"expected (B,T,H,W,D), got {tuple(x_b_t_h_w_d.shape)}")
        bsz, frames, height, width, dim = x_b_t_h_w_d.shape
        if frames < 2 or dim != self.hidden_dim:
            return x_b_t_h_w_d
        target_idx = self._target_index(frames)
        if target_idx == 0:
            return x_b_t_h_w_d
        target = x_b_t_h_w_d[:, target_idx].reshape(bsz, height * width, dim)
        updates: list[torch.Tensor] = []
        if self.frame_adapter is not None and self.adapter_gate is not None:
            refs = x_b_t_h_w_d[:, :target_idx].reshape(bsz, target_idx * height * width, dim)
            updates.append(self.adapter_gate.to(dtype=target.dtype) * self.frame_adapter(target, refs))
        if (
            self.cpm is not None
            and self.cpm_adapter is not None
            and self.cpm_gate is not None
            and identity_emb is not None
        ):
            proto = self.cpm(identity_emb, device=x_b_t_h_w_d.device, dtype=x_b_t_h_w_d.dtype)
            updates.append(self.cpm_gate.to(dtype=target.dtype) * self.cpm_adapter(target, proto))
        if not updates:
            return x_b_t_h_w_d
        changed = list(x_b_t_h_w_d.unbind(dim=1))
        changed[target_idx] = changed[target_idx] + torch.stack(updates).sum(dim=0).reshape(bsz, height, width, dim)
        return torch.stack(changed, dim=1)


class RefConditioner(nn.Module):
    def __init__(
        self,
        block_indices: Sequence[int],
        *,
        hidden_dim: int,
        adapter_dim: int,
        num_heads: int,
        adapter_enabled: bool = True,
        cpm_enabled: bool = False,
        identity_dim: int = 768,
        cpm_tokens: int = 4,
        cpm_train_embeddings: bool = True,
        target_frame_idx: int = -1,
    ) -> None:
        super().__init__()
        indices = tuple(sorted({int(index) for index in block_indices}))
        if not indices:
            raise ValueError("RefConditioner needs at least one block index")
        self.block_indices = indices
        self.blocks = nn.ModuleDict(
            {
                str(index): AdapterGateBlock(
                    hidden_dim,
                    adapter_dim,
                    num_heads,
                    adapter_enabled=adapter_enabled,
                    cpm_enabled=cpm_enabled,
                    identity_dim=identity_dim,
                    cpm_tokens=cpm_tokens,
                    cpm_train_embeddings=cpm_train_embeddings,
                    target_frame_idx=target_frame_idx,
                )
                for index in indices
            }
        )
        self._current_identity_emb: torch.Tensor | None = None

    @contextmanager
    def identity_context(self, identity_emb: torch.Tensor | None) -> Iterator[None]:
        previous = self._current_identity_emb
        self._current_identity_emb = identity_emb
        try:
            yield
        finally:
            self._current_identity_emb = previous

    def apply_after_block(self, block_idx: int, value):
        tensor = _first_tensor(value)
        if tensor is None:
            return value
        key = str(int(block_idx))
        if key not in self.blocks:
            return value
        updated = self.blocks[key](tensor, self._current_identity_emb)
        return _replace_first_tensor(value, updated)

    def scalar_stats(self) -> dict[str, float]:
        adapter = []
        cpm = []
        for block in self.blocks.values():
            if block.adapter_gate is not None:
                adapter.append(float(block.adapter_gate.detach().cpu()))
            if block.cpm_gate is not None:
                cpm.append(float(block.cpm_gate.detach().cpu()))
        out = {}
        if adapter:
            out["adapter_gate_mean"] = sum(adapter) / len(adapter)
        if cpm:
            out["cpm_gate_mean"] = sum(cpm) / len(cpm)
        return out


def _first_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _replace_first_tensor(value, replacement: torch.Tensor):
    if isinstance(value, torch.Tensor):
        return replacement
    if isinstance(value, tuple):
        items = list(value)
        for i, item in enumerate(items):
            if _first_tensor(item) is not None:
                items[i] = _replace_first_tensor(item, replacement)
                return tuple(items)
    if isinstance(value, list):
        for i, item in enumerate(value):
            if _first_tensor(item) is not None:
                value[i] = _replace_first_tensor(item, replacement)
                return value
    if isinstance(value, dict):
        copied = dict(value)
        for key, item in copied.items():
            if _first_tensor(item) is not None:
                copied[key] = _replace_first_tensor(item, replacement)
                return copied
    return value


def parse_block_indices(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(part) for part in value)


def build_ref_conditioner(config, *, hidden_dim: int = 2048, identity_dim: int | None = None) -> RefConditioner:
    return RefConditioner(
        parse_block_indices(config.adapter_blocks),
        hidden_dim=hidden_dim,
        adapter_dim=config.adapter_dim,
        num_heads=config.adapter_heads,
        adapter_enabled=(not config.no_ref_conditioner and not config.no_adapter_gate),
        cpm_enabled=bool(config.cpm),
        identity_dim=int(identity_dim or config.cpm_identity_dim),
        cpm_tokens=config.cpm_tokens,
        cpm_train_embeddings=not config.no_cpm_train_emb,
        target_frame_idx=-1,
    )


def attach_ref_conditioner(anima: nn.Module, conditioner: RefConditioner) -> RefConditioner:
    blocks = getattr(anima, "blocks", None)
    if blocks is None:
        raise AttributeError("Anima model has no .blocks ModuleList for ref conditioner")
    n_blocks = len(blocks)
    bad = [index for index in conditioner.block_indices if index < 0 or index >= n_blocks]
    if bad:
        raise ValueError(f"Ref conditioner block indices out of range for {n_blocks} blocks: {bad}")
    anima.ref_conditioner = conditioner
    for block_idx in conditioner.block_indices:
        block = blocks[block_idx]
        if getattr(block, "_anima_ref_conditioner_patched", False):
            continue
        original_forward = block.forward

        def patched_forward(*args, _orig=original_forward, _idx=block_idx, **kwargs):
            out = _orig(*args, **kwargs)
            active = getattr(anima, "ref_conditioner", None)
            return active.apply_after_block(_idx, out) if active is not None else out

        block._anima_ref_conditioner_original_forward = original_forward
        block._anima_ref_conditioner_patched = True
        block.forward = patched_forward
    return conditioner


def detach_ref_conditioner(anima: nn.Module) -> RefConditioner | None:
    conditioner = getattr(anima, "ref_conditioner", None)
    blocks = getattr(anima, "blocks", None)
    if blocks is not None:
        for block in blocks:
            if getattr(block, "_anima_ref_conditioner_patched", False):
                block.forward = block._anima_ref_conditioner_original_forward
                del block._anima_ref_conditioner_original_forward
                block._anima_ref_conditioner_patched = False
    if hasattr(anima, "ref_conditioner"):
        anima.ref_conditioner = None
    return conditioner


def ref_conditioner_state_path(lora_path: str | Path) -> Path:
    path = Path(lora_path)
    if path.name.startswith("lora_step_"):
        return path.with_name(path.name.replace("lora_step_", "ref_conditioner_step_", 1))
    return path.with_name(f"{path.stem}.ref_conditioner{path.suffix}")


__all__ = [
    "AdapterGateBlock",
    "IdentityPrototypeCPM",
    "RefConditioner",
    "TokenCrossAttention",
    "attach_ref_conditioner",
    "build_ref_conditioner",
    "detach_ref_conditioner",
    "parse_block_indices",
    "ref_conditioner_state_path",
]
