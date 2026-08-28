from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import TrainConfig
from .features import CpmAdapter, RopeRefPositioner


class ReferenceConditioner(nn.Module):
    def __init__(self, channels: int = 16, adapter_gate: bool = True):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.tensor(0.0)) if adapter_gate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ref = x[:, :, :-1].mean(dim=2)
        cond = self.adapter(ref)
        if self.gate is not None:
            cond = cond * self.gate.tanh()
        return cond


class TinyAnimaRefModel(nn.Module):
    """Small trainable backend used for tests and mechanical smoke runs."""

    def __init__(
        self,
        channels: int = 16,
        caption_dim: int = 1024,
        use_ref_conditioner: bool = True,
        adapter_gate: bool = True,
        ccip_dim: int | None = None,
        cpm_train_embeddings: bool = True,
        rope_refpos: bool = False,
        frames: int = 3,
        rope_layout: str = "disjoint",
        rope_shift: float = 1.0,
    ):
        super().__init__()
        self.caption_proj = nn.Linear(caption_dim, channels)
        self.time_proj = nn.Sequential(nn.Linear(1, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.ref_conditioner = ReferenceConditioner(channels, adapter_gate=adapter_gate) if use_ref_conditioner else None
        self.cpm_adapter = CpmAdapter(ccip_dim, channels=channels, train_embeddings=cpm_train_embeddings) if ccip_dim is not None else None
        self.rope_refpos = RopeRefPositioner(frames=frames, layout=rope_layout, shift=rope_shift) if rope_refpos else None
        self.net = nn.Sequential(
            nn.Conv3d(channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        caption_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
        config: TrainConfig | None = None,
    ) -> torch.Tensor:
        x = x_B_C_T_H_W
        if self.rope_refpos is not None:
            x = self.rope_refpos(x)
        if self.cpm_adapter is not None and batch is not None:
            x = self.cpm_adapter(x, batch.get("ccip_embeddings"), batch.get("cpm_valid", batch.get("ccip_valid")))
        if caption_embeds is not None:
            pooled = caption_embeds.float().mean(dim=1)
            cap = self.caption_proj(pooled).to(dtype=x.dtype).view(x.shape[0], x.shape[1], 1, 1, 1)
            x = x + cap
        time = self.time_proj(timesteps_B_T.float().unsqueeze(-1)).to(dtype=x.dtype)
        x = x + time.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        pred = self.net(x)
        if self.ref_conditioner is not None:
            cond = self.ref_conditioner(x).unsqueeze(2)
            pred[:, :, -1:] = pred[:, :, -1:] + cond
        return pred


class ExternalAnimaBackend(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        if not config.model_factory:
            raise RuntimeError(
                "MODEL_BACKEND=external requires ANIMA_REFLORA_MODEL_FACTORY or --model-factory. "
                "Provide a module:function factory that returns an nn.Module accepting "
                "(x_B_C_T_H_W, timesteps_B_T, caption_embeds, attention_mask, batch, config)."
            )
        module_name, sep, function_name = config.model_factory.partition(":")
        if not sep:
            raise ValueError("--model-factory must use module:function format")
        module = importlib.import_module(module_name)
        factory = getattr(module, function_name)
        model = factory(config)
        if not isinstance(model, nn.Module):
            raise TypeError("Model factory must return torch.nn.Module")
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(*args, **kwargs)


def build_model(config: TrainConfig, ccip_dim: int | None = None) -> nn.Module:
    if config.backend == "tiny":
        return TinyAnimaRefModel(
            use_ref_conditioner=not config.no_ref_conditioner,
            adapter_gate=not config.no_adapter_gate,
            ccip_dim=ccip_dim if config.cpm else None,
            cpm_train_embeddings=not config.no_cpm_train_emb,
            rope_refpos=config.rope_refpos,
            frames=config.frames,
            rope_layout=config.rope_layout,
            rope_shift=config.rope_shift,
        )
    if not config.model_factory:
        return _build_sd_scripts_model(config, ccip_dim=ccip_dim)
    return ExternalAnimaBackend(config)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "export_trainable_state_dict"):
        return model.export_trainable_state_dict()
    params = dict(model.named_parameters())
    state = {}
    for name, tensor in model.state_dict().items():
        if name in params and params[name].requires_grad:
            state[name] = tensor.detach().cpu()
    if not state:
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return state


def sidecar_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = {}
    for attr, name in [
        ("ref_conditioner", "ref_conditioner"),
        ("cpm_adapter", "cpm_adapter"),
        ("rope_refpos", "rope_refpos"),
    ]:
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            modules[name] = module
    inner = getattr(model, "model", None)
    for attr, name in [
        ("ref_conditioner", "ref_conditioner"),
        ("cpm_adapter", "cpm_adapter"),
        ("rope_refpos", "rope_refpos"),
    ]:
        module = getattr(inner, attr, None)
        if isinstance(module, nn.Module):
            modules[name] = module
    return modules


def _build_sd_scripts_model(config: TrainConfig, ccip_dim: int | None = None) -> nn.Module:
    from .sd_scripts_bridge import SdScriptsAnimaWrapper

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    if config.dtype == "fp32" or device.type == "cpu":
        dtype = torch.float32
    elif config.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    return SdScriptsAnimaWrapper(config, device=device, dtype=dtype, ccip_dim=ccip_dim)
