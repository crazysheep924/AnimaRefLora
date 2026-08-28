from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import TrainConfig
from .crepa import CrepaHiddenCapture
from .ref_conditioning import attach_ref_conditioner, build_ref_conditioner
from .rope_refpos import RefPosScheme, install_on_anima


def resolve_sd_scripts_path(config: TrainConfig) -> Path:
    path = Path(config.sd_scripts or config.paths().sd_scripts)
    # full checkout carries anima_train_network.py; the trimmed inference
    # subset shipped with the ComfyUI plugin only carries library/.
    markers = (path / "anima_train_network.py", path / "library" / "anima_utils.py")
    if not any(marker.exists() for marker in markers):
        raise FileNotFoundError(
            f"sd-scripts not found at {path}. Set ANIMA_REFLORA_SD_SCRIPTS or --sd-scripts "
            "to a kohya-ss/sd-scripts checkout (or the plugin's bundled sd-scripts subset)."
        )
    return path


def add_sd_scripts_to_path(config: TrainConfig) -> Path:
    path = resolve_sd_scripts_path(config)
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path


def parse_network_args(values: list[str]) -> dict[str, str]:
    args: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--network-arg must be KEY=VALUE, got: {value}")
        key, val = value.split("=", 1)
        args[key] = val
    return args


def default_network_module(network: str) -> str:
    lowered = network.lower()
    if lowered in {"lokr", "lycoris", "lycoris-lokr"}:
        return "networks.lokr"
    if lowered in {"lora", "standard-lora"}:
        return "networks.lora_anima"
    if lowered == "loha":
        return "networks.loha"
    raise ValueError(f"Unsupported sd-scripts network: {network}")


class SdScriptsAnimaWrapper(nn.Module):
    def __init__(self, config: TrainConfig, device: torch.device, dtype: torch.dtype, ccip_dim: int | None = None):
        super().__init__()
        add_sd_scripts_to_path(config)
        from library import anima_utils

        for path in [config.paths().model_dit]:
            if not path.exists():
                raise FileNotFoundError(f"Anima DiT checkpoint not found: {path}")

        loading_device = device if device.type == "cuda" else torch.device("cpu")
        self.dit = anima_utils.load_anima_model(
            device=device,
            dit_path=str(config.paths().model_dit),
            attn_mode=config.attn_mode,
            split_attn=config.split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dtype,
            fp8_scaled=False,
        )
        self.dit.requires_grad_(False)
        if config.grad_checkpoint:
            self.dit.enable_gradient_checkpointing()
        if config.rope_refpos:
            install_on_anima(
                self.dit,
                RefPosScheme.for_layout(config.rope_layout, config.frames, shift=config.rope_shift),
            )

        module_name = config.network_module or default_network_module(config.network)
        network_module = importlib.import_module(module_name)
        net_kwargs = parse_network_args([*config.network_args, *config.train_args])
        net_kwargs.setdefault("exclude_patterns", r"['.*(_modulation|_norm|_embedder|final_layer).*']")
        if config.network.lower() in {"lokr", "lycoris", "lycoris-lokr"}:
            # Align with the AnimaEditV1 LoKr recipe: full_matrix LoKr,
            # factor 4, conv dim/alpha 64/32. full_matrix is the key capacity knob —
            # without it the decomposition stays low-rank and the adapter differs from
            # the reference. (wrap-set must still be verified via GPU smoke.)
            net_kwargs.setdefault("factor", "4")
            net_kwargs.setdefault("full_matrix", "True")
            net_kwargs.setdefault("conv_dim", "64")
            net_kwargs.setdefault("conv_alpha", "32.0")
        self.network = network_module.create_network(
            1.0,
            config.network_dim,
            config.network_alpha,
            None,
            None,
            self.dit,
            neuron_dropout=None,
            **net_kwargs,
        )
        if self.network is None:
            raise RuntimeError(f"sd-scripts network module returned None: {module_name}")
        self.network.apply_to(None, self.dit, apply_text_encoder=False, apply_unet=True)
        self.network.requires_grad_(True)
        hidden_dim = int(
            getattr(self.dit, "hidden_size", None)
            or getattr(self.dit, "hidden_dim", None)
            or getattr(self.dit, "dim", None)
            or 2048
        )
        self.ref_conditioner = None
        if not config.no_ref_conditioner or config.cpm:
            self.ref_conditioner = build_ref_conditioner(
                config,
                hidden_dim=hidden_dim,
                identity_dim=ccip_dim if ccip_dim is not None else config.cpm_identity_dim,
            ).to(device=device, dtype=dtype)
            attach_ref_conditioner(self.dit, self.ref_conditioner)
        self.crepa_hidden_dim = hidden_dim
        self.crepa_capture = CrepaHiddenCapture(self.dit, config.crepa_block) if config.crepa else None
        self._assert_only_lora_trainable()

    # Modules that must stay frozen (mirrors the original config.FREEZE).
    FREEZE_NAMES = ("llm_adapter", "x_embedder", "t_embedder", "final_layer", "adaln_modulation")

    def _assert_only_lora_trainable(self) -> None:
        """Fail loud unless ONLY adapter (LoKr + ref_conditioner) params require grad."""
        lora_param_ids = {id(p) for p in self.network.parameters()}
        allowed_param_ids = set(lora_param_ids)
        if self.ref_conditioner is not None:
            allowed_param_ids |= {id(p) for p in self.ref_conditioner.parameters()}
        offenders = [
            name for name, p in self.dit.named_parameters()
            if p.requires_grad and id(p) not in allowed_param_ids
        ]
        if offenders:
            raise AssertionError(f"non-adapter base params require grad ({len(offenders)}): {offenders[:8]}")
        frozen_lora = [n for n, p in self.network.named_parameters() if not p.requires_grad]
        if frozen_lora:
            raise AssertionError(f"LoRA params NOT requiring grad: {frozen_lora[:8]}")
        if not lora_param_ids:
            raise AssertionError("no LoRA params found")
        for frozen_name in self.FREEZE_NAMES:
            for name, p in self.dit.named_parameters():
                if frozen_name in name and p.requires_grad and id(p) not in allowed_param_ids:
                    raise AssertionError(f"FREEZE module '{frozen_name}' has grad-enabled param: {name}")

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        caption_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
        config: TrainConfig | None = None,
    ) -> torch.Tensor:
        if caption_embeds is None:
            raise ValueError("sd-scripts Anima backend requires cached prompt_embeds")
        if batch is None:
            batch = {}
        x = x_B_C_T_H_W
        h_latent = x_B_C_T_H_W.shape[-2]
        w_latent = x_B_C_T_H_W.shape[-1]
        padding_mask = torch.zeros(
            x_B_C_T_H_W.shape[0],
            1,
            h_latent,
            w_latent,
            dtype=x_B_C_T_H_W.dtype,
            device=x_B_C_T_H_W.device,
        )
        t5_input_ids = batch.get("t5_input_ids")
        t5_attn_mask = batch.get("t5_attn_mask")
        if t5_input_ids is not None:
            t5_input_ids = t5_input_ids.to(x_B_C_T_H_W.device, dtype=torch.long)
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.to(x_B_C_T_H_W.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(x_B_C_T_H_W.device)
        timesteps = timesteps_B_T.to(device=x.device, dtype=x.dtype)
        identity = None
        if self.ref_conditioner is not None and config is not None and config.cpm:
            identity = batch.get("ccip_embeddings")
            valid = batch.get("cpm_valid", batch.get("ccip_valid"))
            if identity is not None and valid is not None:
                identity = identity.to(device=x.device, dtype=torch.float32).clone()
                identity[~valid.to(device=x.device).bool()] = 0
        context = self.ref_conditioner.identity_context(identity) if self.ref_conditioner is not None else None
        if context is None:
            return self.dit(
                x,
                timesteps,
                caption_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attention_mask,
            )
        with context:
            return self.dit(
                x,
                timesteps,
                caption_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attention_mask,
            )

    @property
    def crepa_hidden(self) -> torch.Tensor | None:
        return self.crepa_capture.hidden if self.crepa_capture is not None else None

    def clear_crepa_hidden(self) -> None:
        if self.crepa_capture is not None:
            self.crepa_capture.clear()

    def export_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.network.state_dict().items()}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor], strict: bool = False):
        if all(k.startswith("network.") for k in state):
            state = {k[len("network.") :]: v for k, v in state.items()}
        return self.network.load_state_dict(state, strict=strict)
