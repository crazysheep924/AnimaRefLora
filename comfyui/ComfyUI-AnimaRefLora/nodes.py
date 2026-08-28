"""ComfyUI nodes for AnimaRefLora reference-conditioned generation.

Wraps the validated anima_reflora inference pipeline (local_ref_ab_infer) —
the exact code path used for all training evals — instead of re-implementing
the frame stacking / RoPE refpos / CPM conditioning in Comfy primitives.

Node graph:
    AnimaExtraLora (optional) -> AnimaRefLoraLoader ──┐
                                                      ├── AnimaRefEncode ──┐
                                   (ref IMAGE) ───────┘                    ├── AnimaRefLoraSampler ── IMAGE
                                   (prompt / seed / steps ...) ───────────┘

The package bootstraps a bundled anima_reflora/ and sd-scripts/ directory when
they are placed next to this file, so distributable installs do not need
machine-specific environment variables.
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from .bootstrap import (
    bootstrap_paths,
    checkpoint_names,
    default_storage,
    diffusion_model_names,
    lora_names,
    register_bundle_folder,
    resolve_checkpoint,
    resolve_lora,
    resolve_model_path,
    resolve_text_encoder_dir,
    text_encoder_names,
    vae_names,
)

bootstrap_paths()
register_bundle_folder()

from anima_reflora.build_training_cache import SdScriptsPromptEncoder, load_vae
from anima_reflora.checkpoints import load_checkpoint_into, load_sidecar_into, load_tensor_file
from anima_reflora.config import ANIMA_NEGATIVE_PROMPT
from anima_reflora.local_ref_ab_infer import (
    apply_ref_frame_mode,
    build_caption,
    compute_ccip_head_embedding,
    config_for_infer,
    decode_latent,
    encode_ref_pair,
    load_features,
    sample_target,
    QUALITY_PREFIX,
)
from anima_reflora.models import build_model, sidecar_modules
from anima_reflora.rope_refpos import assert_sidecar_applied, maybe_apply_sidecar

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_MODEL_CACHE: dict[tuple, dict] = {}


def _apply_extra_lora(model, config, path: Path, strength: float, device, dtype):
    """Attach one standard Anima DiT LoRA to the standalone model."""
    from anima_reflora.sd_scripts_bridge import add_sd_scripts_to_path

    add_sd_scripts_to_path(config)
    from networks import lora_anima

    state = load_tensor_file(path)
    dit_state = {k: v for k, v in state.items() if k.startswith("lora_unet_")}
    if not dit_state or not any(".lora_down.weight" in k for k in dit_state):
        raise ValueError(
            f"Unsupported extra LoRA format: {path.name}. Expected standard Anima "
            "lora_unet_*.lora_down/up.weight keys."
        )
    for key, value in list(dit_state.items()):
        if key.endswith(".lora_down.weight"):
            alpha_key = key.removesuffix(".lora_down.weight") + ".alpha"
            dit_state.setdefault(alpha_key, torch.tensor(value.shape[0]))

    dit = getattr(model, "dit", None)
    if dit is None:
        raise TypeError("Extra LoRA requires the sd-scripts Anima backend")
    network, weights = lora_anima.create_network_from_weights(
        strength, str(path), None, None, dit,
        weights_sd=dit_state, for_inference=True,
    )
    if not network.unet_loras:
        raise ValueError(f"Extra LoRA has no weights matching this Anima model: {path.name}")
    network.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
    result = network.load_state_dict(weights, strict=False)
    missing_weights = [key for key in result.missing_keys if key.endswith(".weight")]
    if result.unexpected_keys or missing_weights:
        raise ValueError(
            "Extra LoRA does not match this Anima model: "
            f"unexpected={result.unexpected_keys[:3]}, missing={missing_weights[:3]}"
        )
    network.to(device=device, dtype=dtype).requires_grad_(False).eval()
    model.extra_lora = network


def _comfy_to_pil(image: torch.Tensor) -> Image.Image:
    """Comfy IMAGE (B,H,W,C float 0..1) -> PIL RGB of the first batch item."""
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_comfy(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class AnimaRefLoraLoader:
    """Load Anima base + LoKr checkpoint + sidecars (ref_conditioner/CPM/RoPE)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # single-file .animaref bundles from models/anima_reflora/;
                # falls back to legacy multi-file loras/ checkpoints if none exist
                "checkpoint": (checkpoint_names(),),
                "base_model": (diffusion_model_names(),),
                "text_encoder": (text_encoder_names(),),
                "vae": (vae_names(),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "vae_chunk_size": ("INT", {"default": 8, "min": 0, "max": 64}),
            },
            "optional": {"extra_lora": ("ANIMA_EXTRA_LORA",)},
        }

    RETURN_TYPES = ("ANIMA_MODEL",)
    FUNCTION = "load"
    CATEGORY = "AnimaRefLora"

    def load(self, checkpoint, base_model, text_encoder, vae, device, dtype, vae_chunk_size,
             extra_lora=None):
        extra_key = tuple(extra_lora) if extra_lora else None
        key = (checkpoint, base_model, text_encoder, vae, device, dtype, vae_chunk_size, extra_key)
        if key in _MODEL_CACHE:
            return (_MODEL_CACHE[key],)
        _MODEL_CACHE.clear()  # hold at most one bundle (VRAM)

        ckpt = resolve_checkpoint(checkpoint)
        base_path = resolve_model_path("diffusion_models", base_model)
        text_path = resolve_text_encoder_dir(text_encoder)
        vae_path = resolve_model_path("vae", vae)
        storage = default_storage()
        os.environ["ANIMA_REFLORA_MODEL_DIT"] = str(base_path)
        os.environ["ANIMA_REFLORA_MODEL_TE"] = str(text_path)
        os.environ["ANIMA_REFLORA_MODEL_VAE"] = str(vae_path)
        features = load_features(ckpt)
        args = SimpleNamespace(
            storage=Path(storage),
            output_dir=Path(tempfile.gettempdir()) / "anima_comfy",
            ccip_cache=Path(storage) / "runs/ccip_ref_head_emb_cache.pt",
            head_roi_cache=Path(storage) / "runs/head_roi_cache.pt",
            frames=int(features.get("frames", 3)),
            dtype=dtype,
            device=device,
        )
        config = config_for_infer(args, ckpt, features)
        dev = torch.device(device)
        dt = _DTYPES[dtype]

        vae = load_vae(config, device=dev, dtype=dt)
        if vae_chunk_size > 0 and hasattr(vae, "enable_spatial_chunking"):
            vae.enable_spatial_chunking(vae_chunk_size)
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        ccip_dim = 768 if config.cpm else None
        model = build_model(config, ccip_dim=ccip_dim).to(device=dev, dtype=dt)
        load_checkpoint_into(model, ckpt, strict=False)
        for name, module in sidecar_modules(model).items():
            load_sidecar_into(module, ckpt, name, strict=True)
        dit = getattr(model, "dit", None)
        if dit is not None and config.rope_refpos:
            maybe_apply_sidecar(dit, ckpt, expected_frames=config.frames)
            assert_sidecar_applied(dit, ckpt, expected_frames=config.frames)
        if extra_lora:
            extra_name, extra_strength = extra_lora
            _apply_extra_lora(
                model, config, resolve_lora(extra_name), float(extra_strength), dev, dt,
            )
        model.eval()

        bundle = {
            "config": config, "model": model, "vae": vae,
            "device": dev, "dtype": dt, "features": features,
            "checkpoint": str(ckpt), "prompt_cache": {}, "text_device": "cpu",
        }
        _MODEL_CACHE[key] = bundle
        return (bundle,)


class AnimaExtraLora:
    """Select an additional standard Anima LoRA for the standalone loader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name": (lora_names(),),
            "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("ANIMA_EXTRA_LORA",)
    FUNCTION = "configure"
    CATEGORY = "AnimaRefLora"

    def configure(self, lora_name, strength):
        return ((lora_name, strength),)


class AnimaRefEncode:
    """Encode a reference image into F0/F1 latents + CCIP identity embedding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anima_model": ("ANIMA_MODEL",),
                "ref_image": ("IMAGE",),
                "ref_letterbox": ("BOOLEAN", {"default": True}),
                "generation_width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "generation_height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("ANIMA_REF",)
    FUNCTION = "encode"
    CATEGORY = "AnimaRefLora"

    def encode(self, anima_model, ref_image, ref_letterbox, generation_width, generation_height):
        b = anima_model
        config, dev, dt = b["config"], b["device"], b["dtype"]
        if generation_width % 64 or generation_height % 64:
            raise ValueError("generation_width and generation_height must be multiples of 64")
        pil = _comfy_to_pil(ref_image)
        # head detection + CCIP crop operate on files; use a stable temp path
        tmp = Path(tempfile.gettempdir()) / "anima_comfy_ref.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pil.save(tmp)
        bucket_w, bucket_h = int(generation_width), int(generation_height)
        with torch.no_grad():
            ref_latents = encode_ref_pair(
                b["vae"], tmp, bucket_w, bucket_h,
                frames=config.frames, config=config, device=dev, dtype=dt,
                letterbox=ref_letterbox,
            )
            ccip = compute_ccip_head_embedding(
                tmp, config=config, bucket_size=(bucket_w, bucket_h),
            ) if config.cpm else None
        return ({
            "ref_latents": ref_latents,
            "ccip_embedding": ccip,
            "cpm_valid": ccip is not None,
            "bucket": (bucket_w, bucket_h),
        },)


class AnimaRefLoraSampler:
    """Sample the target frame conditioned on the encoded reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anima_model": ("ANIMA_MODEL",),
                "anima_ref": ("ANIMA_REF",),
                # the full caption, sent as-is (nothing is injected behind the
                # scenes); the default shows the recommended prefix + year tag
                "prompt": ("STRING", {"multiline": True, "default": f"{QUALITY_PREFIX}, 1girl, white dress, smile, year2024"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ANIMA_NEGATIVE_PROMPT}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 15.0, "step": 0.1}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "ref_frame_mode": (["both", "head_only", "full_only", "blank"], {"default": "both"}),
            },
            "optional": {
                # one batch-2 forward for cond+uncond (faster); disable to
                # reproduce the historical sequential two-call outputs exactly
                "batched_cfg": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "AnimaRefLora"

    def _encode_prompts(self, bundle, texts):
        missing = [t for t in texts if t not in bundle["prompt_cache"]]
        if missing:
            encoder = SdScriptsPromptEncoder(
                bundle["config"], device=torch.device(bundle["text_device"]),
                dtype=torch.float32, batch_size=len(missing),
            )
            encoded = encoder.encode(missing)
            bundle["prompt_cache"].update(encoded)
            del encoder
        return [bundle["prompt_cache"][t] for t in texts]

    def sample(self, anima_model, anima_ref, prompt, negative_prompt, seed, steps,
               guidance_scale, flow_shift, ref_frame_mode, batched_cfg=True):
        b = anima_model
        config, dev, dt = b["config"], b["device"], b["dtype"]
        # tag normalization only (underscores/newlines); no hidden additions
        caption = build_caption(prompt, 0, "")
        pos, neg = self._encode_prompts(b, [caption, negative_prompt])
        ref_latents, cpm_valid = apply_ref_frame_mode(
            list(anima_ref["ref_latents"]), ref_frame_mode
        )
        cpm_valid = cpm_valid and anima_ref["cpm_valid"]
        with torch.no_grad():
            latent = sample_target(
                b["model"], config, ref_latents, pos,
                neg if guidance_scale > 1.0 else None,
                anima_ref["ccip_embedding"], cpm_valid,
                steps=steps, flow_shift=flow_shift, guidance_scale=guidance_scale,
                seed=seed, device=dev, dtype=dt, batched_cfg=batched_cfg,
            )
            img = decode_latent(b["vae"], latent, dev)
        return (_pil_to_comfy(img),)


NODE_CLASS_MAPPINGS = {
    "AnimaExtraLora": AnimaExtraLora,
    "AnimaRefLoraLoader": AnimaRefLoraLoader,
    "AnimaRefEncode": AnimaRefEncode,
    "AnimaRefLoraSampler": AnimaRefLoraSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaExtraLora": "Anima Extra LoRA (standalone)",
    "AnimaRefLoraLoader": "Anima RefLora Loader",
    "AnimaRefEncode": "Anima Ref Encode",
    "AnimaRefLoraSampler": "Anima RefLora Sampler",
}
