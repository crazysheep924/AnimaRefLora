from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

# Let the CUDA allocator defragment/reuse memory; avoids spurious OOM at 1024
# from fragmentation across the staged VAE->DiT->VAE phases (matches infer.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageDraw, ImageOps

from .build_training_cache import (
    SdScriptsPromptEncoder,
    encode_image_latent,
    load_vae,
    make_head_image,
    resize_full_fill_crop,
)
from .checkpoints import load_checkpoint_into, load_sidecar_into, load_tensor_file
from .config import ANIMA_DEFAULT_EVAL_PROMPT, ANIMA_NEGATIVE_PROMPT, parse_config
from .features import CcipEmbeddingCache, feature_sidecar_path
from .models import build_model, sidecar_modules
from .rope_refpos import assert_sidecar_applied, maybe_apply_sidecar


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
QUALITY_PREFIX = "masterpiece, best quality, score_9, newest"


def wsl_path(value: str | Path) -> Path:
    text = str(value)
    text = text.replace("\\", "/")
    for prefix in ("//wsl.localhost/Ubuntu", "//wsl$/Ubuntu"):
        if text.startswith(prefix):
            return Path(text[len(prefix) :] or "/")
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2)
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "ref"


def ref_label(path: Path) -> str:
    if path.parent.name == "pick" and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem


def compute_bucket_local(w: int, h: int, bucket_short: int, bucket_long_max: int) -> tuple[int, int, tuple[int, int]]:
    long_side = max(w, h)
    short_side = min(w, h)
    aspect = long_side / short_side
    long_px = round(bucket_short * aspect / 64) * 64
    long_px = min(long_px, bucket_long_max)
    long_px = max(long_px, bucket_short)
    if w >= h:
        bw, bh = long_px, bucket_short
    else:
        bw, bh = bucket_short, long_px
    return bw, bh, (bh // 8, bw // 8)


def select_refs(root: Path, limit: int) -> list[Path]:
    refs = sorted(
        path
        for path in root.glob("*/pick/0sample.*")
        if path.suffix.lower() in IMAGE_EXTS
    )
    if len(refs) < limit:
        raise FileNotFoundError(f"Need {limit} 0sample refs under {root}, found {len(refs)}")
    return refs[:limit]


def build_caption(prompt: str, year: int, prefix: str) -> str:
    parts = [prefix.rstrip(",")]
    parts.extend(part.strip().replace("_", " ") for part in prompt.replace("\n", ",").split(",") if part.strip())
    if int(year) > 0:
        parts.append(f"year{int(year)}")
    return ", ".join(part for part in parts if part)


def load_features(checkpoint: Path) -> dict[str, Any]:
    from .bundle import bundle_features, is_bundle

    if is_bundle(checkpoint):
        return bundle_features(checkpoint)
    sidecar = feature_sidecar_path(checkpoint)
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def config_for_infer(args: argparse.Namespace, checkpoint: Path, features: dict[str, Any]):
    os.environ.setdefault("ANIMA_REFLORA_MODEL_DIT", str(args.storage / "anima_models/diffusion_models/anima-base-v1.0.safetensors"))
    os.environ.setdefault("ANIMA_REFLORA_MODEL_TE", str(args.storage / "anima_models/text_encoders"))
    os.environ.setdefault("ANIMA_REFLORA_MODEL_VAE", str(args.storage / "anima_models/vae/qwen_image_vae.safetensors"))
    os.environ["ANIMA_REFLORA_CCIP_EMB_CACHE"] = str(args.ccip_cache)
    os.environ["ANIMA_REFLORA_HEAD_ROI_CACHE"] = str(args.head_roi_cache)
    cfg_args = [
        "--stage",
        "from0-headroi-rope-cpm",
        "--from-scratch",
        "--frames",
        str(int(features.get("frames", args.frames))),
        "--network",
        "lokr",
        "--network-dim",
        "512",
        "--network-alpha",
        "512",
        "--no-grad-checkpoint",
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--storage",
        str(args.storage),
        "--out-dir",
        str(args.output_dir),
        "--ccip-cache",
        str(args.ccip_cache),
        "--head-roi-cache",
        str(args.head_roi_cache),
        "--head-loss-weight",
        str(float(features.get("head_loss_weight", 4.0))),
        "--prompt-mode",
        "change_only",
    ]
    rope_override = getattr(args, "rope_layout_override", None)
    rope_layout = rope_override or str(features.get("rope_layout", "disjoint"))
    rope_shift_override = getattr(args, "rope_shift_override", None)
    rope_shift = float(features.get("rope_shift", 1.0) if rope_shift_override is None else rope_shift_override)
    if (rope_override != "identity") and (rope_override is not None or features.get("rope_refpos", True)):
        cfg_args += [
            "--rope-refpos",
            "--rope-layout",
            rope_layout,
            "--rope-shift",
            str(rope_shift),
        ]
    if features.get("cpm", True):
        cfg_args.append("--cpm")
    # CRePA is a TRAINING-ONLY auxiliary loss (its projector is a separate sidecar
    # and the DiT itself is unmodified). Enabling it at inference only installs a
    # forward hook that captures+retains hidden activations every step — wasted
    # VRAM/compute that is never read while sampling. Reference infer.py never
    # enables crepa, so we keep it off here regardless of the trained feature flag.
    config = parse_config(cfg_args)
    config.paths().ensure_text_encoder_symlink()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return config


def apply_or_verify_rope(model: torch.nn.Module, checkpoint: Path, *, frames: int, diagnostic_override: str | None) -> None:
    dit = getattr(model, "dit", None)
    if dit is None:
        return
    if diagnostic_override is not None:
        print(
            f"[rope-refpos] DIAGNOSTIC override={diagnostic_override}; checkpoint sidecar compatibility intentionally bypassed",
            flush=True,
        )
        return
    maybe_apply_sidecar(dit, checkpoint, expected_frames=frames)
    assert_sidecar_applied(dit, checkpoint, expected_frames=frames)


def set_ref_conditioner_components(
    model: torch.nn.Module,
    *,
    frame_adapter_on: bool,
    cpm_on: bool,
) -> dict[str, int]:
    dit = getattr(model, "dit", None)
    conditioner = getattr(dit, "ref_conditioner", None)
    counts = {"frame_adapter_gates_zeroed": 0, "cpm_gates_zeroed": 0}
    if conditioner is None:
        return counts
    with torch.no_grad():
        for block in conditioner.blocks.values():
            if not frame_adapter_on and block.adapter_gate is not None:
                block.adapter_gate.zero_()
                counts["frame_adapter_gates_zeroed"] += 1
            if not cpm_on and block.cpm_gate is not None:
                block.cpm_gate.zero_()
                counts["cpm_gates_zeroed"] += 1
    return counts


def letterbox_to_bucket(image: Image.Image, bucket_w: int, bucket_h: int, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Pad (never crop) the image to the bucket aspect so the subsequent
    resize_full_fill_crop degenerates to a pure resize.

    Without this, an aspect mismatch between the ref and the generation bucket
    center-crops the ref — edge details (animal ears, horns, tails, held
    weapons) silently vanish from the F1 frame. White fill is close to the
    training distribution (white-background character art)."""
    target = bucket_w / bucket_h
    w, h = image.size
    if abs(w / h - target) < 1e-3:
        return image
    if w / h < target:
        new_w = int(round(h * target))
        canvas = Image.new("RGB", (new_w, h), fill)
        canvas.paste(image, ((new_w - w) // 2, 0))
    else:
        new_h = int(round(w / target))
        canvas = Image.new("RGB", (w, new_h), fill)
        canvas.paste(image, (0, (new_h - h) // 2))
    return canvas


def encode_ref_pair(
    vae: Any,
    image_path: Path,
    bucket_w: int,
    bucket_h: int,
    *,
    frames: int,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    letterbox: bool = False,
) -> list[torch.Tensor]:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if letterbox:
            image = letterbox_to_bucket(image, bucket_w, bucket_h)
        full = resize_full_fill_crop(image, bucket_w, bucket_h)
    full_latent = encode_image_latent(vae, full, device, dtype)[0, :, 0]
    if frames <= 2:
        return [full_latent]
    head = make_head_image(
        image_path,
        bucket_w,
        bucket_h,
        conf_threshold=config.head_crop_conf,
        padding=config.head_crop_padding,
        detector=None,
    )
    if head is None:
        head = full
    head_latent = encode_image_latent(vae, head, device, dtype)[0, :, 0]
    return [head_latent, full_latent]


def decode_latent(vae: Any, latent: torch.Tensor, device: torch.device) -> Image.Image:
    lat = latent.unsqueeze(0).to(device=device, dtype=vae.dtype)
    pixels = vae.decode_to_pixels(lat)
    img = ((pixels[0].float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())


def lookup_ccip(cache: CcipEmbeddingCache | None, path: Path) -> tuple[torch.Tensor | None, bool, str | None]:
    if cache is None:
        return None, False, None
    candidates = [
        str(path),
        path.as_posix(),
        path.name,
        path.stem,
        path.parent.parent.name if len(path.parents) >= 2 else "",
    ]
    if path.as_posix().startswith("/path/to/"):
        candidates.append("E:/" + path.as_posix()[7:])
    if "/workspace/storage/val/test/" in path.as_posix():
        candidates.append(path.as_posix().replace("/workspace/storage/val/test", "/path/to/eval_refs"))
    for candidate in candidates:
        if not candidate:
            continue
        tensor = cache.lookup(candidate)
        if tensor is not None:
            return tensor.float(), True, candidate
    return torch.zeros(cache.dim, dtype=torch.float32), False, None


# CCIP head-crop feature dim the CPM identity adapter was trained on.
CCIP_IDENTITY_DIM = 768


def compute_ccip_head_embedding(
    image_path: Path,
    *,
    config: Any,
    bucket_short: int | None = None,
    bucket_long_max: int | None = None,
    bucket_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    """On-the-fly CCIP head-crop identity embedding for CPM, matching exactly how
    the training cache was built: head_cache.crop_head_image (imgutils detect_heads
    -> highest-score bbox -> expand -> resize to the bucket) -> ccip_batch_extract_features
    (size=384, ccip-caformer-24-randaug-pruned) -> L2 normalise. Returns a 768-d unit
    vector, or None if head detection / CCIP extraction fails (caller keeps CPM invalid)."""
    try:
        from imgutils.metrics.ccip import ccip_batch_extract_features

        from .ccip_head_cache import _normalise
        from .head_cache import crop_head_image
    except Exception as exc:  # imgutils missing -> cannot condition CPM on the fly
        print(f"[infer] CCIP on-the-fly unavailable ({exc}); CPM stays invalid", flush=True)
        return None
    try:
        with Image.open(image_path) as opened:
            w, h = ImageOps.exif_transpose(opened).size
        if bucket_size is not None:
            bucket_w, bucket_h = bucket_size
            latent_hw = (bucket_h // 8, bucket_w // 8)
        else:
            if bucket_short is None or bucket_long_max is None:
                raise ValueError("bucket_short and bucket_long_max are required without bucket_size")
            _, _, latent_hw = compute_bucket_local(w, h, bucket_short, bucket_long_max)
        crop = crop_head_image(
            Path(image_path),
            latent_hw,
            conf_threshold=config.head_crop_conf,
            padding=config.head_crop_padding,
            detector=None,
        )
        feats = list(ccip_batch_extract_features([crop], size=384, model="ccip-caformer-24-randaug-pruned"))
        if not feats:
            return None
        return _normalise(torch.as_tensor(feats[0]))
    except Exception as exc:
        print(f"[infer] CCIP on-the-fly failed for {image_path}: {exc}", flush=True)
        return None


@torch.no_grad()
def sample_target(
    model: torch.nn.Module,
    config: Any,
    ref_latents: list[torch.Tensor],
    prompt: dict[str, torch.Tensor],
    negative: dict[str, torch.Tensor] | None,
    ccip_embedding: torch.Tensor | None,
    cpm_valid: bool,
    *,
    steps: int,
    flow_shift: float,
    guidance_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    batched_cfg: bool = True,
) -> torch.Tensor:
    refs = torch.stack([latent.to(device=device, dtype=dtype) for latent in ref_latents], dim=0)
    refs = refs.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    _, _, frames_minus_one, lat_h, lat_w = refs.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x1 = torch.randn((1, 16, 1, lat_h, lat_w), generator=gen, dtype=torch.float32).to(device)
    sigmas = torch.linspace(1.0, 0.0, int(steps) + 1, device=device, dtype=torch.float32)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    batch = {
        "t5_input_ids": prompt["t5_input_ids"].to(device=device, dtype=torch.long),
        "t5_attn_mask": prompt["t5_attn_mask"].to(device=device),
        "ccip_embeddings": (
            ccip_embedding.view(1, -1).to(device=device, dtype=torch.float32)
            if ccip_embedding is not None
            else None
        ),
        "ccip_valid": torch.tensor([bool(cpm_valid)], device=device, dtype=torch.bool),
        "cpm_valid": torch.tensor([bool(cpm_valid)], device=device, dtype=torch.bool),
    }
    prompt_embeds = prompt["prompt_embeds"].to(device=device, dtype=dtype)
    attn_mask = prompt["attn_mask"].to(device=device)
    neg_embeds = negative["prompt_embeds"].to(device=device, dtype=dtype) if negative else None
    neg_mask = negative["attn_mask"].to(device=device) if negative else None
    neg_batch = dict(batch)
    if negative:
        neg_batch["t5_input_ids"] = negative["t5_input_ids"].to(device=device, dtype=torch.long)
        neg_batch["t5_attn_mask"] = negative["t5_attn_mask"].to(device=device)

    use_cfg = bool(negative) and guidance_scale > 1.0
    use_batched = use_cfg and batched_cfg
    if use_batched:
        # Fuse cond+uncond into one batch-2 forward: halves the per-step model
        # calls at the cost of transient activations (fine on 24GB+ cards).
        # The encoder pads every caption to a fixed 512 tokens, so cond/uncond
        # embeddings always share sequence length and concatenate cleanly.
        cfg_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)
        cfg_mask = torch.cat([attn_mask, neg_mask], dim=0)
        cfg_batch = {
            "t5_input_ids": torch.cat([batch["t5_input_ids"], neg_batch["t5_input_ids"]], dim=0),
            "t5_attn_mask": torch.cat([batch["t5_attn_mask"], neg_batch["t5_attn_mask"]], dim=0),
            "ccip_embeddings": (
                torch.cat([batch["ccip_embeddings"], batch["ccip_embeddings"]], dim=0)
                if batch["ccip_embeddings"] is not None
                else None
            ),
            "ccip_valid": torch.cat([batch["ccip_valid"], batch["ccip_valid"]], dim=0),
            "cpm_valid": torch.cat([batch["cpm_valid"], batch["cpm_valid"]], dim=0),
        }

    target_idx = frames_minus_one
    for index in range(int(steps)):
        sigma = sigmas[index]
        x = torch.cat([refs, x1.to(dtype=dtype)], dim=2)
        timesteps = torch.zeros((1, frames_minus_one + 1), device=device, dtype=torch.float32)
        timesteps[:, target_idx] = float(sigma)
        if use_batched:
            v_both = model(
                torch.cat([x, x], dim=0),
                torch.cat([timesteps, timesteps], dim=0),
                caption_embeds=cfg_embeds,
                attention_mask=cfg_mask,
                batch=cfg_batch,
                config=config,
            )[:, :, target_idx : target_idx + 1].float()
            v_pos, v_neg = v_both[0:1], v_both[1:2]
            velocity = v_pos + (float(guidance_scale) - 1.0) * (v_pos - v_neg)
        else:
            v_pos = model(
                x,
                timesteps,
                caption_embeds=prompt_embeds,
                attention_mask=attn_mask,
                batch=batch,
                config=config,
            )[:, :, target_idx : target_idx + 1].float()
            velocity = v_pos
            if use_cfg:
                v_neg = model(
                    x,
                    timesteps,
                    caption_embeds=neg_embeds,
                    attention_mask=neg_mask,
                    batch=neg_batch,
                    config=config,
                )[:, :, target_idx : target_idx + 1].float()
                velocity = velocity + (float(guidance_scale) - 1.0) * (v_pos - v_neg)
        x1 = x1 + velocity * (sigmas[index + 1] - sigma)
    return x1[0, :, 0]


def apply_ref_frame_mode(ref_latents: list[torch.Tensor], mode: str) -> tuple[list[torch.Tensor], bool]:
    if mode == "both":
        return ref_latents, True
    if len(ref_latents) < 2:
        # frames=2 checkpoints have a single (full) ref frame: full_only keeps
        # it, blank zeroes it; head_only has no head frame to isolate.
        if mode == "full_only":
            return ref_latents, True
        if mode == "blank":
            return [torch.zeros_like(item) for item in ref_latents], False
        raise ValueError(f"ref_frame_mode {mode!r} requires a head+full (frames=3) checkpoint")
    head, full = ref_latents[0], ref_latents[1]
    zero_head = torch.zeros_like(head)
    zero_full = torch.zeros_like(full)
    if mode == "head_only":
        return [head, zero_full], False
    if mode == "full_only":
        return [zero_head, full], True
    if mode == "blank":
        return [zero_head, zero_full], False
    raise ValueError(f"Unknown ref_frame_mode: {mode}")


def condition_seed(base_seed: int, ref_index: int, condition: str, same_condition_seed: bool = False) -> int:
    offset = 0 if same_condition_seed else {"correct": 0, "wrong": 1, "blank": 2}[condition]
    return int(base_seed) + int(ref_index) * 100 + offset


def parse_extra_lora_arg(value: str) -> tuple[Path, float]:
    """Parse PATH[:MULT]. The trailing :MULT is optional; a colon that does not
    parse as a float (e.g. the drive colon in E:/loras/x.safetensors) is kept
    as part of the path."""
    path_text, multiplier = value, 1.0
    head, sep, tail = value.rpartition(":")
    if sep:
        try:
            multiplier = float(tail)
            path_text = head
        except ValueError:
            pass
    return wsl_path(path_text), multiplier


def normalize_extra_lora_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept sd-scripts native keys (lora_unet_<module>.lora_down.weight) as-is
    and convert diffusers/peft-style keys (diffusion_model.<module.path>.lora_A.weight)
    to that layout. peft files carry no alpha tensors, so a missing alpha is
    synthesized as rank (scale 1.0)."""
    if all(key.startswith(("lora_unet_", "lora_te")) or "." not in key for key in state):
        return state
    suffix_map = {"lora_A": "lora_down", "lora_B": "lora_up", "lora_down": "lora_down", "lora_up": "lora_up"}
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        base = key
        for prefix in ("diffusion_model.", "transformer."):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                break
        parts = base.split(".")
        if parts[-1] == "alpha":
            converted["lora_unet_" + "_".join(parts[:-1]) + ".alpha"] = value
            continue
        if len(parts) >= 3 and parts[-1] == "weight" and parts[-2] in suffix_map:
            module = "_".join(parts[:-2])
            converted[f"lora_unet_{module}.{suffix_map[parts[-2]]}.weight"] = value
            continue
        raise ValueError(f"Unsupported extra-LoRA key format: {key}")
    for key in list(converted):
        if key.endswith(".lora_down.weight"):
            alpha_key = key[: -len(".lora_down.weight")] + ".alpha"
            if alpha_key not in converted:
                converted[alpha_key] = torch.tensor(float(converted[key].shape[0]))
    return converted


def _merge_dora_network(network: Any, weights_sd: dict[str, torch.Tensor], multiplier: float, device: torch.device) -> None:
    """DoRA-aware merge (sd-scripts lora_anima has no dora_scale support).
    W_dora = dora_scale * V / ||V||_row with V = W0 + (alpha/rank) * up@down,
    then W' = W0 + multiplier * (W_dora - W0) (ComfyUI weight_decompose strength
    semantics, so partial strength interpolates the decomposed result instead of
    scaling the delta before normalisation)."""
    for lora in network.unet_loras:
        prefix = lora.lora_name + "."
        sd = {key[len(prefix) :]: value for key, value in weights_sd.items() if key.startswith(prefix)}
        org = lora.org_module
        org_sd = org.state_dict()
        w0 = org_sd["weight"].to(device=device, dtype=torch.float32)
        down = sd["lora_down.weight"].to(device=device, dtype=torch.float32)
        up = sd["lora_up.weight"].to(device=device, dtype=torch.float32)
        alpha = float(sd["alpha"]) if "alpha" in sd else float(down.shape[0])
        scale = alpha / down.shape[0]
        if w0.dim() == 2:
            delta = up @ down
        elif down.shape[2:4] == (1, 1):
            delta = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            delta = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        v = w0 + scale * delta
        tail = [1] * (v.dim() - 1)
        dora_scale = sd["dora_scale"].to(device=device, dtype=torch.float32).reshape(v.shape[0], *tail)
        norm = v.reshape(v.shape[0], -1).norm(dim=1).clamp_min(1e-8).reshape(v.shape[0], *tail)
        w_dora = dora_scale / norm * v
        merged = w0 + multiplier * (w_dora - w0)
        org_sd["weight"] = merged.to(dtype=org_sd["weight"].dtype, device=org_sd["weight"].device)
        org.load_state_dict(org_sd)


def merge_extra_loras(
    model: torch.nn.Module,
    specs: list[tuple[Path, float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Merge external Anima LoRAs directly into the frozen base DiT weights.
    The trained LoKr network computes its delta on top of the (now merged) base,
    so checkpoint adapter + extra LoRA compose additively."""
    if not specs:
        return []
    dit = getattr(model, "dit", None)
    if dit is None:
        raise AttributeError("--extra-lora requires the sd-scripts Anima backend (model has no .dit)")
    from networks import lora_anima  # sd-scripts is on sys.path after build_model

    merged: list[dict[str, Any]] = []
    for path, multiplier in specs:
        if not path.exists():
            raise FileNotFoundError(f"Extra LoRA not found: {path}")
        state = normalize_extra_lora_state(load_tensor_file(path))
        te_keys = [key for key in state if key.startswith("lora_te")]
        if te_keys:
            print(
                f"[extra-lora] {path.name}: ignoring {len(te_keys)} text-encoder keys (captions are pre-encoded)",
                flush=True,
            )
            state = {key: value for key, value in state.items() if not key.startswith("lora_te")}
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier, None, None, [], dit, weights_sd=state, for_inference=True
        )
        has_dora = any(key.endswith(".dora_scale") for key in weights_sd)
        if has_dora:
            _merge_dora_network(network, weights_sd, multiplier, device)
        else:
            network.merge_to([], dit, weights_sd, dtype=None, device=device)
        if not network.unet_loras:
            raise RuntimeError(f"Extra LoRA {path} matched no DiT modules (unsupported key format?)")
        expected = len({key.split(".")[0] for key in weights_sd if "lora_down" in key})
        if len(network.unet_loras) != expected:
            print(
                f"[extra-lora] WARNING {path.name}: {expected} modules in file but only "
                f"{len(network.unet_loras)} matched the DiT",
                flush=True,
            )
        info = {
            "path": str(path),
            "multiplier": multiplier,
            "modules": len(network.unet_loras),
            "dora": has_dora,
        }
        print(
            f"[extra-lora] merged {path.name} x{multiplier} into base DiT "
            f"({info['modules']} modules{', DoRA' if has_dora else ''})",
            flush=True,
        )
        merged.append(info)
    return merged


def zero_lora_network(model: torch.nn.Module) -> int:
    network = getattr(model, "network", None)
    if network is None:
        network = getattr(getattr(model, "model", None), "network", None)
    if network is None:
        raise AttributeError("model has no LoRA network to zero")
    total = 0
    with torch.no_grad():
        for param in network.parameters():
            param.zero_()
            total += param.numel()
    return total


def set_lora_multiplier(model: torch.nn.Module, multiplier: float) -> int:
    """Scale the trained adapter network (LoKr) at runtime: every adapter module
    applies `multiplier` to its delta in forward, so no weights are touched."""
    network = getattr(model, "network", None)
    if network is None:
        network = getattr(getattr(model, "model", None), "network", None)
    if network is None:
        raise AttributeError("model has no LoRA network to scale")
    if hasattr(network, "multiplier"):
        network.multiplier = multiplier
    count = 0
    for module in network.modules():
        if module is not network and hasattr(module, "multiplier"):
            module.multiplier = multiplier
            count += 1
    if count == 0:
        raise RuntimeError("no adapter modules with a multiplier attribute found")
    return count


def write_grid(rows: list[dict[str, Any]], out_path: Path, thumb: int = 256) -> None:
    cols = ["reference", "correct", "wrong_ref", "wrong", "blank"]
    labels = ["ref", "correct", "wrong ref", "wrong", "blank"]
    cell_w = thumb
    cell_h = thumb + 28
    canvas = Image.new("RGB", (cell_w * len(cols), cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        for col_idx, key in enumerate(cols):
            path = row.get(key)
            if path is None:
                continue
            with Image.open(path) as image:
                img = ImageOps.contain(image.convert("RGB"), (thumb, thumb))
            x = col_idx * cell_w + (cell_w - img.width) // 2
            y = row_idx * cell_h
            canvas.paste(img, (x, y))
            draw.text((col_idx * cell_w + 6, y + thumb + 6), labels[col_idx], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local step checkpoint correct/wrong/blank reference inference.")
    parser.add_argument("--checkpoint", default="/work/RunpodTraining/checkpoints/lora_step_5000.safetensors")
    parser.add_argument("--ref-root", default="/workspace/storage/val/test")
    parser.add_argument("--wrong-ref", default="/work/RunpodTraining/wrong.jpg")
    parser.add_argument("--output-dir", default="/work/RunpodTraining/generated/ref_ab_step5000")
    parser.add_argument("--storage", default="/workspace/storage")
    parser.add_argument("--ccip-cache", default="/workspace/storage/runs/ccip_ref_head_emb_cache.pt")
    parser.add_argument("--head-roi-cache", default="/workspace/storage/runs/head_roi_cache.pt")
    parser.add_argument("--prompt", default=ANIMA_DEFAULT_EVAL_PROMPT)
    parser.add_argument("--negative-prompt", default=ANIMA_NEGATIVE_PROMPT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--frames", type=int, default=3, choices=[2, 3])
    parser.add_argument("--bucket-short", type=int, default=1024)
    parser.add_argument("--bucket-long-max", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default="cpu")
    # Decode on the GPU by default: the DiT is freed before decoding (the GPU is
    # empty by then), and VAE tiling keeps peak VRAM low. CPU decode of N x 1024
    # images is the dominant slowdown — pass --decode-device cpu only as a
    # last-resort fallback on very small cards.
    parser.add_argument("--decode-device", default="cuda")
    parser.add_argument("--vae-chunk-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    # When the CCIP cache has no entry for a reference (e.g. held-out val/test refs),
    # compute the CCIP head-crop identity embedding on the fly so CPM is actually
    # exercised instead of receiving a zero (invalid) identity. Matches training.
    parser.add_argument("--ccip-on-the-fly", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ref-letterbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad the ref to the generation bucket aspect (white) instead of center-cropping it; --no-ref-letterbox restores the old crop behavior.",
    )
    parser.add_argument("--ref-frame-mode", choices=["both", "head_only", "full_only", "blank"], default="both")
    parser.add_argument("--frame-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpm-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rope-layout-override",
        choices=["identity", "disjoint", "shifted", "packed"],
        help="Diagnostic only: intentionally override the checkpoint RoPE sidecar.",
    )
    parser.add_argument("--rope-shift-override", type=float, help="Diagnostic RoPE shift used with --rope-layout-override.")
    parser.add_argument(
        "--lora-multiplier",
        type=float,
        default=1.0,
        help="Runtime strength of the trained adapter network (LoKr); 1.0 = as trained, 0.0 ~ --zero-lora.",
    )
    parser.add_argument(
        "--zero-lora",
        action="store_true",
        help="Ablate LoRA safely: load the checkpoint and sidecars, then zero only the LoRA network weights.",
    )
    parser.add_argument(
        "--extra-lora",
        action="append",
        default=[],
        metavar="PATH[:MULT]",
        help="Merge an external Anima LoRA (sd-scripts lora_unet_* or diffusers lora_A/B keys) "
        "into the base DiT before sampling; repeatable, optional :MULT strength (default 1.0).",
    )
    parser.add_argument(
        "--conditions",
        default="correct,wrong,blank",
        help="comma-separated subset of correct/wrong/blank to generate",
    )
    parser.add_argument(
        "--same-condition-seed",
        action="store_true",
        help="Use identical noise for correct/wrong/blank; intended for paired visual comparisons.",
    )
    parser.add_argument(
        "--batched-cfg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run cond+uncond as one batch-2 forward (faster); --no-batched-cfg reproduces the sequential two-call behavior exactly.",
    )
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args(argv)
    if args.rope_shift_override is not None and args.rope_layout_override is None:
        parser.error("--rope-shift-override requires --rope-layout-override")
    if args.rope_shift_override is not None and args.rope_shift_override < 0:
        parser.error("--rope-shift-override must be >= 0")

    checkpoint = wsl_path(args.checkpoint)
    args.ref_root = wsl_path(args.ref_root)
    args.wrong_ref = wsl_path(args.wrong_ref)
    args.output_dir = wsl_path(args.output_dir)
    args.storage = wsl_path(args.storage)
    args.ccip_cache = wsl_path(args.ccip_cache)
    args.head_roi_cache = wsl_path(args.head_roi_cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(checkpoint)
    config = config_for_infer(args, checkpoint, features)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    text_device = torch.device(args.text_device)
    decode_device = torch.device(args.decode_device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32

    refs = select_refs(args.ref_root, args.limit)
    if not args.wrong_ref.exists():
        raise FileNotFoundError(f"Wrong reference not found: {args.wrong_ref}")
    caption = build_caption(args.prompt, args.year, QUALITY_PREFIX)
    text_dtype = torch.float32 if text_device.type == "cpu" else dtype
    encoder = SdScriptsPromptEncoder(config, device=text_device, dtype=text_dtype, batch_size=2)
    encoded = encoder.encode([caption, args.negative_prompt])
    prompt = encoded[caption]
    negative = encoded[args.negative_prompt] if args.guidance_scale > 1.0 else None
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ccip_cache = CcipEmbeddingCache(args.ccip_cache) if args.ccip_cache.exists() else None
    vae = load_vae(config, device=device, dtype=dtype)
    if args.vae_chunk_size > 0:
        try:
            vae.enable_spatial_chunking(args.vae_chunk_size)
        except Exception:
            pass
    try:
        vae.enable_tiling()
    except Exception:
        pass

    encoded_refs: dict[tuple[int, str], list[torch.Tensor]] = {}
    ref_meta: list[dict[str, Any]] = []
    for index, ref in enumerate(refs):
        with Image.open(ref) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            bucket_w, bucket_h, latent_hw = compute_bucket_local(*image.size, args.bucket_short, args.bucket_long_max)
        wrong = args.wrong_ref
        encoded_refs[(index, "correct")] = encode_ref_pair(
            vae, ref, bucket_w, bucket_h, frames=config.frames, config=config, device=device, dtype=dtype, letterbox=args.ref_letterbox
        )
        encoded_refs[(index, "wrong")] = encode_ref_pair(
            vae, wrong, bucket_w, bucket_h, frames=config.frames, config=config, device=device, dtype=dtype, letterbox=args.ref_letterbox
        )
        encoded_refs[(index, "blank")] = [torch.zeros_like(item) for item in encoded_refs[(index, "correct")]]
        ref_meta.append({
            "index": index,
            "name": ref_label(ref),
            "path": str(ref),
            "wrong_name": ref_label(wrong),
            "wrong_path": str(wrong),
            "bucket": [bucket_w, bucket_h],
            "latent_hw": list(latent_hw),
        })

    del vae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ccip_dim = ccip_cache.dim if ccip_cache is not None else (
        CCIP_IDENTITY_DIM if (config.cpm and args.ccip_on_the_fly) else None
    )
    model = build_model(config, ccip_dim=ccip_dim).to(device=device, dtype=dtype)
    load_checkpoint_into(model, checkpoint, strict=False)
    extra_lora_info = merge_extra_loras(model, [parse_extra_lora_arg(v) for v in args.extra_lora], device)
    zeroed_lora_params = zero_lora_network(model) if args.zero_lora else 0
    if args.lora_multiplier != 1.0:
        scaled = set_lora_multiplier(model, args.lora_multiplier)
        print(f"[lora-multiplier] adapter network scaled to x{args.lora_multiplier} ({scaled} modules)", flush=True)
    for name, module in sidecar_modules(model).items():
        load_sidecar_into(module, checkpoint, name, strict=True)
    component_counts = set_ref_conditioner_components(
        model,
        frame_adapter_on=args.frame_adapter,
        cpm_on=args.cpm_component,
    )
    apply_or_verify_rope(
        model,
        checkpoint,
        frames=config.frames,
        diagnostic_override=args.rope_layout_override,
    )
    model.eval()

    rows_for_grid: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "features": features,
        "prompt": caption,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "same_condition_seed": bool(args.same_condition_seed),
        "steps": args.steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "ref_frame_mode": args.ref_frame_mode,
        "extra_loras": extra_lora_info,
        "lora_multiplier": float(args.lora_multiplier),
        "zero_lora": bool(args.zero_lora),
        "zeroed_lora_params": int(zeroed_lora_params),
        "frame_adapter_on": bool(args.frame_adapter),
        "cpm_component_on": bool(args.cpm_component),
        "component_counts": component_counts,
        "rope_layout_override": args.rope_layout_override,
        "rope_shift_override": args.rope_shift_override,
        "refs": ref_meta,
        "outputs": [],
    }
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - {"correct", "wrong", "blank"}
    if unknown:
        raise SystemExit(f"Unknown --conditions entries: {sorted(unknown)}")
    pending_decodes: list[tuple[torch.Tensor, Path, dict[str, Any]]] = []
    onfly_emb: dict[str, torch.Tensor | None] = {}
    for item in ref_meta:
        row_paths: dict[str, Any] = {"reference": item["path"], "wrong_ref": item["wrong_path"]}
        for condition in conditions:
            cond_ref = Path(item["path"] if condition == "correct" else item["wrong_path"])
            if condition == "blank":
                ccip_embedding = torch.zeros(ccip_dim or config.cpm_identity_dim, dtype=torch.float32) if config.cpm else None
                cpm_valid = False
                ccip_key = None
            else:
                ccip_embedding, cpm_valid, ccip_key = lookup_ccip(ccip_cache, cond_ref)
                if config.cpm and not cpm_valid and args.ccip_on_the_fly:
                    key = str(cond_ref)
                    if key not in onfly_emb:
                        onfly_emb[key] = compute_ccip_head_embedding(
                            cond_ref,
                            config=config,
                            bucket_short=args.bucket_short,
                            bucket_long_max=args.bucket_long_max,
                        )
                    emb = onfly_emb[key]
                    if emb is not None:
                        ccip_embedding, cpm_valid, ccip_key = emb.float(), True, f"onfly:{cond_ref.name}"
            ref_latents, frame_allows_cpm = apply_ref_frame_mode(encoded_refs[(item["index"], condition)], args.ref_frame_mode)
            effective_cpm_valid = bool(cpm_valid and frame_allows_cpm)
            latent = sample_target(
                model,
                config,
                ref_latents,
                prompt,
                negative,
                ccip_embedding,
                effective_cpm_valid,
                steps=args.steps,
                flow_shift=args.flow_shift,
                guidance_scale=args.guidance_scale,
                seed=condition_seed(args.seed, item["index"], condition, args.same_condition_seed),
                device=device,
                dtype=dtype,
                batched_cfg=args.batched_cfg,
            )
            name = safe_name(item["name"])
            mode_tag = "" if args.ref_frame_mode == "both" else f"{args.ref_frame_mode}_"
            out_path = args.output_dir / f"{condition}_{mode_tag}{item['index']:02d}_{name}.png"
            row_paths[condition] = str(out_path)
            record = {
                "index": item["index"],
                "name": item["name"],
                "condition": condition,
                "ref_frame_mode": args.ref_frame_mode,
                "path": str(out_path),
                "reference_path": str(cond_ref) if condition != "blank" else None,
                "ccip_valid": bool(effective_cpm_valid),
                "ccip_key": ccip_key,
            }
            pending_decodes.append((latent.detach().cpu(), out_path, record))
        rows_for_grid.append(row_paths)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae = load_vae(config, device=decode_device, dtype=dtype if decode_device.type == "cuda" else torch.float32)
    if args.vae_chunk_size > 0:
        try:
            vae.enable_spatial_chunking(args.vae_chunk_size)
        except Exception:
            pass
    try:
        vae.enable_tiling()
    except Exception:
        pass
    for latent, out_path, record in pending_decodes:
        image = decode_latent(vae, latent, decode_device)
        image.save(out_path)
        manifest["outputs"].append(record)
    del vae
    gc.collect()
    if decode_device.type == "cuda":
        torch.cuda.empty_cache()

    mode_suffix = "" if args.ref_frame_mode == "both" else f"_{args.ref_frame_mode}"
    if not args.skip_grid and set(conditions) == {"correct", "wrong", "blank"}:
        write_grid(rows_for_grid, args.output_dir / f"comparison_grid{mode_suffix}.png")
    (args.output_dir / f"manifest{mode_suffix}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "images": len(manifest["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
