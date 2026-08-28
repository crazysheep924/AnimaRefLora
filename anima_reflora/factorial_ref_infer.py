from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageDraw, ImageOps

from .checkpoints import load_checkpoint_into, load_sidecar_into
from .features import CcipEmbeddingCache
from .local_ref_ab_infer import (
    ANIMA_DEFAULT_EVAL_PROMPT,
    ANIMA_NEGATIVE_PROMPT,
    CCIP_IDENTITY_DIM,
    QUALITY_PREFIX,
    apply_or_verify_rope,
    build_caption,
    compute_bucket_local,
    compute_ccip_head_embedding,
    config_for_infer,
    decode_latent,
    encode_ref_pair,
    load_features,
    load_vae,
    lookup_ccip,
    ref_label,
    safe_name,
    sample_target,
    wsl_path,
)
from .models import build_model, sidecar_modules
from .ref_conditioning import detach_ref_conditioner
from .build_training_cache import SdScriptsPromptEncoder


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def find_character_refs(root: Path, names: list[str]) -> list[Path]:
    refs = []
    for name in names:
        pick = root / name / "pick"
        matches = sorted(path for path in pick.glob("0sample.*") if path.suffix.lower() in IMAGE_EXTS)
        if not matches:
            raise FileNotFoundError(f"No 0sample image found for {name}: {pick}")
        refs.append(matches[0])
    return refs


def apply_frame_switches(ref_latents: list[torch.Tensor], f0: bool, f1: bool) -> tuple[list[torch.Tensor], bool]:
    if len(ref_latents) < 2:
        return ([ref_latents[0] if f1 else torch.zeros_like(ref_latents[0])], bool(f1))
    head, full = ref_latents[0], ref_latents[1]
    return [
        head if f0 else torch.zeros_like(head),
        full if f1 else torch.zeros_like(full),
    ], bool(f1)


def disable_lora(model: torch.nn.Module) -> None:
    network = getattr(model, "network", None)
    if network is None:
        return
    for param in network.parameters():
        param.detach().zero_()


def disable_ref_conditioner(model: torch.nn.Module) -> None:
    dit = getattr(model, "dit", None)
    if dit is not None:
        detach_ref_conditioner(dit)
    if hasattr(model, "ref_conditioner"):
        model.ref_conditioner = None


def build_variant_model(
    config: Any,
    checkpoint: Path,
    *,
    ccip_dim: int | None,
    lora_on: bool,
    ref_on: bool,
    device: torch.device,
    dtype: torch.dtype,
    rope_diagnostic_override: str | None = None,
) -> torch.nn.Module:
    model = build_model(config, ccip_dim=ccip_dim).to(device=device, dtype=dtype)
    if lora_on:
        load_checkpoint_into(model, checkpoint, strict=False)
    else:
        disable_lora(model)
    if ref_on:
        for name, module in sidecar_modules(model).items():
            load_sidecar_into(module, checkpoint, name, strict=True)
    else:
        disable_ref_conditioner(model)
    apply_or_verify_rope(
        model,
        checkpoint,
        frames=config.frames,
        diagnostic_override=rope_diagnostic_override,
    )
    model.eval()
    return model


def write_character_grid(records: list[dict[str, Any]], out_path: Path, thumb: int = 256) -> None:
    cols = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    rows = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    by_key = {
        (item["lora_on"], item["ref_on"], item["f0_on"], item["f1_on"]): item
        for item in records
    }
    cell_w = thumb
    cell_h = thumb + 34
    header_h = 28
    label_w = 92
    canvas = Image.new("RGB", (label_w + cell_w * len(cols), header_h + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (f0, f1) in enumerate(cols):
        draw.text((label_w + col * cell_w + 8, 8), f"F0 {int(f0)} / F1 {int(f1)}", fill=(0, 0, 0))
    for row, (lora, ref) in enumerate(rows):
        y = header_h + row * cell_h
        draw.text((8, y + 8), f"L{int(lora)} R{int(ref)}", fill=(0, 0, 0))
        for col, (f0, f1) in enumerate(cols):
            item = by_key[(lora, ref, f0, f1)]
            with Image.open(item["path"]) as image:
                img = ImageOps.contain(image.convert("RGB"), (thumb, thumb))
            x = label_w + col * cell_w + (cell_w - img.width) // 2
            canvas.paste(img, (x, y))
            draw.text((label_w + col * cell_w + 6, y + thumb + 6), item["label"], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="16-way LoRA/ref/F0/F1 factorial reference inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref-root", default="/workspace/storage/val/test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--storage", default="/workspace/storage")
    parser.add_argument("--ccip-cache", default="/workspace/storage/runs/ccip_ref_head_emb_cache.pt")
    parser.add_argument("--head-roi-cache", default="/workspace/storage/runs/head_roi_cache.pt")
    parser.add_argument("--characters", nargs="+", required=True)
    parser.add_argument("--prompt", default=ANIMA_DEFAULT_EVAL_PROMPT)
    parser.add_argument("--negative-prompt", default=ANIMA_NEGATIVE_PROMPT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--frames", type=int, default=3, choices=[2, 3])
    parser.add_argument("--bucket-short", type=int, default=1024)
    parser.add_argument("--bucket-long-max", type=int, default=1024)
    parser.add_argument(
        "--ref-letterbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad the ref to the generation bucket aspect (white) instead of center-cropping it; --no-ref-letterbox restores the old crop behavior.",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--decode-device", default="cuda")
    parser.add_argument("--vae-chunk-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--ccip-on-the-fly", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rope-layout-override",
        choices=["identity", "disjoint", "shifted", "packed"],
        help="Diagnostic only: intentionally override the checkpoint RoPE sidecar.",
    )
    parser.add_argument("--rope-shift-override", type=float, help="Diagnostic RoPE shift used with --rope-layout-override.")
    args = parser.parse_args(argv)
    if args.rope_shift_override is not None and args.rope_layout_override is None:
        parser.error("--rope-shift-override requires --rope-layout-override")
    if args.rope_shift_override is not None and args.rope_shift_override < 0:
        parser.error("--rope-shift-override must be >= 0")

    checkpoint = wsl_path(args.checkpoint)
    args.ref_root = wsl_path(args.ref_root)
    args.output_dir = wsl_path(args.output_dir)
    args.storage = wsl_path(args.storage)
    args.ccip_cache = wsl_path(args.ccip_cache)
    args.head_roi_cache = wsl_path(args.head_roi_cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(checkpoint)
    config = config_for_infer(args, checkpoint, features)
    if config.frames < 3:
        raise SystemExit(
            "factorial_ref_infer needs a frames=3 (head+full) checkpoint: "
            "with a single ref frame the F0 columns would be meaningless."
        )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    text_device = torch.device(args.text_device)
    decode_device = torch.device(args.decode_device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32

    refs = find_character_refs(args.ref_root, args.characters)
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

    encoded_refs: dict[int, list[torch.Tensor]] = {}
    ref_meta: list[dict[str, Any]] = []
    for index, ref in enumerate(refs):
        with Image.open(ref) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            bucket_w, bucket_h, latent_hw = compute_bucket_local(*image.size, args.bucket_short, args.bucket_long_max)
        encoded_refs[index] = encode_ref_pair(
            vae, ref, bucket_w, bucket_h, frames=config.frames, config=config, device=device, dtype=dtype, letterbox=args.ref_letterbox
        )
        ref_meta.append({
            "index": index,
            "name": ref_label(ref),
            "path": str(ref),
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
    ccip_by_ref: dict[int, tuple[torch.Tensor | None, bool, str | None]] = {}
    for item in ref_meta:
        ref = Path(item["path"])
        ccip_embedding, cpm_valid, ccip_key = lookup_ccip(ccip_cache, ref)
        if config.cpm and not cpm_valid and args.ccip_on_the_fly:
            emb = compute_ccip_head_embedding(
                ref,
                config=config,
                bucket_short=args.bucket_short,
                bucket_long_max=args.bucket_long_max,
            )
            if emb is not None:
                ccip_embedding, cpm_valid, ccip_key = emb.float(), True, f"onfly:{ref.name}"
        ccip_by_ref[item["index"]] = (ccip_embedding, cpm_valid, ccip_key)

    manifest: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "features": features,
        "prompt": caption,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "rope_layout_override": args.rope_layout_override,
        "rope_shift_override": args.rope_shift_override,
        "refs": ref_meta,
        "outputs": [],
    }
    pending_decodes: list[tuple[torch.Tensor, Path, dict[str, Any]]] = []
    switches = [(l, r, f0, f1) for l in [False, True] for r in [False, True] for f0 in [False, True] for f1 in [False, True]]

    for lora_on in [False, True]:
        for ref_on in [False, True]:
            print(f"[factorial] building model lora={int(lora_on)} ref={int(ref_on)}", flush=True)
            model = build_variant_model(
                config,
                checkpoint,
                ccip_dim=ccip_dim,
                lora_on=lora_on,
                ref_on=ref_on,
                device=device,
                dtype=dtype,
                rope_diagnostic_override=args.rope_layout_override,
            )
            for item in ref_meta:
                # One seed per character, shared by ALL 16 cells: switch effects
                # must not be confounded with initial-noise variation.
                cell_seed = args.seed + item["index"] * 1000
                ccip_embedding, cpm_valid, ccip_key = ccip_by_ref[item["index"]]
                for f0_on in [False, True]:
                    for f1_on in [False, True]:
                        ref_latents, frame_allows_cpm = apply_frame_switches(encoded_refs[item["index"]], f0_on, f1_on)
                        effective_cpm_valid = bool(ref_on and cpm_valid and frame_allows_cpm)
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
                            seed=cell_seed,
                            device=device,
                            dtype=dtype,
                        )
                        name = safe_name(item["name"])
                        label = f"L{int(lora_on)} R{int(ref_on)} F0{int(f0_on)} F1{int(f1_on)}"
                        out_path = args.output_dir / name / f"{label.replace(' ', '_')}.png"
                        record = {
                            "index": item["index"],
                            "name": item["name"],
                            "label": label,
                            "lora_on": lora_on,
                            "ref_on": ref_on,
                            "f0_on": f0_on,
                            "f1_on": f1_on,
                            "path": str(out_path),
                            "reference_path": item["path"],
                            "ccip_valid": effective_cpm_valid,
                            "ccip_key": ccip_key,
                        }
                        pending_decodes.append((latent.detach().cpu(), out_path, record))
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = decode_latent(vae, latent, decode_device)
        image.save(out_path)
        manifest["outputs"].append(record)
    del vae
    gc.collect()
    if decode_device.type == "cuda":
        torch.cuda.empty_cache()

    for item in ref_meta:
        records = [record for record in manifest["outputs"] if record["index"] == item["index"]]
        write_character_grid(records, args.output_dir / f"{safe_name(item['name'])}_grid.png")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "images": len(manifest["outputs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
