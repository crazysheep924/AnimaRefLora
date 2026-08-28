from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from .cache import LatentCacheIndex, LatentRecord
from .config import parse_config
from .sd_scripts_bridge import add_sd_scripts_to_path


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"]


@dataclass(frozen=True)
class MissingHead:
    record: LatentRecord
    head_key: tuple[str, tuple[int, int], str]


def missing_head_records(index: LatentCacheIndex) -> list[MissingHead]:
    by_key = index.records_by_key()
    missing: list[MissingHead] = []
    for record in index.records:
        if record.kind != "full" or not record.ref_eligible:
            continue
        head_key = (record.path, record.bucket, "head")
        if head_key not in by_key and head_key not in index.index.get("lat_idx", {}):
            missing.append(MissingHead(record=record, head_key=head_key))
    return missing


def resolve_image_path(record_path: str, image_root: str | None = None, source_prefix: str | None = None) -> Path:
    original = Path(record_path)
    candidates = []
    if original.exists():
        return original
    if image_root:
        root = Path(image_root)
        if source_prefix and record_path.startswith(source_prefix):
            rel = record_path[len(source_prefix) :].lstrip("/\\")
            candidates.append(root / rel)
        candidates.append(root / original.name)
        stem = original.stem
        for ext in IMAGE_EXTENSIONS:
            candidates.append(root / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Source image not found for head crop: {record_path}. "
        "Mount the images at their cache paths or set --image-root/ANIMA_REFLORA_IMAGES."
    )


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    target_aspect: float,
    padding: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    crop_w = width * padding
    crop_h = height * padding
    if crop_w / crop_h < target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect
    img_w, img_h = image_size
    left = int(round(cx - crop_w / 2.0))
    right = int(round(cx + crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    bottom = int(round(cy + crop_h / 2.0))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= right - img_w
        right = img_w
    if bottom > img_h:
        top -= bottom - img_h
        bottom = img_h
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, max(left + 1, right))
    bottom = min(img_h, max(top + 1, bottom))
    return left, top, right, bottom


def crop_head_image(
    image_path: Path,
    latent_bucket: tuple[int, int],
    conf_threshold: float,
    padding: float,
    detector: Callable | None = None,
) -> Image.Image:
    if detector is None:
        from imgutils.detect import detect_heads

        detector = detect_heads
    detections = detector(str(image_path), conf_threshold=conf_threshold)
    if not detections:
        raise RuntimeError(f"No head detected for reference image: {image_path}")
    bbox, _label, _score = max(detections, key=lambda item: item[2])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        lat_h, lat_w = latent_bucket
        pixel_h = lat_h * 8
        pixel_w = lat_w * 8
        crop_box = expand_bbox(tuple(int(v) for v in bbox), image.size, target_aspect=pixel_w / pixel_h, padding=padding)
        return image.crop(crop_box).resize((pixel_w, pixel_h), Image.Resampling.LANCZOS)


def image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)


def load_vae(config, device: torch.device, dtype: torch.dtype):
    add_sd_scripts_to_path(config)
    # Direct VAE-module import; anima_train_utils would pull checkpoint_io ->
    # model_util -> diffusers' SD pipelines (see build_training_cache.load_vae).
    from library import qwen_image_autoencoder_kl

    vae = qwen_image_autoencoder_kl.load_vae(
        str(config.paths().model_vae),
        device=device,
        disable_mmap=True,
        spatial_chunk_size=None,
        disable_cache=False,
    )
    vae.to(dtype)
    vae.eval()
    return vae


def encode_head_latent(vae, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad():
        pixels = image_to_tensor(image, device=device, dtype=dtype)
        latent = vae.encode_pixels_to_latents(pixels)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return latent.to("cpu", dtype=torch.bfloat16)


def atomic_pickle_dump(obj, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_missing_head_cache(config, dry_run: bool = False) -> dict:
    cache_dir = config.paths().latcache
    index = LatentCacheIndex.load(cache_dir)
    missing = missing_head_records(index)
    report = {"missing": len(missing), "built": 0, "cache_dir": str(cache_dir), "shard": config.head_cache_shard}
    if not missing:
        return report
    if dry_run:
        return report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = load_vae(config, device=device, dtype=dtype)

    shard_path = cache_dir / config.head_cache_shard
    if shard_path.exists():
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        lat_section = shard.setdefault("lat", {})
        shard.setdefault("cap", {})
        shard.setdefault("meta", {})
    else:
        shard = {"lat": {}, "cap": {}, "meta": {}}
        lat_section = shard["lat"]

    new_shard_idx = None
    for i, (name, *_rest) in enumerate(index.index.get("sig", [])):
        if name == config.head_cache_shard:
            new_shard_idx = i
            break
    if new_shard_idx is None:
        new_shard_idx = len(index.index.get("sig", []))

    for item in missing:
        image_path = resolve_image_path(item.record.path, image_root=config.image_root, source_prefix=config.image_source_prefix)
        cropped = crop_head_image(
            image_path,
            item.record.bucket,
            conf_threshold=config.head_crop_conf,
            padding=config.head_crop_padding,
        )
        lat_section[item.head_key] = encode_head_latent(vae, cropped, device=device, dtype=dtype)
        index.index["lat_idx"][item.head_key] = new_shard_idx
        path_kinds = index.index.setdefault("path_kinds", {})
        bucket_kinds = path_kinds.setdefault(item.record.path, {})
        bucket_kinds.setdefault(item.record.bucket, set()).add("head")
        meta = index._meta_for_path(item.record.path)
        meta["has_head"] = True
        report["built"] += 1

    torch.save(shard, shard_path)
    stat = shard_path.stat()
    sig = list(index.index.get("sig", []))
    sig_entry = (config.head_cache_shard, stat.st_mtime_ns, stat.st_size)
    if new_shard_idx == len(sig):
        sig.append(sig_entry)
    else:
        sig[new_shard_idx] = sig_entry
    index.index["sig"] = sig
    atomic_pickle_dump(index.index, cache_dir / "_cache_index.pkl")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run", action="store_true")
    known, rest = parser.parse_known_args(argv)
    config = parse_config(rest)
    report = build_missing_head_cache(config, dry_run=known.dry_run)
    print(report)
    if report["missing"] and known.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
