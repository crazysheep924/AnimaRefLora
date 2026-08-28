from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch
from PIL import Image, ImageOps

from .anima_caption import build_caption, build_signature, compute_bucket
from .config import parse_config
from .head_cache import IMAGE_EXTENSIONS, crop_head_image, image_to_tensor
from .sd_scripts_bridge import add_sd_scripts_to_path


DEFAULT_METADATA = "/path/to/dataset/index.parquet"
DEFAULT_IMAGE_ROOT = "/path/to/dataset/images"


@dataclass(frozen=True)
class RawImageRecord:
    image_id: int
    image_path: Path
    cache_path: str
    character: str
    rating: str
    score: int
    tag_string_general: str
    ref_eligible: bool
    captions: dict[str, str]


class PromptEncoder(Protocol):
    def encode(self, captions: list[str]) -> dict[str, dict[str, torch.Tensor]]:
        ...


def resolve_image_path(image_id: int, download_path: str, image_root: Path) -> Path:
    """Resolve the flat per-id image path: <image_root>/<id>.<ext>.

    Extension comes from the parquet download_path (which uses backslashes).
    """
    ext = Path(str(download_path).replace("\\", "/")).suffix or ".webp"
    candidate = image_root / f"{image_id}{ext}"
    if candidate.exists():
        return candidate
    # fall back to scanning known extensions
    for alt in IMAGE_EXTENSIONS:
        cand = image_root / f"{image_id}{alt}"
        if cand.exists():
            return cand
    return candidate


CAPTION_FIELDS = ("caption", "change_caption", "filtered_caption", "training_caption", "reflo_caption", "target_caption", "raw_caption")


def load_parquet_rows(metadata: Path):
    import pandas as pd

    suffix = metadata.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(metadata, lines=True)
    elif suffix == ".json":
        df = pd.read_json(metadata)
    elif suffix == ".csv":
        df = pd.read_csv(metadata)
    else:
        df = pd.read_parquet(metadata)
    for column, default in {
        "character": "unknown",
        "rating": "g",
        "score": 0,
        "tag_string_general": "",
        "ref_eligible": True,
    }.items():
        if column not in df:
            df[column] = default
        else:
            df[column] = df[column].fillna(default)
    return df


def row_value(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        if str(value) == "":
            continue
        return value
    return default


def records_from_parquet(
    df: Any,
    image_root: Path,
    characters: set[str] | None = None,
    image_ids: set[int] | None = None,
) -> list[RawImageRecord]:
    if characters is not None:
        df = df[df["character"].isin(characters)]
    if image_ids is not None:
        if "id" not in df:
            raise ValueError("image_ids filtering requires metadata with an id column")
        df = df[df["id"].isin(image_ids)]
    records: list[RawImageRecord] = []
    for row_index, row in enumerate(df.to_dict("records")):
        image_id = int(row_value(row, "id", default=row_index))
        path_value = row_value(row, "path", "image_path", "file", default="")
        if path_value:
            image_path = Path(str(path_value))
            if not image_path.is_absolute():
                image_path = image_root / image_path
        else:
            image_path = resolve_image_path(image_id, str(row_value(row, "download_path", default="")), image_root)
        captions = {field: str(row[field]) for field in CAPTION_FIELDS if row_value(row, field, default="")}
        records.append(
            RawImageRecord(
                image_id=image_id,
                image_path=image_path,
                cache_path=str(image_path),
                character=str(row_value(row, "character", default="unknown")),
                rating=str(row_value(row, "rating", default="g")),
                score=int(row_value(row, "score", default=0)),
                tag_string_general=str(row_value(row, "tag_string_general", default="")),
                ref_eligible=bool(row_value(row, "ref_eligible", default=True)),
                captions=captions,
            )
        )
    return records


# --- Resize helpers ----------------------------------------------------------
def resize_full_fill_crop(image: Image.Image, pixel_w: int, pixel_h: int) -> Image.Image:
    """FILL + center-crop: scale = max(bw/w, bh/h), resize, center-crop."""
    w, h = image.size
    scale = max(pixel_w / w, pixel_h / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - pixel_w) // 2
    top = (new_h - pixel_h) // 2
    return resized.crop((left, top, left + pixel_w, top + pixel_h))


def resize_head_letterbox(image: Image.Image, pixel_w: int, pixel_h: int) -> Image.Image:
    """CONTAIN + pad black (letterbox): scale = min, paste centered on black."""
    w, h = image.size
    scale = min(pixel_w / w, pixel_h / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (pixel_w, pixel_h), (0, 0, 0))
    canvas.paste(resized, ((pixel_w - new_w) // 2, (pixel_h - new_h) // 2))
    return canvas


def open_full_image(path: Path, pixel_w: int, pixel_h: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return resize_full_fill_crop(image, pixel_w, pixel_h)


def make_head_image(
    path: Path,
    pixel_w: int,
    pixel_h: int,
    conf_threshold: float,
    padding: float,
    detector: Any | None,
) -> Image.Image | None:
    if detector is None:
        from imgutils.detect import detect_heads

        detector = detect_heads
    detections = detector(str(path), conf_threshold=conf_threshold)
    if not detections:
        return None
    bbox, _label, _score = max(detections, key=lambda item: item[2])
    from .head_cache import expand_bbox

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        crop_box = expand_bbox(
            tuple(int(v) for v in bbox),
            image.size,
            target_aspect=pixel_w / pixel_h,
            padding=padding,
        )
        crop = image.crop(crop_box)
    return resize_head_letterbox(crop, pixel_w, pixel_h)


def encode_image_latent(vae: Any, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad():
        pixels = image_to_tensor(image, device=device, dtype=dtype)
        latent = vae.encode_pixels_to_latents(pixels)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return latent.to("cpu", dtype=torch.bfloat16)


def load_vae(config: Any, device: torch.device, dtype: torch.dtype) -> Any:
    add_sd_scripts_to_path(config)
    # Import the VAE module directly instead of anima_train_utils.load_qwen_image_vae:
    # that training wrapper drags in checkpoint_io -> model_util -> diffusers' SD
    # pipelines, which inference doesn't need and which break on some
    # transformers/diffusers version combinations (e.g. ComfyUI portable envs).
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


def qwen3_model_path(config: Any) -> Path:
    model_te = config.paths().model_te
    linked = model_te / "model.safetensors"
    return linked if linked.exists() else model_te / "qwen_3_06b_base.safetensors"


class SdScriptsPromptEncoder:
    def __init__(self, config: Any, device: torch.device, dtype: torch.dtype, batch_size: int = 8):
        add_sd_scripts_to_path(config)
        from library import anima_utils, strategy_anima

        qwen_path = str(qwen3_model_path(config))
        text_encoder, qwen_tokenizer = anima_utils.load_qwen3_text_encoder(qwen_path, dtype=dtype, device=str(device))
        text_encoder.eval()
        self.text_encoder = text_encoder
        self.tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_tokenizer=qwen_tokenizer,
            t5_tokenizer=anima_utils.load_t5_tokenizer(None),
        )
        self.text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        self.batch_size = max(1, int(batch_size))

    def encode(self, captions: list[str]) -> dict[str, dict[str, torch.Tensor]]:
        encoded: dict[str, dict[str, torch.Tensor]] = {}
        for start in range(0, len(captions), self.batch_size):
            batch = captions[start : start + self.batch_size]
            tokens = self.tokenize_strategy.tokenize(batch)
            with torch.no_grad():
                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = self.text_encoding_strategy.encode_tokens(
                    self.tokenize_strategy,
                    [self.text_encoder],
                    tokens,
                )
            for i, caption in enumerate(batch):
                encoded[caption] = {
                    "prompt_embeds": prompt_embeds[i : i + 1].detach().cpu().to(dtype=torch.bfloat16),
                    "attn_mask": attn_mask[i : i + 1].detach().cpu().to(dtype=torch.int32),
                    "t5_input_ids": t5_input_ids[i : i + 1].detach().cpu().to(dtype=torch.int32),
                    "t5_attn_mask": t5_attn_mask[i : i + 1].detach().cpu().to(dtype=torch.int32),
                }
        return encoded


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(path)


def chunked(items: list[RawImageRecord], size: int) -> Iterable[list[RawImageRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def save_shard(cache_dir: Path, shard_idx: int, lat: dict, cap: dict) -> tuple[str, int, int]:
    name = f"shard_{shard_idx:03d}.pt"
    path = cache_dir / name
    torch.save({"lat": lat, "cap": cap, "meta": {"source": "anima_training_cache"}}, path)
    stat = path.stat()
    return name, stat.st_mtime_ns, stat.st_size


def reconstruct_index_from_shards(
    cache_dir: Path,
    records: list[RawImageRecord],
    record_captions: dict[int, dict[str, str]],
    shard_size: int,
    index: dict,
) -> tuple[int, int, int]:
    """Rebuild index entries (lat_idx/cap_idx/path_kinds/sig/meta) from shards already
    on disk so an interrupted build can resume. Returns (n_done_shards, full, head).
    Latent tensors are not materialized (mmap); meta is recomputed from parquet.
    """
    shard_files = sorted(cache_dir.glob("shard_*.pt"))
    if not shard_files:
        return 0, 0, 0
    expected = [f"shard_{i:03d}.pt" for i in range(len(shard_files))]
    if [p.name for p in shard_files] != expected:
        raise RuntimeError(
            f"non-contiguous shards in {cache_dir}; cannot safely resume "
            f"(found {[p.name for p in shard_files][:3]}...)"
        )
    for shard_idx, shard_path in enumerate(shard_files):
        try:
            data = torch.load(shard_path, map_location="cpu", mmap=True)
        except Exception:
            data = torch.load(shard_path, map_location="cpu")
        for key in data["lat"].keys():
            path, bucket, kind = key
            bucket = tuple(bucket)
            index["lat_idx"][(path, bucket, kind)] = shard_idx
            index["path_kinds"].setdefault(path, {}).setdefault(bucket, set()).add(kind)
        for caption in data["cap"].keys():
            index["cap_idx"].setdefault(caption, shard_idx)
        st = shard_path.stat()
        index["sig"].append((shard_path.name, st.st_mtime_ns, st.st_size))
        del data
    n_done = len(shard_files)
    built_full = built_head = 0
    for record in records[: n_done * shard_size]:
        pk = index["path_kinds"].get(record.cache_path)
        if not pk:
            continue  # this record failed/was skipped in the prior run
        bucket = next(iter(pk))
        has_head = "head" in pk[bucket]
        built_full += 1
        built_head += int(has_head)
        # orig_wh is not consumed downstream (cache.py / ccip / head_roi builders ignore
        # it); derive it from the bucket to avoid re-opening 17k+ images on resume.
        meta = {
            "path": record.cache_path,
            "character": record.character,
            "rating": record.rating,
            "ref_eligible": bool(record.ref_eligible),
            "bucket": bucket,
            "orig_wh": (bucket[1] * 8, bucket[0] * 8),
            "has_head": has_head,
        }
        meta.update(record_captions[record.image_id])
        index["meta"][record.image_id] = meta
    return n_done, built_full, built_head


def build_training_cache(
    config: Any,
    *,
    image_root: str | Path | None = None,
    metadata: str | Path | None = None,
    output_cache: str | Path | None = None,
    shard_size: int = 128,
    max_items: int | None = None,
    skip_failed: bool = False,
    no_head: bool = False,
    no_prompts: bool = False,
    resume: bool = False,
    characters: set[str] | None = None,
    image_ids: set[int] | None = None,
    vae: Any | None = None,
    prompt_encoder: PromptEncoder | None = None,
    detector: Any | None = None,
) -> dict[str, Any]:
    root = Path(image_root or config.image_root or DEFAULT_IMAGE_ROOT)
    meta_path = Path(metadata or DEFAULT_METADATA)
    df = load_parquet_rows(meta_path)

    # signatures are computed over ALL parquet rows of the selected characters
    sig_df = df[df["character"].isin(characters)] if characters is not None else df
    signatures = build_signature(sig_df)

    records = records_from_parquet(df, root, characters=characters, image_ids=image_ids)
    if max_items is not None:
        records = records[:max_items]
    if not records:
        raise RuntimeError(f"No image records found for metadata {meta_path}")

    cache_dir = Path(output_cache or config.paths().latcache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    index = {"version": 3, "sig": [], "lat_idx": {}, "cap_idx": {}, "path_kinds": {}, "meta": {}}
    failures: dict[str, str] = {}
    built_full = 0
    built_head = 0

    # precompute captions for all records (deterministic, no VAE; strings only, cheap)
    record_captions: dict[int, dict[str, str]] = {}
    for record in records:
        captions = dict(record.captions)
        if not captions.get("caption"):
            captions["caption"] = build_caption(
                record.image_id,
                record.score,
                record.rating,
                record.tag_string_general,
                signatures.get(record.character, set()),
            )
        record_captions[record.image_id] = captions

    # resume: rebuild index from shards already on disk, then skip them in the loop.
    start_shard = 0
    if resume:
        start_shard, built_full, built_head = reconstruct_index_from_shards(
            cache_dir, records, record_captions, max(1, int(shard_size)), index
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = vae or load_vae(config, device=device, dtype=dtype)
    prompt_encoder = None if no_prompts else (prompt_encoder or SdScriptsPromptEncoder(config, device=device, dtype=dtype))

    index_path = cache_dir / "_cache_index.pkl"
    for shard_idx, group in enumerate(chunked(records, max(1, int(shard_size)))):
        if shard_idx < start_shard:
            continue
        lat: dict[tuple[str, tuple[int, int], str], torch.Tensor] = {}
        cap: dict[str, dict[str, torch.Tensor]] = {}
        # Encode prompts for THIS shard's NEW captions only — streaming keeps caption
        # embeddings bounded to one shard (avoids holding all ~54k in RAM at once,
        # which can exhaust WSL memory). Globally-seen captions live in cap_idx and are
        # loaded from their first shard, so re-encoding them here would be wasted — skip.
        if prompt_encoder is None:
            shard_prompt_cache: dict[str, dict[str, torch.Tensor]] = {}
        else:
            new_caps = sorted({
                caption
                for r in group
                for caption in record_captions[r.image_id].values()
                if caption not in index["cap_idx"]
            })
            shard_prompt_cache = prompt_encoder.encode(new_caps) if new_caps else {}
        for record in group:
            try:
                with Image.open(record.image_path) as source_image:
                    orig_wh = source_image.size
                pixel_w, pixel_h, latent_bucket = compute_bucket(*orig_wh)

                full_image = open_full_image(record.image_path, pixel_w, pixel_h)
                full_key = (record.cache_path, latent_bucket, "full")
                record_lat = {full_key: encode_image_latent(vae, full_image, device=device, dtype=dtype)}
                kinds = {"full"}

                has_head = False
                if not no_head:
                    head_image = make_head_image(
                        record.image_path,
                        pixel_w,
                        pixel_h,
                        conf_threshold=config.head_crop_conf,
                        padding=config.head_crop_padding,
                        detector=detector,
                    )
                    if head_image is not None:
                        head_key = (record.cache_path, latent_bucket, "head")
                        record_lat[head_key] = encode_image_latent(vae, head_image, device=device, dtype=dtype)
                        kinds.add("head")
                        has_head = True

                lat.update(record_lat)
                for key in record_lat:
                    index["lat_idx"][key] = shard_idx
                built_full += 1
                built_head += int(has_head)

                path_kinds = index["path_kinds"].setdefault(record.cache_path, {})
                path_kinds[latent_bucket] = kinds

                captions = record_captions[record.image_id]
                meta = {
                    "path": record.cache_path,
                    "character": record.character,
                    "rating": record.rating,
                    "ref_eligible": bool(record.ref_eligible),
                    "bucket": latent_bucket,
                    "orig_wh": orig_wh,
                    "has_head": has_head,
                }
                meta.update(captions)
                index["meta"][record.image_id] = meta
                for caption in captions.values():
                    if caption in shard_prompt_cache and caption not in index["cap_idx"]:
                        cap[caption] = shard_prompt_cache[caption]
                        index["cap_idx"][caption] = shard_idx
            except Exception as exc:
                if not skip_failed:
                    raise
                failures[record.cache_path] = str(exc)

        sig = save_shard(cache_dir, shard_idx, lat, cap)
        index["sig"].append(sig)
        # incremental index flush (every 10 shards) so a crash stays resumable cheaply
        if (shard_idx + 1) % 10 == 0:
            atomic_pickle_dump(index, index_path)

    atomic_pickle_dump(index, index_path)
    return {
        "output": str(cache_dir),
        "records": len(records),
        "built_full": built_full,
        "built_head": built_head,
        "captions": len(index["cap_idx"]),
        "failed": len(failures),
        "failed_examples": list(failures.items())[:10],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image-root")
    parser.add_argument("--metadata")
    parser.add_argument("--output-cache")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--no-head", action="store_true")
    parser.add_argument("--no-prompts", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="rebuild index from existing shard_*.pt and continue from the next shard")
    parser.add_argument("--json", action="store_true")
    known, rest = parser.parse_known_args(argv)
    report = build_training_cache(
        parse_config(rest),
        image_root=known.image_root,
        metadata=known.metadata,
        output_cache=known.output_cache,
        shard_size=known.shard_size,
        max_items=known.max_items,
        skip_failed=known.skip_failed,
        no_head=known.no_head,
        no_prompts=known.no_prompts,
        resume=known.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True) if known.json else report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
