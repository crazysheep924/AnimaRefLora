from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .cache import LatentCacheIndex, LatentRecord
from .config import parse_config
from .head_cache import crop_head_image, resolve_image_path


def reference_full_records(index: LatentCacheIndex) -> list[LatentRecord]:
    chosen: dict[str, LatentRecord] = {}
    for record in index.records:
        if record.kind != "full" or not record.ref_eligible:
            continue
        current = chosen.get(record.path)
        if current is None or record.bucket[0] * record.bucket[1] > current.bucket[0] * current.bucket[1]:
            chosen[record.path] = record
    return sorted(chosen.values(), key=lambda r: r.path)


def _chunks(items: list[LatentRecord], size: int) -> Iterable[list[LatentRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalise(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float().flatten(), dim=0).cpu()


def build_head_ccip_cache(
    config,
    output: str | Path | None = None,
    prototype_by: str = "character",
    batch_size: int = 16,
    ccip_size: int = 384,
    ccip_model: str = "ccip-caformer-24-randaug-pruned",
    skip_failed: bool = False,
    detector: Callable | None = None,
    encoder: Callable | None = None,
) -> dict:
    if prototype_by not in {"path", "character"}:
        raise ValueError(f"Unsupported prototype mode: {prototype_by}")
    if encoder is None:
        from imgutils.metrics.ccip import ccip_batch_extract_features

        encoder = ccip_batch_extract_features

    index = LatentCacheIndex.load(config.paths().latcache)
    records = reference_full_records(index)
    out_path = Path(output or (config.paths().ccip_cache.parent / "ccip_ref_head_emb_cache.pt"))
    features: dict[str, torch.Tensor] = {}
    characters: dict[str, str] = {}
    failed: dict[str, str] = {}

    for batch_records in _chunks(records, max(1, batch_size)):
        crops = []
        ok_records = []
        for record in batch_records:
            try:
                image_path = resolve_image_path(record.path, image_root=config.image_root, source_prefix=config.image_source_prefix)
                crop = crop_head_image(
                    image_path,
                    record.bucket,
                    conf_threshold=config.head_crop_conf,
                    padding=config.head_crop_padding,
                    detector=detector,
                )
            except Exception as exc:
                if not skip_failed:
                    raise
                failed[record.path] = str(exc)
                continue
            crops.append(crop)
            ok_records.append(record)
        if not crops:
            continue
        encoded = list(encoder(crops, size=ccip_size, model=ccip_model))
        if len(encoded) != len(ok_records):
            raise RuntimeError(f"CCIP encoder returned {len(encoded)} features for {len(ok_records)} crops")
        for record, feature in zip(ok_records, encoded):
            features[record.path] = _normalise(torch.as_tensor(feature))
            characters[record.path] = record.character

    if prototype_by == "character":
        def prototype_key(path: str) -> str:
            character = characters.get(path, "unknown")
            return path if character == "unknown" else character

        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
        for path, feature in features.items():
            grouped[prototype_key(path)].append(feature)
        prototypes = {key: _normalise(torch.stack(values).mean(dim=0)) for key, values in grouped.items()}
        path_to_emb = {path: prototypes[prototype_key(path)] for path in features}
    else:
        path_to_emb = features

    if not path_to_emb:
        raise RuntimeError("No head-crop CCIP embeddings were built")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "path_to_emb": path_to_emb,
            "meta": {
                "source": "head_crop_ccip",
                "prototype_by": prototype_by,
                "ccip_model": ccip_model,
                "ccip_size": ccip_size,
                "head_crop_padding": config.head_crop_padding,
                "head_crop_conf": config.head_crop_conf,
                "unknown_characters_are_per_path": prototype_by == "character",
                "records": len(records),
                "built": len(path_to_emb),
                "failed": failed,
            },
        },
        out_path,
    )
    return {"output": str(out_path), "records": len(records), "built": len(path_to_emb), "failed": len(failed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output")
    parser.add_argument("--prototype-by", choices=["character", "path"], default="character")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ccip-size", type=int, default=384)
    parser.add_argument("--ccip-model", default="ccip-caformer-24-randaug-pruned")
    parser.add_argument("--skip-failed", action="store_true")
    known, rest = parser.parse_known_args(argv)
    config = parse_config(rest)
    report = build_head_ccip_cache(
        config,
        output=known.output,
        prototype_by=known.prototype_by,
        batch_size=known.batch_size,
        ccip_size=known.ccip_size,
        ccip_model=known.ccip_model,
        skip_failed=known.skip_failed,
    )
    print(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
