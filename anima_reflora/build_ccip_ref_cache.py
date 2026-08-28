from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from .cache import LatentCacheIndex, LatentRecord
from .config import parse_config
from .head_cache import resolve_image_path


def reference_records(index: LatentCacheIndex) -> list[LatentRecord]:
    by_path: dict[str, LatentRecord] = {}
    for record in index.records:
        if record.kind != "full" or not record.ref_eligible:
            continue
        current = by_path.get(record.path)
        if current is None or record.bucket[0] * record.bucket[1] > current.bucket[0] * current.bucket[1]:
            by_path[record.path] = record
    return sorted(by_path.values(), key=lambda item: item.path)


def chunks(items: list[LatentRecord], size: int) -> Iterable[list[LatentRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_feature(value) -> torch.Tensor:
    return F.normalize(torch.as_tensor(value).float().flatten(), dim=0).cpu()


def build_ccip_ref_cache(
    config,
    *,
    output: str | Path | None = None,
    batch_size: int = 16,
    ccip_size: int = 384,
    ccip_model: str = "ccip-caformer-24-randaug-pruned",
    skip_failed: bool = False,
    encoder=None,
) -> dict:
    if encoder is None:
        from imgutils.metrics.ccip import ccip_batch_extract_features

        encoder = ccip_batch_extract_features
    index = LatentCacheIndex.load(config.paths().latcache, prompt_mode=config.prompt_mode)
    records = reference_records(index)
    path_to_emb = {}
    failed = {}
    for group in chunks(records, max(1, int(batch_size))):
        images = []
        ok = []
        for record in group:
            try:
                path = resolve_image_path(record.path, image_root=config.image_root, source_prefix=config.image_source_prefix)
                image = Image.open(path).convert("RGB")
            except Exception as exc:
                if not skip_failed:
                    raise
                failed[record.path] = str(exc)
                continue
            images.append(image)
            ok.append(record)
        if not images:
            continue
        try:
            features = list(encoder(images, size=ccip_size, model=ccip_model))
        finally:
            for image in images:
                image.close()
        if len(features) != len(ok):
            raise RuntimeError(f"CCIP encoder returned {len(features)} features for {len(ok)} images")
        for record, feature in zip(ok, features):
            path_to_emb[record.path] = normalize_feature(feature)
    if not path_to_emb:
        raise RuntimeError("No full-reference CCIP embeddings were built")
    out = Path(output or config.paths().ccip_cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "path_to_emb": path_to_emb,
            "meta": {
                "source": "full_image_ccip",
                "ccip_model": ccip_model,
                "ccip_size": ccip_size,
                "records": len(records),
                "built": len(path_to_emb),
                "failed": failed,
            },
        },
        out,
    )
    return {"output": str(out), "records": len(records), "built": len(path_to_emb), "failed": len(failed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ccip-size", type=int, default=384)
    parser.add_argument("--ccip-model", default="ccip-caformer-24-randaug-pruned")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--json", action="store_true")
    known, rest = parser.parse_known_args(argv)
    report = build_ccip_ref_cache(
        parse_config(rest),
        output=known.output,
        batch_size=known.batch_size,
        ccip_size=known.ccip_size,
        ccip_model=known.ccip_model,
        skip_failed=known.skip_failed,
    )
    print(json.dumps(report, indent=2, sort_keys=True) if known.json else report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
