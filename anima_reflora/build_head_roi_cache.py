from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import LatentCacheIndex
from .config import parse_config
from .head_cache import resolve_image_path
from .head_detect import detect_head_box
from .head_geom import box_to_latent_mask
from .head_roi_mask import save_roi_cache


def build_head_roi_cache(config, *, output: str | Path | None = None, skip_failed: bool = False) -> dict:
    index = LatentCacheIndex.load(config.paths().latcache, prompt_mode=config.prompt_mode)
    masks = {}
    failed = {}
    for record in index.records:
        if record.kind != "full":
            continue
        try:
            image_path = resolve_image_path(record.path, image_root=config.image_root, source_prefix=config.image_source_prefix)
            box, image_size = detect_head_box(image_path, conf=getattr(config, "head_conf", config.head_crop_conf))
            masks[record.path] = box_to_latent_mask(
                box,
                image_size,
                record.bucket,
                margin=getattr(config, "head_margin", 0.6),
                min_lat=getattr(config, "head_loss_min_lat", 24),
                max_lat=getattr(config, "head_loss_max_lat", 64),
            )
        except Exception as exc:
            if not skip_failed:
                raise
            failed[record.path] = str(exc)
    if not masks:
        raise RuntimeError("No target head ROI masks were built")
    out = save_roi_cache(
        output or config.head_roi_cache,
        masks,
        meta={"records": len([r for r in index.records if r.kind == "full"]), "built": len(masks), "failed": failed},
    )
    return {"output": str(out), "built": len(masks), "failed": len(failed), "failed_examples": list(failed.items())[:10]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--json", action="store_true")
    known, rest = parser.parse_known_args(argv)
    report = build_head_roi_cache(parse_config(rest), output=known.output, skip_failed=known.skip_failed)
    print(json.dumps(report, indent=2, sort_keys=True) if known.json else report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
