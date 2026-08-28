#!/usr/bin/env python3
"""Build the pair-dhash cache used by 差分-aware ref pairing (cache.select_ref_candidate).

Computes a 16x16 (256-bit) dHash per dataset image and saves
{lowercase basename: packed uint8[32]} to a .pt. Runs where the ORIGINAL images
exist (the training pod's volume has latents only), so build locally and upload
to S3 runs/ next to the other caches:

    docker run --rm \
      -v /path/to/dataset:/data:ro \
      -v /path/to/anima-reflora/scripts:/s:ro \
      -v /path/to/storage/runs:/out \
      anima-reflora-runpod:latest \
      python /s/build_pair_dhash_cache.py --data /data --out /out/pair_dhash_cache.pt
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image


CROP_SCALES = (1.0, 0.8, 0.65)


def _hash_gray(g: Image.Image) -> np.ndarray:
    a = np.asarray(g.resize((17, 16), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).astype(np.uint8).flatten()
    return np.packbits(bits)


def dhash256(path: str) -> np.ndarray | None:
    """Multi-crop dHash: one 256-bit hash per center-crop scale, stacked (K, 32).

    Crop-type 差分 (one image is a zoom/crop of another) misalign a single
    whole-image hash; hashing several center crops lets the pair comparison
    take the min distance over crop combinations and still catch them.
    """
    try:
        with Image.open(path) as im:
            g = im.convert("L")
        w, h = g.size
        rows = []
        for s in CROP_SCALES:
            cw, ch = max(int(w * s), 1), max(int(h * s), 1)
            left, top = (w - cw) // 2, (h - ch) // 2
            rows.append(_hash_gray(g.crop((left, top, left + cw, top + ch))))
        return np.stack(rows)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data", help="dataset root containing index.parquet + images/")
    ap.add_argument("--out", default="/out/pair_dhash_cache.pt")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(args.data, "index.parquet"))
    names = [f"{row_id}.webp" for row_id in df["id"]]
    paths = [os.path.join(args.data, "images", n) for n in names]
    print(f"hashing {len(paths)} images...", flush=True)
    with Pool(processes=args.workers) as pool:
        hashes = pool.map(dhash256, paths, chunksize=256)

    out = {}
    missing = 0
    for name, h in zip(names, hashes):
        if h is None:
            missing += 1
            continue
        out[name.lower()] = torch.from_numpy(h.copy())
    payload = {"hashes": out, "bits": 256, "algo": "dhash16x16-multicrop", "crop_scales": list(CROP_SCALES)}
    torch.save(payload, args.out)
    print(f"saved {len(out)} hashes ({missing} unreadable) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
