#!/usr/bin/env python3
"""Merge extra latent caches (built separately) into a base cache, in place.

Why not build_training_cache --resume on a combined metadata: the base cache's
final shard is partial (82/128), and resume's shard-aligned skip would silently
drop the first records that fall inside the "done" region. Offline merge avoids
touching the validated builder.

Usage:
  python merge_latcache.py <base_cache_dir> <extra_cache_dir> [<extra2> ...]

For each extra cache: shard files are copied in as shard_{N+i:03d}.pt continuing
the base numbering, and the base _cache_index.pkl gets the extra's lat_idx /
cap_idx / path_kinds / meta with shard indices offset. The original index is
backed up to _cache_index.pkl.bak.<n> first. Collisions on latent keys or meta
ids abort before anything is written.
"""
import pickle
import shutil
import sys
from pathlib import Path


def load_index(cache_dir: Path) -> dict:
    with (cache_dir / "_cache_index.pkl").open("rb") as fh:
        index = pickle.load(fh)
    if "lat_idx" not in index or "sig" not in index:
        raise ValueError(f"unsupported index format in {cache_dir}")
    return index


def merge(base_dir: Path, extra_dirs: list[Path]) -> None:
    base = load_index(base_dir)
    extras = [(d, load_index(d)) for d in extra_dirs]

    # --- preflight: verify no collisions before mutating anything -----------
    lat_keys = set(base["lat_idx"])
    meta_ids = set(base.get("meta", {}))
    for d, idx in extras:
        overlap = lat_keys & set(idx["lat_idx"])
        if overlap:
            raise SystemExit(f"ABORT: {len(overlap)} latent keys of {d} already in base, e.g. {next(iter(overlap))}")
        dup = meta_ids & set(idx.get("meta", {}))
        if dup:
            raise SystemExit(f"ABORT: {len(dup)} meta ids of {d} already in base, e.g. {next(iter(dup))}")
        lat_keys |= set(idx["lat_idx"])
        meta_ids |= set(idx.get("meta", {}))
        expected = [f"shard_{i:03d}.pt" for i in range(len(idx["sig"]))]
        actual = [name for name, *_ in idx["sig"]]
        if actual != expected:
            raise SystemExit(f"ABORT: {d} sig list is not contiguous shards: {actual[:3]}...")
        for name in actual:
            if not (d / name).exists():
                raise SystemExit(f"ABORT: missing shard file {d / name}")

    # --- backup ---------------------------------------------------------------
    n = 0
    while (base_dir / f"_cache_index.pkl.bak.{n}").exists():
        n += 1
    shutil.copy2(base_dir / "_cache_index.pkl", base_dir / f"_cache_index.pkl.bak.{n}")
    print(f"index backed up to _cache_index.pkl.bak.{n}")

    # --- merge ------------------------------------------------------------------
    offset = len(base["sig"])
    for d, idx in extras:
        print(f"merging {d} ({len(idx['sig'])} shards, {len(idx['lat_idx'])} latents) at offset {offset}")
        for i, (name, _, _) in enumerate(idx["sig"]):
            new_name = f"shard_{offset + i:03d}.pt"
            dst = base_dir / new_name
            if dst.exists():
                raise SystemExit(f"ABORT: target shard exists: {dst}")
            shutil.copy2(d / name, dst)
            st = dst.stat()
            base["sig"].append((new_name, st.st_mtime_ns, st.st_size))
        for key, sidx in idx["lat_idx"].items():
            base["lat_idx"][key] = sidx + offset
        for caption, sidx in idx.get("cap_idx", {}).items():
            base["cap_idx"].setdefault(caption, sidx + offset)
        for path, buckets in idx.get("path_kinds", {}).items():
            tgt = base["path_kinds"].setdefault(path, {})
            for bucket, kinds in buckets.items():
                tgt.setdefault(bucket, set()).update(kinds)
        base.setdefault("meta", {}).update(idx.get("meta", {}))
        offset += len(idx["sig"])

    tmp = base_dir / "_cache_index.pkl.tmp"
    with tmp.open("wb") as fh:
        pickle.dump(base, fh)
    tmp.replace(base_dir / "_cache_index.pkl")
    print(f"done: {len(base['sig'])} shards, {len(base['lat_idx'])} latents, "
          f"{len(base.get('meta', {}))} meta entries")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
