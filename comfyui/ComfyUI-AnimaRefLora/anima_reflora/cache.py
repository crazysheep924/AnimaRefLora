from __future__ import annotations

import os
import pickle
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, Sampler


LatentKey = tuple[str, tuple[int, int], str]
CHANGE_CAPTION_FIELDS = ("change_caption", "filtered_caption", "training_caption", "reflo_caption", "caption")
TARGET_CAPTION_FIELDS = ("target_caption", "raw_caption", "caption")


@dataclass(frozen=True)
class LatentRecord:
    path: str
    bucket: tuple[int, int]
    kind: str
    shard_idx: int
    character: str
    caption: str
    caption_source: str
    ref_eligible: bool
    has_head: bool

    @property
    def key(self) -> LatentKey:
        return (self.path, self.bucket, self.kind)


@dataclass(frozen=True)
class PairRecord:
    ref_full: LatentRecord
    target_full: LatentRecord
    ref_head: LatentRecord | None
    bucket: tuple[int, int]
    character: str
    # All eligible references for this (character, bucket) cell. The training
    # dataset resamples a ref from here per epoch (instead of a static i%n bind);
    # ref_full above is just the deterministic default used by eval/preflight.
    ref_candidates: tuple[LatentRecord, ...] = ()


class LatentCacheIndex:
    def __init__(
        self,
        cache_dir: Path,
        index: dict[str, Any],
        prompt_mode: str = "change_only",
        max_cached_shards: int | None = None,
    ):
        self.cache_dir = cache_dir
        self.index = index
        self.prompt_mode = prompt_mode
        self.shards = [name for name, *_ in index.get("sig", [])]
        self.records = self._build_records()
        if max_cached_shards is None:
            max_cached_shards = int(os.environ.get("ANIMA_REFLORA_SHARD_CACHE", "64"))
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._shard_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._mmap_ok = True

    @classmethod
    def load(
        cls,
        cache_dir: str | Path,
        prompt_mode: str = "change_only",
        max_cached_shards: int | None = None,
    ) -> "LatentCacheIndex":
        cache_dir = Path(cache_dir)
        index_path = cache_dir / "_cache_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Latent cache index not found: {index_path}")
        with index_path.open("rb") as fh:
            index = pickle.load(fh)
        if not isinstance(index, dict) or "lat_idx" not in index:
            raise ValueError(f"Unsupported latent cache index format: {index_path}")
        return cls(cache_dir, index, prompt_mode=prompt_mode, max_cached_shards=max_cached_shards)

    def _meta_for_path(self, path: str) -> dict[str, Any]:
        stem = Path(path).stem
        meta = self.index.get("meta", {})
        if stem.isdigit() and int(stem) in meta:
            return meta[int(stem)]
        if stem in meta:
            return meta[stem]
        for value in meta.values():
            if isinstance(value, dict) and value.get("path") == path:
                return value
        return {"path": path, "character": "unknown", "caption": "", "ref_eligible": False, "has_head": False}

    def _caption_for_meta(self, meta: dict[str, Any]) -> tuple[str, str]:
        fields = CHANGE_CAPTION_FIELDS if self.prompt_mode == "change_only" else TARGET_CAPTION_FIELDS
        cap_idx = self.index.get("cap_idx", {})
        fallback: tuple[str, str] | None = None
        for field in fields:
            value = str(meta.get(field) or "")
            if not value:
                continue
            if fallback is None:
                fallback = (value, field)
            if not cap_idx or value in cap_idx:
                return value, field
        return fallback or (str(meta.get("caption") or ""), "caption")

    def _build_records(self) -> list[LatentRecord]:
        records: list[LatentRecord] = []
        for key, shard_idx in self.index["lat_idx"].items():
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            path, bucket, kind = key
            meta = self._meta_for_path(path)
            caption, caption_source = self._caption_for_meta(meta)
            records.append(
                LatentRecord(
                    path=path,
                    bucket=tuple(bucket),
                    kind=str(kind),
                    shard_idx=int(shard_idx),
                    character=str(meta.get("character") or "unknown"),
                    caption=caption,
                    caption_source=caption_source,
                    ref_eligible=bool(meta.get("ref_eligible", False)),
                    has_head=bool(meta.get("has_head", False)),
                )
            )
        return records

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached
        if shard_idx < 0 or shard_idx >= len(self.shards):
            raise IndexError(f"Shard index out of range: {shard_idx}")
        path = self.cache_dir / self.shards[shard_idx]
        # mmap keeps latents on disk until touched: the resident cost per access is
        # one latent's pages (reclaimable page cache), not the whole ~0.3GB shard.
        # Without it a shuffled epoch over a 346GB/1084-shard cache OOMs the host.
        data = None
        if self._mmap_ok:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception as exc:
                self._mmap_ok = False
                print(f"shard mmap load failed ({exc}); falling back to full loads")
        if data is None:
            data = torch.load(path, map_location="cpu", weights_only=False)
        self._shard_cache[shard_idx] = data
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return data

    def load_latent(self, record: LatentRecord) -> torch.Tensor:
        shard = self._load_shard(record.shard_idx)
        return shard["lat"][record.key]

    def load_caption(self, caption: str) -> dict[str, torch.Tensor]:
        if caption not in self.index.get("cap_idx", {}):
            raise KeyError("Caption embedding missing from cache")
        shard_idx = int(self.index["cap_idx"][caption])
        shard = self._load_shard(shard_idx)
        return shard["cap"][caption]

    def records_by_key(self) -> dict[LatentKey, LatentRecord]:
        return {record.key: record for record in self.records}


def build_pairs(index: LatentCacheIndex, frames: int, require_head_ref: bool = True) -> list[PairRecord]:
    by_group: dict[tuple[str, tuple[int, int]], list[LatentRecord]] = defaultdict(list)
    by_key = index.records_by_key()
    for record in index.records:
        if record.kind == "full":
            by_group[(record.character, record.bucket)].append(record)

    pairs: list[PairRecord] = []
    for (character, bucket), full_records in sorted(by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        if not full_records:
            continue
        ref_candidates = [r for r in full_records if r.ref_eligible]
        if frames == 3:
            ref_candidates = [r for r in ref_candidates if (r.path, r.bucket, "head") in by_key]
            if require_head_ref and not ref_candidates:
                continue
        if not ref_candidates:
            ref_candidates = full_records
        candidates = tuple(ref_candidates)
        for i, target in enumerate(full_records):
            ref = candidates[i % len(candidates)]
            if len(candidates) > 1 and ref.path == target.path:
                ref = candidates[(i + 1) % len(candidates)]
            head = by_key.get((ref.path, ref.bucket, "head")) if frames == 3 else None
            if frames == 3 and require_head_ref and head is None:
                continue
            pairs.append(
                PairRecord(
                    ref_full=ref,
                    target_full=target,
                    ref_head=head,
                    bucket=bucket,
                    character=character,
                    ref_candidates=candidates,
                )
            )
    return pairs


def latent_to_chw(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 5:
        return latent[0, :, 0].contiguous()
    if latent.ndim == 4:
        return latent[:, 0].contiguous() if latent.shape[1] == 1 else latent[0].contiguous()
    if latent.ndim == 3:
        return latent.contiguous()
    raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")


_POPCOUNT = None


def dhash_key(path: str) -> str:
    """Normalize an image path to the dhash-cache key: lowercase basename.

    Latcache records may carry host-specific prefixes (and Windows backslashes),
    so the cache is keyed by filename only — filenames are danbooru ids, unique
    across the dataset.
    """
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()


def load_pair_dhash_cache(path: str | Path | None) -> dict[str, torch.Tensor] | None:
    """Load {dhash_key: packed uint8 hash} built by build_pair_dhash_cache.py.

    Missing/unset path is not an error: pair selection falls back to plain
    epoch-shuffled sampling (the pre-dhash behavior).
    """
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        print(f"pair-dhash cache not found (falling back to random ref pairing): {path}")
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    hashes = payload.get("hashes", payload) if isinstance(payload, dict) else payload
    print(f"pair-dhash cache loaded: {len(hashes)} entries from {path}")
    return hashes


def _hamming(a: torch.Tensor, b: torch.Tensor) -> int:
    """Min hamming distance over all crop-hash combinations.

    Hashes are (32,) for the legacy single-crop cache or (K, 32) for the
    multi-crop cache; the min over crop pairs catches zoom/crop-type 差分 that
    a single whole-image hash misaligns on.
    """
    global _POPCOUNT
    if _POPCOUNT is None:
        _POPCOUNT = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32)
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    xor = torch.bitwise_xor(a2.unsqueeze(1), b2.unsqueeze(0))  # (Ka, Kb, 32)
    dists = _POPCOUNT[xor.long()].sum(dim=-1)
    return int(dists.min())


def select_ref_candidate(
    cands: tuple[LatentRecord, ...],
    target_path: str,
    rng: random.Random,
    dhash: dict[str, torch.Tensor] | None = None,
    min_dist: int = 25,
) -> LatentRecord:
    """Pick a ref among candidates: never the target itself when avoidable, and
    with dhash available never a near-duplicate 差分 of the target (distance
    below min_dist) when a genuinely different candidate exists. Uniform among
    the surviving pool keeps per-epoch pairing diversity."""
    pool = [c for c in cands if c.path != target_path]
    if not pool:
        return cands[rng.randrange(len(cands))]
    if dhash is not None:
        target_hash = dhash.get(dhash_key(target_path))
        if target_hash is not None:
            dists = {}
            for c in pool:
                h = dhash.get(dhash_key(c.path))
                if h is not None:
                    dists[c.path] = _hamming(target_hash, h)
            far = [c for c in pool if dists.get(c.path, min_dist) >= min_dist]
            if far:
                pool = far
            elif dists:
                # every candidate is a near-duplicate: least-bad = farthest
                best = max(dists.values())
                pool = [c for c in pool if dists.get(c.path) == best]
    return pool[rng.randrange(len(pool))]


class LatentCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        frames: int = 3,
        max_items: int | None = None,
        require_head_ref: bool = True,
        prompt_mode: str = "change_only",
        seed: int = 1234,
        pair_dhash_cache: str | Path | None = None,
        pair_min_dhash: int = 25,
        singleton_ref_mode: str = "self",
    ):
        self.index = LatentCacheIndex.load(cache_dir, prompt_mode=prompt_mode)
        self.frames = frames
        self.require_head_ref = require_head_ref
        self.seed = int(seed)
        self.pairs = build_pairs(self.index, frames=frames, require_head_ref=require_head_ref)
        if max_items is not None:
            self.pairs = self.pairs[:max_items]
        if not self.pairs:
            raise ValueError(
                f"No trainable same-bucket reference pairs found in latent cache: {cache_dir}. "
                "For T=3 training the cache must contain both ('head') and ('full') latents for reference images."
            )
        self._by_key = self.index.records_by_key()
        self._access = 0
        self.pair_min_dhash = int(pair_min_dhash)
        self._dhash = load_pair_dhash_cache(pair_dhash_cache)
        if singleton_ref_mode not in {"self", "blank"}:
            raise ValueError(f"Unsupported singleton_ref_mode: {singleton_ref_mode}")
        self.singleton_ref_mode = singleton_ref_mode
        if singleton_ref_mode == "blank":
            n_single = sum(1 for p in self.pairs if p.ref_candidates and len(p.ref_candidates) == 1
                           and p.ref_candidates[0].path == p.target_full.path)
            print(f"singleton-ref-mode=blank: {n_single}/{len(self.pairs)} samples "
                  f"({100 * n_single / max(len(self.pairs), 1):.1f}%) will train with blanked refs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _choose_ref(self, pair: PairRecord, idx: int) -> LatentRecord:
        """Pick a reference for this target, resampled per epoch (seeded).

        Breaks the static i%n ref->target binding so the model sees varied
        ref/target pairings across epochs (the real ref-binding signal). With a
        single candidate the choice is deterministic; ref==target is only allowed
        when the whole cell shares one path.

        With a pair-dhash cache, near-duplicate 差分 candidates (hamming distance
        to the target below pair_min_dhash) are excluded — such pairs are
        effectively self-pairs and teach the copy reflex.
        """
        cands = pair.ref_candidates or (pair.ref_full,)
        if len(cands) <= 1:
            return cands[0]
        epoch = self._access // max(len(self.pairs), 1)
        rng = random.Random(self.seed * 1_000_003 + epoch * 10_007 + idx)
        return select_ref_candidate(
            cands,
            pair.target_full.path,
            rng,
            dhash=self._dhash,
            min_dist=self.pair_min_dhash,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        ref_rec = self._choose_ref(pair, idx)
        self._access += 1
        ref_full = latent_to_chw(self.index.load_latent(ref_rec))
        target = latent_to_chw(self.index.load_latent(pair.target_full))
        if self.frames == 2:
            frames = [ref_full, target]
            head_available = False
        else:
            ref_head_rec = self._by_key.get((ref_rec.path, ref_rec.bucket, "head"))
            ref_head = latent_to_chw(self.index.load_latent(ref_head_rec))
            head_available = True
            frames = [ref_head, ref_full, target]
        cap = self.index.load_caption(pair.target_full.caption)
        return {
            "frames": torch.stack(frames, dim=1),
            "frame_roles": ("head_ref", "full_ref", "target") if self.frames == 3 else ("full_ref", "target"),
            "prompt_embeds": cap["prompt_embeds"][0],
            "attn_mask": cap.get("attn_mask", cap.get("t5_attn_mask"))[0],
            "t5_input_ids": cap.get("t5_input_ids", torch.zeros_like(cap.get("attn_mask", cap.get("t5_attn_mask"))))[0],
            "t5_attn_mask": cap.get("t5_attn_mask", cap.get("attn_mask"))[0],
            "target_path": pair.target_full.path,
            "ref_path": ref_rec.path,
            "caption_key": pair.target_full.caption,
            "caption_source": pair.target_full.caption_source,
            "bucket": pair.bucket,
            "character": pair.character,
            "head_available": head_available,
            # singleton (character,bucket) cells have no ref other than the target
            # itself; 'blank' mode routes them into the ref-dropout path so they
            # train the unconditional branch instead of teaching ref->target copy.
            "force_ref_blank": self.singleton_ref_mode == "blank" and ref_rec.path == pair.target_full.path,
        }


class SyntheticLatentDataset(Dataset):
    def __init__(self, length: int = 16, frames: int = 3, channels: int = 16, size: tuple[int, int] = (16, 16), seed: int = 1234):
        self.length = length
        self.frames = frames
        self.channels = channels
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        clean = torch.randn(self.channels, self.frames, *self.size, generator=gen, dtype=torch.float32)
        clean[:, -1] = clean[:, 0] * 0.25 + clean[:, -1] * 0.75
        prompt = torch.randn(512, 1024, generator=gen, dtype=torch.float32) * 0.01
        return {
            "frames": clean,
            "frame_roles": ("head_ref", "full_ref", "target") if self.frames == 3 else ("full_ref", "target"),
            "prompt_embeds": prompt,
            "attn_mask": torch.ones(512, dtype=torch.int32),
            "t5_input_ids": torch.ones(512, dtype=torch.int32),
            "t5_attn_mask": torch.ones(512, dtype=torch.int32),
            "target_path": f"synthetic/target_{idx}.latent",
            "ref_path": f"synthetic/ref_{idx}.latent",
            "caption_key": f"synthetic/change_only_{idx}",
            "caption_source": "synthetic",
            "bucket": self.size,
            "character": f"synthetic_{idx % 2}",
            "head_available": self.frames == 3,
            "force_ref_blank": False,
        }


def collate_latent_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {tuple(item["bucket"]) for item in items}
    if len(buckets) != 1:
        raise ValueError(f"Batch mixes latent buckets: {sorted(buckets)}")
    return {
        "clean": torch.stack([item["frames"] for item in items], dim=0),
        "prompt_embeds": torch.stack([item["prompt_embeds"] for item in items], dim=0),
        "attn_mask": torch.stack([item["attn_mask"] for item in items], dim=0),
        "t5_input_ids": torch.stack([item["t5_input_ids"] for item in items], dim=0),
        "t5_attn_mask": torch.stack([item["t5_attn_mask"] for item in items], dim=0),
        "target_paths": [item["target_path"] for item in items],
        "ref_paths": [item["ref_path"] for item in items],
        "caption_keys": [item["caption_key"] for item in items],
        "caption_sources": [item["caption_source"] for item in items],
        "frame_roles": items[0]["frame_roles"],
        "bucket": items[0]["bucket"],
        "characters": [item["character"] for item in items],
        "head_available": torch.tensor([bool(item["head_available"]) for item in items], dtype=torch.bool),
        "force_ref_blank": torch.tensor([bool(item.get("force_ref_blank", False)) for item in items], dtype=torch.bool),
    }


class SameBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, pairs: Iterable[PairRecord], batch_size: int, shuffle: bool, seed: int):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.by_bucket: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, pair in enumerate(pairs):
            self.by_bucket[pair.bucket].append(idx)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        batches: list[list[int]] = []
        for indices in self.by_bucket.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if batch:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.by_bucket.values())
