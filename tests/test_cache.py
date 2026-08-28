import pickle
from pathlib import Path

import torch

from anima_reflora.cache import LatentCacheDataset, LatentCacheIndex, SameBucketBatchSampler, collate_latent_batch
from anima_reflora.head_cache import missing_head_records


def write_cache(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    bucket = (16, 16)
    paths = ["/cache/images/1.webp", "/cache/images/2.webp"]
    captions = ["change-one", "change-two"]
    raw_captions = ["hair eyes raw one", "hair eyes raw two"]
    lat = {}
    cap = {}
    meta = {}
    lat_idx = {}
    cap_idx = {}
    path_kinds = {}
    for i, path in enumerate(paths):
        full_key = (path, bucket, "full")
        tensor = torch.full((1, 16, 1, *bucket), float(i + 1), dtype=torch.bfloat16)
        lat[full_key] = tensor
        lat_idx[full_key] = 0
        kinds = {"full"}
        if i == 0:
            head_key = (path, bucket, "head")
            lat[head_key] = tensor + 10
            lat_idx[head_key] = 0
            kinds.add("head")
        cap[captions[i]] = {
            "prompt_embeds": torch.zeros(1, 512, 1024, dtype=torch.bfloat16) + i,
            "attn_mask": torch.ones(1, 512, dtype=torch.int32),
            "t5_input_ids": torch.ones(1, 512, dtype=torch.int32),
            "t5_attn_mask": torch.ones(1, 512, dtype=torch.int32),
        }
        cap[raw_captions[i]] = {
            "prompt_embeds": torch.zeros(1, 512, 1024, dtype=torch.bfloat16) + 100 + i,
            "attn_mask": torch.ones(1, 512, dtype=torch.int32),
            "t5_input_ids": torch.ones(1, 512, dtype=torch.int32),
            "t5_attn_mask": torch.ones(1, 512, dtype=torch.int32),
        }
        cap_idx[captions[i]] = 0
        cap_idx[raw_captions[i]] = 0
        meta[i + 1] = {
            "path": path,
            "character": "char-a",
            "rating": "s",
            "ref_eligible": i == 0,
            "bucket": bucket,
            "orig_wh": (128, 128),
            "caption": raw_captions[i],
            "change_caption": captions[i],
            "raw_caption": raw_captions[i],
            "has_head": i == 0,
        }
        path_kinds[path] = {bucket: kinds}
    torch.save({"lat": lat, "cap": cap, "meta": meta}, root / "shard_000.pt")
    with (root / "_cache_index.pkl").open("wb") as fh:
        pickle.dump({"version": 3, "sig": [("shard_000.pt", 0, 0)], "lat_idx": lat_idx, "cap_idx": cap_idx, "path_kinds": path_kinds, "meta": meta}, fh)


def test_latent_cache_dataset_pairs_same_bucket(tmp_path):
    write_cache(tmp_path)
    dataset = LatentCacheDataset(tmp_path, frames=3)
    assert len(dataset) == 2
    item = dataset[1]
    assert item["frames"].shape == (16, 3, 16, 16)
    assert item["frame_roles"] == ("head_ref", "full_ref", "target")
    assert torch.all(item["frames"][:, 0] == 11)
    assert torch.all(item["frames"][:, 1] == 1)
    assert torch.all(item["frames"][:, 2] == 2)
    assert item["prompt_embeds"].shape == (512, 1024)
    assert item["t5_input_ids"].shape == (512,)
    assert item["t5_attn_mask"].shape == (512,)
    assert item["caption_key"] == "change-two"
    assert item["caption_source"] == "change_caption"
    assert item["bucket"] == (16, 16)
    batch = collate_latent_batch([dataset[0], dataset[1]])
    assert batch["clean"].shape == (2, 16, 3, 16, 16)
    assert batch["frame_roles"] == ("head_ref", "full_ref", "target")
    assert batch["t5_input_ids"].shape == (2, 512)
    assert batch["caption_sources"] == ["change_caption", "change_caption"]


def test_target_caption_prompt_mode_uses_raw_caption(tmp_path):
    write_cache(tmp_path)
    dataset = LatentCacheDataset(tmp_path, frames=3, prompt_mode="target_caption")
    item = dataset[1]
    assert item["caption_key"] == "hair eyes raw two"
    assert item["caption_source"] == "raw_caption"
    assert torch.all(item["prompt_embeds"] == 101)


def test_t3_requires_head_reference(tmp_path):
    write_cache(tmp_path)
    index_path = tmp_path / "_cache_index.pkl"
    with index_path.open("rb") as fh:
        index = pickle.load(fh)
    index["lat_idx"] = {key: value for key, value in index["lat_idx"].items() if key[2] != "head"}
    with index_path.open("wb") as fh:
        pickle.dump(index, fh)
    try:
        LatentCacheDataset(tmp_path, frames=3)
    except ValueError as exc:
        assert "head" in str(exc)
    else:
        raise AssertionError("T=3 dataset must reject caches without head reference latents")


def test_missing_head_records_detects_ref_eligible_only(tmp_path):
    write_cache(tmp_path)
    index_path = tmp_path / "_cache_index.pkl"
    with index_path.open("rb") as fh:
        index = pickle.load(fh)
    index["lat_idx"] = {key: value for key, value in index["lat_idx"].items() if key[2] != "head"}
    with index_path.open("wb") as fh:
        pickle.dump(index, fh)
    missing = missing_head_records(LatentCacheIndex.load(tmp_path))
    assert len(missing) == 1
    assert missing[0].record.ref_eligible is True


def test_same_bucket_batch_sampler(tmp_path):
    write_cache(tmp_path)
    dataset = LatentCacheDataset(tmp_path, frames=2)
    sampler = SameBucketBatchSampler(dataset.pairs, batch_size=2, shuffle=False, seed=1)
    assert list(sampler) == [[0, 1]]


def write_multi_shard_cache(root: Path, n_shards: int = 5):
    root.mkdir(parents=True, exist_ok=True)
    bucket = (16, 16)
    lat_idx = {}
    cap_idx = {}
    meta = {}
    path_kinds = {}
    sig = []
    for s in range(n_shards):
        path = f"/cache/images/{s + 1}.webp"
        full_key = (path, bucket, "full")
        caption = f"change-{s}"
        lat = {full_key: torch.full((1, 16, 1, *bucket), float(s + 1), dtype=torch.bfloat16)}
        cap = {
            caption: {
                "prompt_embeds": torch.zeros(1, 512, 1024, dtype=torch.bfloat16) + s,
                "attn_mask": torch.ones(1, 512, dtype=torch.int32),
                "t5_input_ids": torch.ones(1, 512, dtype=torch.int32),
                "t5_attn_mask": torch.ones(1, 512, dtype=torch.int32),
            }
        }
        lat_idx[full_key] = s
        cap_idx[caption] = s
        meta[s + 1] = {
            "path": path,
            "character": "char-a",
            "ref_eligible": True,
            "bucket": bucket,
            "caption": caption,
            "change_caption": caption,
            "has_head": False,
        }
        path_kinds[path] = {bucket: {"full"}}
        torch.save({"lat": lat, "cap": cap, "meta": {}}, root / f"shard_{s:03d}.pt")
        sig.append((f"shard_{s:03d}.pt", 0, 0))
    with (root / "_cache_index.pkl").open("wb") as fh:
        pickle.dump({"version": 3, "sig": sig, "lat_idx": lat_idx, "cap_idx": cap_idx, "path_kinds": path_kinds, "meta": meta}, fh)


def test_shard_cache_lru_eviction(tmp_path):
    write_multi_shard_cache(tmp_path, n_shards=5)
    index = LatentCacheIndex.load(tmp_path, max_cached_shards=2)
    recs = {r.shard_idx: r for r in index.records}
    for s in range(5):
        lat = index.load_latent(recs[s])
        assert torch.all(lat.float() == s + 1)
        assert len(index._shard_cache) <= 2
    assert set(index._shard_cache) == {3, 4}
    # recency: touching 3 makes 4 the LRU victim when 0 comes in
    index.load_latent(recs[3])
    index.load_latent(recs[0])
    assert set(index._shard_cache) == {3, 0}
    # evicted shards reload with correct contents
    assert torch.all(index.load_latent(recs[4]).float() == 5)
    cap = index.load_caption("change-1")
    assert torch.all(cap["prompt_embeds"].float() == 1)


def test_shard_cache_size_env_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_REFLORA_SHARD_CACHE", "1")
    write_multi_shard_cache(tmp_path, n_shards=3)
    index = LatentCacheIndex.load(tmp_path)
    assert index.max_cached_shards == 1
    recs = {r.shard_idx: r for r in index.records}
    index.load_latent(recs[0])
    index.load_latent(recs[1])
    assert set(index._shard_cache) == {1}


def test_dataset_items_survive_shard_eviction(tmp_path):
    write_multi_shard_cache(tmp_path, n_shards=4)
    dataset = LatentCacheDataset(tmp_path, frames=2, require_head_ref=False)
    dataset.index.max_cached_shards = 1
    items = [dataset[i] for i in range(len(dataset))]
    # stacked frames are real copies, not views into evicted mmap shards
    for item in items:
        assert item["frames"].is_contiguous()
        assert torch.all(item["frames"][:, 1] != 0)
