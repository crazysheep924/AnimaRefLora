import random

import torch

from anima_reflora.cache import LatentRecord, dhash_key, select_ref_candidate


def rec(path: str) -> LatentRecord:
    return LatentRecord(
        path=path,
        bucket=(16, 16),
        kind="full",
        shard_idx=0,
        character="char-a",
        caption="cap",
        caption_source="caption",
        ref_eligible=True,
        has_head=True,
    )


def packed(bits_on: int) -> torch.Tensor:
    h = torch.zeros(32, dtype=torch.uint8)
    full_bytes, rest = divmod(bits_on, 8)
    if full_bytes:
        h[:full_bytes] = 0xFF
    if rest:
        h[full_bytes] = (0xFF << (8 - rest)) & 0xFF
    return h


def test_dhash_key_normalizes_prefixes():
    assert dhash_key("/path/to/dataset/images/123.webp") == "123.webp"
    assert dhash_key("images\\0542\\5422207.WEBP") == "5422207.webp"


def test_select_avoids_target_without_dhash():
    cands = (rec("a.webp"), rec("b.webp"))
    rng = random.Random(0)
    for _ in range(10):
        assert select_ref_candidate(cands, "a.webp", rng).path == "b.webp"


def test_select_excludes_near_duplicates():
    # target hash = all zeros; near.webp differs by 3 bits, far.webp by 128 bits
    dhash = {"t.webp": packed(0), "near.webp": packed(3), "far.webp": packed(128)}
    cands = (rec("near.webp"), rec("far.webp"))
    rng = random.Random(0)
    for _ in range(20):
        assert select_ref_candidate(cands, "t.webp", rng, dhash=dhash, min_dist=25).path == "far.webp"


def test_select_falls_back_to_farthest_when_all_near():
    dhash = {"t.webp": packed(0), "n3.webp": packed(3), "n10.webp": packed(10)}
    cands = (rec("n3.webp"), rec("n10.webp"))
    rng = random.Random(0)
    for _ in range(20):
        assert select_ref_candidate(cands, "t.webp", rng, dhash=dhash, min_dist=25).path == "n10.webp"


def test_select_unknown_hashes_stay_in_pool():
    dhash = {"t.webp": packed(0), "near.webp": packed(3)}
    cands = (rec("near.webp"), rec("unknown.webp"))
    rng = random.Random(0)
    picks = {select_ref_candidate(cands, "t.webp", rng, dhash=dhash, min_dist=25).path for _ in range(30)}
    # unknown-hash candidate is treated as acceptable, near-dup excluded
    assert picks == {"unknown.webp"}


def test_select_keeps_diversity_among_far_candidates():
    dhash = {"t.webp": packed(0), "f1.webp": packed(100), "f2.webp": packed(140)}
    cands = (rec("f1.webp"), rec("f2.webp"))
    rng = random.Random(0)
    picks = {select_ref_candidate(cands, "t.webp", rng, dhash=dhash, min_dist=25).path for _ in range(30)}
    assert picks == {"f1.webp", "f2.webp"}
