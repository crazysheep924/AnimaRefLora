"""Anima caption + aspect-ratio bucket construction.

Pure, dependency-light helpers used by build_training_cache.py to reproduce the
Anima training latcache byte-exact. Kept free of torch/VAE imports so the
metadata side (captions, buckets, ratings) can be computed and verified without
loading any heavy model.
"""
from __future__ import annotations

import bisect
import random
import re
from typing import Iterable

# --- Bucketing ---------------------------------------------------------------
BUCKET_SHORT = 1024
BUCKET_LONG_MAX = 1536
BUCKET_STEP = 64


def compute_bucket(w: int, h: int) -> tuple[int, int, tuple[int, int]]:
    """Aspect-ratio bucket (no stretch). Returns (pixel_w, pixel_h, (latH, latW))."""
    long_side = max(w, h)
    short_side = min(w, h)
    aspect = long_side / short_side
    long_px = round(BUCKET_SHORT * aspect / BUCKET_STEP) * BUCKET_STEP
    long_px = min(long_px, BUCKET_LONG_MAX)
    long_px = max(long_px, BUCKET_SHORT)
    if w >= h:  # landscape
        bw, bh = long_px, BUCKET_SHORT
    else:  # portrait
        bw, bh = BUCKET_SHORT, long_px
    return bw, bh, (bh // 8, bw // 8)


# --- Caption -----------------------------------------------------------------
QUALITY_PREFIX = "masterpiece, best quality"
SCORE_FLOOR = 4
SCORE_THRESHOLDS = [91, 113, 138, 173, 237]
RATING_TAGS = {"g": "safe", "s": "sensitive", "q": "nsfw", "e": "explicit"}

YEAR_ANCHORS = [
    (2007, 5000),
    (2012, 1_150_000),
    (2015, 1_900_000),
    (2018, 3_000_000),
    (2020, 3_700_000),
    (2021, 4_300_000),
    (2022, 5_000_000),
    (2023, 5_900_000),
    (2024, 7_000_000),
    (2025, 8_600_000),
    (2026, 10_400_000),
]

RECENCY_THRESHOLDS = [
    (2024, "newest"),
    (2022, "recent"),
    (2020, "mid"),
    (2018, "early"),
    (0, "old"),
]

SIGNATURE_THRESHOLD = 0.45

BLACKLIST_EXACT = {
    "long_hair", "short_hair", "very_long_hair", "medium_hair",
    "absurdly_long_hair", "bald", "pale_skin", "tan", "tanned",
}
# HARD blacklist: fixed identity marks that ALWAYS drop, even if a whitelist
# substring would otherwise rescue them (e.g. `pubic_tattoo` must not survive via
# the explicit `pubic` whitelist). Checked before the whitelist in sig_subtract.
HARD_BLACKLIST_SUBSTR = [
    "tattoo", "mole", "freckle", "birthmark", "pointy_ears",
]
BLACKLIST_SUBSTR = [
    "_hair", "hair_", "_eyes", "eye_color", "skin", "dark-skinned",
    "watermark", "signature", "artist_name", "username", "web_address",
    "_bangs", "hairband",
    # --- NSFW-tuned additions (diverges from 100-char byte-exact): hairstyle / eye
    #     conditions that are fixed identity (yield to whitelist, like _hair/_eyes). ---
    "sidelocks", "heterochromia",
]
WHITELIST_EXACT = {
    "1girl", "solo", "multiple_girls", "1boy", "2girls",
    "multiple_boys", "solo_focus",
    # --- NSFW-tuned: body-form bare forms (controllable target attributes) ---
    "ass", "huge_ass", "big_ass", "large_ass", "small_ass", "flat_ass", "hips",
}
WHITELIST_SUBSTR = [
    "standing", "sitting", "lying", "kneeling", "squatting", "pose",
    "looking", "smile", "grin", "blush", "crying", "open_mouth",
    "closed_eyes", "frown", "expression", "spread", "arms_up", "leaning",
    "bent_over", "all_fours", "on_back", "on_side", "from_behind",
    "from_above", "from_below", "breasts", "chest", "cleavage", "flat_chest",
    "underboob", "sideboob", "wide_hips", "thighs", "thick", "curvy",
    "plump", "petite", "muscular", "abs", "navel", "midriff", "fat", "slim",
    # --- NSFW-tuned additions: clothing / accessories MUST stay prompt-controllable
    #     for a costume-change LoRA (whitelist now takes priority over blacklist). ---
    "dress", "skirt", "shirt", "leotard", "thighhigh", "pantyhose", "swimsuit",
    "bikini", "bodysuit", "lingerie", "panties", "underwear", "glove", "choker",
    "collar", "ribbon", "jacket", "coat", "uniform", "kimono", "gown", "corset",
    "garter", "harness", "earring", "jewelry", "necklace", "bracelet", "frills",
    "sleeve", "detached_", "legwear", "socks", "boots", "shoes", "heels", "scarf",
    "apron", "cape", "hood", "veil", "leggings", "kneehigh", "armband", "wristband",
    "headband", "bra", "clothes", "costume", "outfit", "pasties", "bandeau",
    "suspenders", "halter",
    # --- NSFW-tuned: explicit content tags (controllable; pubic_hair overrides
    #     the _hair blacklist now that whitelist is checked first). ---
    "pubic_hair", "pubic", "nipple", "pussy", "penis", "areola", "vaginal", "anal",
    "censored", "uncensored", "nude", "naked", "topless", "bottomless", "clitoris",
    "testicle", "erection", "ejaculat", "fellatio", "cunnilingus", "penetration",
    "orgasm", "cum", "_sex", "sex_", "vagina", "cameltoe", "ass_visible",
]


def score_to_n(score: float) -> int:
    return min(9, SCORE_FLOOR + bisect.bisect_right(SCORE_THRESHOLDS, float(score)))


def id_to_year(image_id: int) -> int:
    x = float(image_id)
    anchors = YEAR_ANCHORS
    if x <= anchors[0][1]:
        year = anchors[0][0]
    elif x >= anchors[-1][1]:
        year = anchors[-1][0]
    else:
        year = anchors[-1][0]
        for (y0, id0), (y1, id1) in zip(anchors, anchors[1:]):
            if id0 <= x <= id1:
                frac = (x - id0) / (id1 - id0)
                year = y0 + frac * (y1 - y0)
                break
    year = round(year)
    year = max(2007, min(2026, year))
    return int(year)


def year_to_recency(year: int) -> str:
    for threshold, tag in RECENCY_THRESHOLDS:
        if threshold <= year:
            return tag
    return RECENCY_THRESHOLDS[-1][1]


def _is_hard_blacklisted(tag: str) -> bool:
    return any(s in tag for s in HARD_BLACKLIST_SUBSTR)


def _is_blacklisted(tag: str) -> bool:
    if tag in BLACKLIST_EXACT:
        return True
    return any(s in tag for s in BLACKLIST_SUBSTR)


def _is_whitelisted(tag: str) -> bool:
    if tag in WHITELIST_EXACT:
        return True
    return any(s in tag for s in WHITELIST_SUBSTR)


def sig_subtract(tags: list[str], signature: set[str]) -> str:
    kept: list[str] = []
    for tag in tags:
        # hard blacklist (fixed identity marks) wins over everything, incl. whitelist.
        if _is_hard_blacklisted(tag):
            continue
        # whitelist priority: controllable pose/body/clothing/explicit tags survive even
        # if they match a (soft) blacklist substring (pubic_hair vs _hair, closed_eyes
        # vs _eyes) or are character-signature.
        if _is_whitelisted(tag):
            kept.append(tag)
            continue
        if _is_blacklisted(tag):
            continue
        if tag in signature:
            continue
        kept.append(tag)
    # dedupe preserving first occurrence
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in kept:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return ", ".join(t.replace("_", " ") for t in deduped)


def build_caption(image_id: int, score, rating: str, tag_string_general: str,
                  signature: set[str]) -> str:
    score_n = score_to_n(score)
    year = id_to_year(image_id)
    recency = year_to_recency(year)
    parts = [QUALITY_PREFIX, f"score_{score_n}", recency]
    rating_tag = RATING_TAGS.get(str(rating), "")
    if rating_tag:
        parts.append(rating_tag)
    tags = str(tag_string_general).split()
    sub_tags = sig_subtract(tags, signature)
    if sub_tags:
        parts.append(sub_tags)
    parts.append(f"year{year}")
    return ", ".join(parts)


# Tags in a built caption that encode fixed structure/quality (not controllable
# delta) and must never be dropped by tag-dropout. Compared against the space-form
# tag text (sig_subtract already replaces "_" with " ").
_RECENCY_WORDS = {label for _, label in RECENCY_THRESHOLDS}
_RATING_WORDS = set(RATING_TAGS.values())
_STRUCTURAL_EXACT = {"masterpiece", "best quality"} | _RECENCY_WORDS | _RATING_WORDS
_YEAR_RE = re.compile(r"^year\d+$")
# Subject / character-count anchors: dropping these makes the caption ambiguous
# about who/how many, so they are always kept regardless of tag-dropout.
_SUBJECT_ANCHORS = {
    "1girl", "solo", "multiple girls", "1boy", "2girls", "multiple boys", "solo focus",
}


def _is_structural_tag(tag: str) -> bool:
    return (
        tag in _STRUCTURAL_EXACT
        or tag in _SUBJECT_ANCHORS
        or tag.startswith("score_")
        or bool(_YEAR_RE.match(tag))
    )


def apply_tag_dropout(caption: str, keep_prob: float, keep_min: int, rng: random.Random) -> str:
    """Randomly drop a subset of a caption's controllable *delta* tags.

    Structural tags (quality/score/recency/rating/year) and subject anchors
    (1girl/solo/...) are always kept in their original positions. Each remaining
    (delta) tag is kept with probability ``keep_prob``; if fewer than ``keep_min``
    delta tags survive, extra ones are randomly reinstated up to that floor (or all
    if there are fewer than ``keep_min``). Tag order is preserved.

    Purpose: expose the model to short-but-non-empty captions so short user prompts
    stay in-distribution and the model fills unstated attributes rather than copying
    the reference. See scripts/run_headroi_rope_cpm_editfix2_100k.sh.
    """
    tags = [t.strip() for t in caption.split(",")]
    tags = [t for t in tags if t]
    delta_idx = [i for i, t in enumerate(tags) if not _is_structural_tag(t)]
    if not delta_idx:
        return ", ".join(tags)
    kept = {i for i in delta_idx if rng.random() < keep_prob}
    floor = min(keep_min, len(delta_idx))
    if len(kept) < floor:
        pool = [i for i in delta_idx if i not in kept]
        rng.shuffle(pool)
        for i in pool[: floor - len(kept)]:
            kept.add(i)
    out = [t for i, t in enumerate(tags) if i not in delta_idx or i in kept]
    return ", ".join(out)


def inject_identity_tags(caption: str, inject_words, inject_prob: float, rng: random.Random) -> str:
    """Re-insert signature identity ACCESSORY words back into a built caption.

    ``build_caption``/``sig_subtract`` strip a character's signature tags (>=45%
    of their images) out of the caption, so recurring identity accessories
    (velma's glasses, tenryuu's eyepatch) become invisible to the text encoder.
    This re-injects a per-image, GT-faithful subset of them so they can be a
    controllable prompt handle. ``inject_words`` must already be the caption
    (space) form and pre-filtered to (character signature ∩ accessory-type ∩
    THIS image's tags) — callers pass the precomputed per-image list.

    Each word is injected with probability ``inject_prob`` (a KEEP rate; set it
    higher than the generic tag-dropout keep so accessories survive more often).
    Words already present are left untouched. Inserted before the trailing
    ``yearXXXX`` token so the year stays last (matches build_caption order).
    """
    if not inject_words or inject_prob <= 0:
        return caption
    parts = [t.strip() for t in caption.split(",")]
    parts = [t for t in parts if t]
    present = set(parts)
    add = [w for w in inject_words if w not in present and rng.random() < inject_prob]
    if not add:
        return caption
    if parts and _YEAR_RE.match(parts[-1]):
        return ", ".join(parts[:-1] + add + [parts[-1]])
    return ", ".join(parts + add)


def build_signature(rows: Iterable) -> dict[str, set[str]]:
    """Compute per-character signature: tags present in >= 0.45 of that
    character's images. `rows` is a pandas DataFrame with columns
    `character` and `tag_string_general`.

    Counting uses set(tags) per image (presence, not frequency).
    """
    from collections import defaultdict

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for char, tag_str in zip(rows["character"], rows["tag_string_general"]):
        totals[char] += 1
        for tag in set(str(tag_str).split()):
            counts[char][tag] += 1
    sigs: dict[str, set[str]] = {}
    for char, n in totals.items():
        sig = {t for t, c in counts[char].items() if c / n >= SIGNATURE_THRESHOLD}
        sigs[char] = sig
    return sigs
