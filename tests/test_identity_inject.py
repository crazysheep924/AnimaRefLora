import random

from anima_reflora.anima_caption import (
    apply_tag_dropout,
    build_caption,
    build_signature,
    inject_identity_tags,
)


def test_inject_prob_one_adds_all_absent_words():
    cap = "masterpiece, best quality, score_9, recent, safe, 1girl, indoors, year2024"
    out = inject_identity_tags(cap, ["glasses", "halo"], 1.0, random.Random(0))
    parts = [t.strip() for t in out.split(",")]
    assert "glasses" in parts and "halo" in parts
    # year stays last
    assert parts[-1] == "year2024"
    # injected before the year token
    assert parts.index("glasses") < parts.index("year2024")


def test_inject_prob_zero_is_noop():
    cap = "masterpiece, best quality, 1girl, year2024"
    assert inject_identity_tags(cap, ["glasses"], 0.0, random.Random(0)) == cap


def test_inject_does_not_duplicate_present_word():
    cap = "masterpiece, best quality, 1girl, glasses, year2024"
    out = inject_identity_tags(cap, ["glasses"], 1.0, random.Random(0))
    assert out.split(", ").count("glasses") == 1
    assert out == cap  # nothing to add -> unchanged


def test_inject_empty_words_is_noop():
    cap = "masterpiece, 1girl, year2024"
    assert inject_identity_tags(cap, [], 1.0, random.Random(0)) == cap


def test_inject_appends_when_no_year_token():
    cap = "masterpiece, best quality, 1girl, indoors"
    out = inject_identity_tags(cap, ["eyepatch"], 1.0, random.Random(0))
    assert out.split(", ")[-1] == "eyepatch"


def test_inject_probability_is_per_word():
    # With prob ~0.5 over many trials, each word is added roughly half the time.
    cap = "masterpiece, 1girl, year2024"
    rng = random.Random(1234)
    added = 0
    trials = 2000
    for _ in range(trials):
        out = inject_identity_tags(cap, ["glasses"], 0.5, rng)
        if "glasses" in out:
            added += 1
    assert 0.4 * trials < added < 0.6 * trials


def test_inject_composes_with_tag_dropout():
    # A dropout step can strip delta tags; injection then restores a signature
    # accessory that was never in the (sig-subtracted) caption to begin with.
    cap = "masterpiece, best quality, score_9, recent, safe, 1girl, solo, standing, indoors, year2024"
    rng = random.Random(7)
    dropped = apply_tag_dropout(cap, keep_prob=0.5, keep_min=3, rng=rng)
    out = inject_identity_tags(dropped, ["cat ear headphones"], 1.0, rng)
    parts = [t.strip() for t in out.split(",")]
    assert "cat ear headphones" in parts
    assert parts[0] == "masterpiece"  # structural prefix survives dropout
    assert parts[-1] == "year2024"


def test_signature_accessory_intersection_matches_map_semantics():
    # Mirror build_identity_inject_map: signature ∩ accessory ∩ image tags,
    # in caption (space) form. A recurring accessory should surface; a one-off
    # should not (it isn't in the signature).
    rows = {
        "character": ["velma"] * 5,
        "tag_string_general": [
            "1girl glasses orange_sweater",
            "1girl glasses orange_sweater",
            "1girl glasses orange_sweater",
            "1girl glasses orange_sweater smile",
            "1girl orange_sweater santa_hat",  # santa_hat one-off (1/5 = 0.2 < 0.45)
        ],
    }
    sig = build_signature(rows)["velma"]
    assert "glasses" in sig  # 4/5 = 0.8 >= 0.45
    assert "santa_hat" not in sig  # 1/5 one-off, not signature
    accessory = {"glasses", "santa_hat"}
    img_tags = {"1girl", "orange_sweater", "santa_hat"}  # the one-off image
    injectable = [t.replace("_", " ") for t in img_tags if t in sig and t in accessory]
    assert injectable == []  # santa_hat not signature -> nothing injected here
    img_tags2 = {"1girl", "glasses", "orange_sweater"}
    injectable2 = [t.replace("_", " ") for t in img_tags2 if t in sig and t in accessory]
    assert injectable2 == ["glasses"]
