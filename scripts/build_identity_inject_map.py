#!/usr/bin/env python3
"""Build the identity-accessory injection map for training.

For every image, emit the subset of its ORIGINAL tags that are (a) in the
character's signature (>=45% of their images, i.e. exactly what sig_subtract
strips out of the caption) AND (b) classified as a wearable ACCESSORY (see
anima_reflora/data/identity_tag_classes.json, produced by AI classification of
the real tag vocabulary). These are the recurring identity accessories that are
invisible to the text encoder after sig_subtract and get dropped once the copy
pathway is suppressed. Emitting them per-image keeps injection GT-faithful:
train.py re-inserts them (in caption/space form) only when the image truly has
them.

Output: JSON {"<danbooru_id>": ["accessory word", ...]} for images with >=1 word.
Key = basename-without-extension of the image path (matches train._inject_key).

Usage:
  python scripts/build_identity_inject_map.py \
      --parquet /path/to/dataset/index.parquet \
      --parquet /path/to/additions.parquet \
      --classes anima_reflora/data/identity_tag_classes.json \
      --out /path/to/identity_inject_map.json
"""
import argparse, json
from pathlib import Path

import pandas as pd

from anima_reflora.anima_caption import build_signature


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", action="append", required=True,
                    help="index parquet(s) with id/character/tag_string_general; repeatable")
    ap.add_argument("--classes", default="anima_reflora/data/identity_tag_classes.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = [pd.read_parquet(p)[["id", "character", "tag_string_general"]] for p in args.parquet]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="id", keep="first")
    print(f"images={len(df)} from {len(args.parquet)} parquet(s)")

    accessory = set(json.load(open(args.classes, encoding="utf-8"))["accessory"])
    print(f"accessory-class tags={len(accessory)}")

    signatures = build_signature(df)  # {character: set(signature tags, underscore form)}

    out: dict[str, list[str]] = {}
    n_words = 0
    for r in df.itertuples():
        sig = signatures.get(str(r.character))
        if not sig:
            continue
        tags = set(str(r.tag_string_general).split())
        words = [t for t in tags if t in sig and t in accessory]
        if not words:
            continue
        # caption (space) form, matching sig_subtract / build_caption; dedupe, stable order
        seen, space = set(), []
        for w in sorted(words):
            s = w.replace("_", " ")
            if s not in seen:
                seen.add(s)
                space.append(s)
        out[str(int(r.id))] = space
        n_words += len(space)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {args.out}: {len(out)} images with >=1 injectable accessory word, "
          f"{n_words} words total ({n_words/max(len(out),1):.2f} avg)")


if __name__ == "__main__":
    main()
