import pandas as pd
import re
from collections import Counter

df = pd.read_parquet("/data/index.parquet")
print("rows:", len(df))
print("columns:", list(df.columns))
print()

# --- rating distribution ---
if "rating" in df:
    print("=== rating distribution ===")
    vc = df["rating"].fillna("(nan)").value_counts(dropna=False)
    for k, v in vc.items():
        print(f"  {k!r:12} {v:8d}  ({100*v/len(df):5.2f}%)")
print()

# --- gather all tag-ish columns ---
tag_cols = [c for c in df.columns if "tag" in c.lower() or c.lower() in ("caption","general","captions")]
print("tag-ish columns scanned:", tag_cols)
print()

# loli-class tags (danbooru vocabulary)
LOLI_TAGS = [
    "loli", "shota", "toddlercon", "baby", "infant", "child", "toddler",
    "age_regression", "aged_down", "younger", "young_boy", "young_girl",
    "kindergarten", "elementary", "preschool", "preteen",
]
# ambiguous / borderline (report separately, NOT necessarily loli)
BORDER_TAGS = ["petite", "flat_chest", "small_breasts", "child_drawing"]

def norm_tags(s):
    if not isinstance(s, str):
        return []
    return re.split(r"[,\s]+", s.strip().lower())

loli_hits = Counter()
border_hits = Counter()
rows_with_loli = set()
for col in tag_cols:
    for idx, val in df[col].items():
        toks = set(norm_tags(val))
        for t in LOLI_TAGS:
            if t in toks:
                loli_hits[f"{col}:{t}"] += 1
                rows_with_loli.add(idx)
        for t in BORDER_TAGS:
            if t in toks:
                border_hits[f"{col}:{t}"] += 1

print("=== LOLI-CLASS tag hits (exact token match) ===")
if loli_hits:
    for k, v in loli_hits.most_common():
        print(f"  {k:35} {v}")
else:
    print("  (none)")
print(f"  -> distinct rows containing any loli-class tag: {len(rows_with_loli)} / {len(df)} ({100*len(rows_with_loli)/len(df):.3f}%)")
print()

print("=== BORDERLINE tags (ambiguous, not necessarily loli) ===")
if border_hits:
    for k, v in border_hits.most_common():
        print(f"  {k:35} {v}")
else:
    print("  (none)")
print()

# substring scan as a safety net (catches loli in compound tags e.g. 'loli, ...')
print("=== substring 'loli'/'shota' anywhere in tag columns (safety net) ===")
for col in tag_cols:
    ser = df[col].astype(str).str.lower()
    n_loli = ser.str.contains("loli").sum()
    n_shota = ser.str.contains("shota").sum()
    print(f"  {col}: contains 'loli'={n_loli}, 'shota'={n_shota}")
