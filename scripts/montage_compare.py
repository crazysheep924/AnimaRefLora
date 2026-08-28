#!/usr/bin/env python3
"""Side-by-side: reference vs short-prompt vs rich-prompt `correct` outputs.

Rows = [ref, short prompt, rich prompt]; columns = characters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

GEN = Path("/work/RunpodTraining/generated")
SHORT = GEN / "ref_ab_step30000_1024"
RICH = GEN / "ref_ab_step30000_rich"
COND = "correct"
THUMB_W, LABEL_H, ROWLABEL_W = 240, 22, 120


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or "ref"


def main() -> int:
    refs = sorted(json.loads((SHORT / "manifest.json").read_text())["refs"], key=lambda r: r["index"])
    names = [r["name"] for r in refs]
    ncol = len(refs)
    rows = [
        ("ref", lambda r: Path(r["path"])),
        ("short prompt", lambda r: SHORT / f"{COND}_{r['index']:02d}_{safe(r['name'])}.png"),
        ("rich prompt", lambda r: RICH / f"{COND}_{r['index']:02d}_{safe(r['name'])}.png"),
    ]

    def thumb(p: Path):
        if not p.exists():
            return None
        im = Image.open(p).convert("RGB")
        return im.resize((THUMB_W, int(im.height * THUMB_W / im.width)), Image.Resampling.LANCZOS)

    cells = {}
    for ri, (_, fn) in enumerate(rows):
        for ci, r in enumerate(refs):
            t = thumb(fn(r))
            if t:
                cells[(ri, ci)] = t
    cell_h = max((im.height for im in cells.values()), default=THUMB_W) + LABEL_H
    W = ROWLABEL_W + ncol * THUMB_W
    H = LABEL_H + len(rows) * cell_h
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for ci, nm in enumerate(names):
        draw.text((ROWLABEL_W + ci * THUMB_W + 4, 4), nm[:30], fill=(0, 0, 0))
    for ri, (label, _) in enumerate(rows):
        y0 = LABEL_H + ri * cell_h
        draw.text((4, y0 + cell_h // 2), label, fill=(0, 0, 0))
        for ci in range(ncol):
            im = cells.get((ri, ci))
            if im:
                canvas.paste(im, (ROWLABEL_W + ci * THUMB_W, y0))
    out = GEN / "compare_short_vs_rich_step30000.png"
    canvas.save(out)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
