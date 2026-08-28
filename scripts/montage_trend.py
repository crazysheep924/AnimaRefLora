#!/usr/bin/env python3
"""Build a checkpoint-trend montage for the REF A/B `correct` condition.

Rows = [reference, step 5000, 10000, ...]; columns = reference characters.
Lets you eyeball how identity adherence / prompt-following drifts across
training steps. Reads generated/ref_ab_step<N>_<bucket>/ dirs + their
manifests. Run inside the container (PIL + ref images available).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

GEN = Path("/work/RunpodTraining/generated")
COND = "correct"
THUMB_W, LABEL_H, ROWLABEL_W = 220, 22, 90
STEP_RE = re.compile(r"ref_ab_step(\d+)_\d+$")


def step_dirs() -> list[tuple[int, Path]]:
    out = []
    for d in GEN.glob("ref_ab_step*_*"):
        m = STEP_RE.search(d.name)
        if m and (d / "manifest.json").exists():
            out.append((int(m.group(1)), d))
    return sorted(out)


def main() -> int:
    dirs = step_dirs()
    if not dirs:
        print("no step dirs found")
        return 1
    # char order + ref paths from the first manifest
    m0 = json.loads((dirs[0][1] / "manifest.json").read_text())
    refs = sorted(m0["refs"], key=lambda r: r["index"])
    names = [r["name"] for r in refs]
    ref_paths = [Path(r["path"]) for r in refs]
    ncol = len(refs)
    rows = [("ref", None)] + [(f"step {s}", d) for s, d in dirs]
    print(f"cols={ncol} chars={names} rows={[r[0] for r in rows]}")

    # cpm_valid summary
    for s, d in dirs:
        mm = json.loads((d / "manifest.json").read_text())
        vc = sum(1 for o in mm["outputs"] if o["condition"] == COND and o["ccip_valid"])
        print(f"  step {s}: {COND} ccip_valid {vc}/{ncol}")

    def thumb(path: Path) -> Image.Image | None:
        if not path.exists():
            return None
        im = Image.open(path).convert("RGB")
        h = int(im.height * THUMB_W / im.width)
        return im.resize((THUMB_W, h), Image.Resampling.LANCZOS)

    # precompute to get a common cell height (max thumb height)
    cell_imgs: dict[tuple[int, int], Image.Image] = {}
    for ci, rp in enumerate(ref_paths):
        t = thumb(rp)
        if t:
            cell_imgs[(0, ci)] = t
    for ri, (_, d) in enumerate(rows[1:], start=1):
        for ci, r in enumerate(refs):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", r["name"].strip()).strip("_") or "ref"
            p = d / f"{COND}_{r['index']:02d}_{safe}.png"
            t = thumb(p)
            if t:
                cell_imgs[(ri, ci)] = t
    cell_h = max((im.height for im in cell_imgs.values()), default=THUMB_W) + LABEL_H

    W = ROWLABEL_W + ncol * THUMB_W
    H = LABEL_H + len(rows) * cell_h
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    # column headers (char names)
    for ci, nm in enumerate(names):
        draw.text((ROWLABEL_W + ci * THUMB_W + 4, 4), nm[:26], fill=(0, 0, 0))
    for ri, (label, _) in enumerate(rows):
        y0 = LABEL_H + ri * cell_h
        draw.text((4, y0 + cell_h // 2), label, fill=(0, 0, 0))
        for ci in range(ncol):
            im = cell_imgs.get((ri, ci))
            if im is None:
                continue
            canvas.paste(im, (ROWLABEL_W + ci * THUMB_W, y0))

    out = GEN / f"trend_{COND}.png"
    canvas.save(out)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
