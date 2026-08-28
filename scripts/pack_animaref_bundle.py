#!/usr/bin/env python3
"""Pack a multi-file training checkpoint into a single .animaref.safetensors bundle.

The bundle is what the ComfyUI plugin's Loader consumes from
ComfyUI/models/anima_reflora/ — one file instead of five step-matched ones.

Usage:
  python scripts/pack_animaref_bundle.py <run_or_checkpoints_dir> \
      [--step N | --latest] [--name idinject_485k] [-o OUT.animaref.safetensors]

Examples:
  python scripts/pack_animaref_bundle.py \
      RunpodTraining/experiments/<run> --latest \
      --name idinject_485k -o dist_models/idinject_485k.animaref.safetensors
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anima_reflora.bundle import BUNDLE_SUFFIX, pack_bundle, verify_bundle  # noqa: E402


def find_checkpoints_dir(path: Path) -> Path:
    if (path / "checkpoints").is_dir():
        return path / "checkpoints"
    return path


def latest_step(ckpt_dir: Path) -> int:
    steps = [
        int(m.group(1))
        for p in ckpt_dir.glob("lora_step_*.safetensors")
        if (m := re.match(r"lora_step_(\d+)\.safetensors$", p.name))
    ]
    if not steps:
        raise SystemExit(f"no lora_step_*.safetensors under {ckpt_dir}")
    return max(steps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="run dir or checkpoints dir")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, help="checkpoint step to pack")
    group.add_argument("--latest", action="store_true", help="pack the highest available step (default)")
    parser.add_argument("--name", help="bundle display name stored in metadata (default: animaref_<step>)")
    parser.add_argument("-o", "--output", type=Path, help=f"output path (default: <name>{BUNDLE_SUFFIX} in cwd)")
    parser.add_argument("--no-verify", action="store_true", help="skip the bit-exact reload verification")
    args = parser.parse_args()

    ckpt_dir = find_checkpoints_dir(args.source)
    step = args.step if args.step is not None else latest_step(ckpt_dir)
    name = args.name or f"animaref_{step}"
    out = args.output or Path(f"{name}{BUNDLE_SUFFIX}")
    if not str(out).endswith(".safetensors"):
        out = out.with_name(out.name + BUNDLE_SUFFIX)

    print(f"packing step {step} from {ckpt_dir}")
    bundle = pack_bundle(ckpt_dir, step, out, name=name)
    size_mb = bundle.stat().st_size / 1e6
    print(f"wrote {bundle} ({size_mb:.1f} MB)")

    if not args.no_verify:
        verify_bundle(ckpt_dir, step, bundle)
        print("verified: bit-exact against the multi-file checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
