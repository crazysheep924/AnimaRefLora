#!/usr/bin/env python3
"""Idempotent batch REF A/B generator for every training checkpoint.

Scans the checkpoints dir for ``lora_step_*.safetensors``, and for each one
generates the correct/wrong/blank REF A/B grid at 1024 via the docker image —
but only if that checkpoint's output is missing or incomplete. Re-running picks
up newly-saved checkpoints automatically (training keeps writing step_35000,
step_40000, ...), so this can be run repeatedly / on a schedule.

Each checkpoint is one ``docker run`` of ``anima_reflora.local_ref_ab_infer``
with the validated low-VRAM flags (GPU decode + raised fd limit). Output lands
in ``<work>/generated/ref_ab_step<N>_1024``.

Usage (host / WSL):
    python3 scripts/gen_ref_ab_all.py                 # generate all missing
    python3 scripts/gen_ref_ab_all.py --dry-run       # show plan only
    python3 scripts/gen_ref_ab_all.py --only 30000    # one specific step
    python3 scripts/gen_ref_ab_all.py --force         # re-generate even if present
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Host paths (defaults match the local WSL layout).
DEF_REPO = Path("/path/to/anima-reflora")
DEF_STORAGE = Path("/path/to/storage")            # -> /workspace/storage
DEF_REFS = Path("/path/to/eval_refs")        # -> /workspace/storage/val/test
DEF_IMAGE = "anima-reflora-runpod:nobuildhead-20260629"

# Container paths (fixed; the mounts below map host -> these).
C_REPO_PKG = "/opt/AnimaRefLora/anima_reflora"
C_STORAGE = "/workspace/storage"
C_REFS = "/workspace/storage/val/test"
C_WORK = "/work/RunpodTraining"

STEP_RE = re.compile(r"lora_step_(\d+)\.safetensors$")


def parse_extra_lora(value: str) -> tuple[Path, float]:
    """Parse HOSTPATH[:MULT]; a trailing colon-segment that is not a float stays
    part of the path (drive-letter safe)."""
    path_text, mult = value, 1.0
    head, sep, tail = value.rpartition(":")
    if sep:
        try:
            mult = float(tail)
            path_text = head
        except ValueError:
            pass
    return Path(path_text), mult


def discover_steps(ckpt_dir: Path) -> list[int]:
    steps = []
    for path in ckpt_dir.glob("lora_step_*.safetensors"):
        m = STEP_RE.search(path.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def is_complete(out_dir: Path, expected_images: int, require_grid: bool = True) -> bool:
    """A run is complete iff its manifest reports the expected image count and
    the grid + every listed PNG actually exist on disk."""
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    if int(data.get("images", -1)) != expected_images:
        return False
    if require_grid and not (out_dir / "comparison_grid.png").exists():
        return False
    pngs = [p for p in out_dir.glob("*.png") if p.name != "comparison_grid.png"]
    return len(pngs) >= expected_images


def docker_cmd(args, step: int, out_name: str) -> list[str]:
    work_run = f"{C_WORK}/{args.run_subdir}" if args.run_subdir else C_WORK
    lora_mounts: list[str] = []
    lora_args: list[str] = []
    for idx, (host_path, mult) in enumerate(args.extra_lora_specs):
        c_path = f"/extra_loras/{idx}_{host_path.name}"
        lora_mounts += ["-v", f"{host_path}:{c_path}:ro"]
        lora_args += ["--extra-lora", f"{c_path}:{mult}"]
    return [
        "docker", "run", "--gpus", "all", "--rm",
        "--ulimit", "nofile=1048576:1048576",
        # The runpod images bake workspace-root ANIMA_REFLORA_* envs
        # (/workspace/anima_models, ...); local_ref_ab_infer only setdefault()s,
        # so explicitly override to the /workspace/storage mount layout.
        "-e", f"ANIMA_REFLORA_STORAGE={C_STORAGE}",
        "-e", f"ANIMA_REFLORA_MODEL_DIT={C_STORAGE}/anima_models/diffusion_models/anima-base-v1.0.safetensors",
        "-e", f"ANIMA_REFLORA_MODEL_TE={C_STORAGE}/anima_models/text_encoders",
        "-e", f"ANIMA_REFLORA_MODEL_VAE={C_STORAGE}/anima_models/vae/qwen_image_vae.safetensors",
        "-e", f"ANIMA_REFLORA_VAL={C_REFS}",
        "-v", f"{args.repo / 'anima_reflora'}:{C_REPO_PKG}",
        "-v", f"{args.storage}:{C_STORAGE}",
        "-v", f"{args.refs}:{C_REFS}",
        "-v", f"{args.repo / 'RunpodTraining'}:{C_WORK}",
        *lora_mounts,
        args.image,
        "python", "-m", "anima_reflora.local_ref_ab_infer",
        "--checkpoint", f"{work_run}/checkpoints/lora_step_{step}.safetensors",
        "--ref-root", C_REFS,
        "--wrong-ref", f"{C_WORK}/wrong.jpg",
        "--output-dir", f"{work_run}/generated/{out_name}",
        "--storage", C_STORAGE,
        "--limit", str(args.limit),
        "--steps", str(args.steps),
        "--guidance-scale", str(args.guidance_scale),
        "--bucket-short", str(args.bucket),
        "--bucket-long-max", str(args.bucket),
        "--device", "cuda",
        "--decode-device", "cuda",
        *lora_args,
        *(["--prompt", args.prompt] if args.prompt else []),
        *(["--conditions", args.conditions, "--skip-grid"] if args.conditions else []),
        *(["--lora-multiplier", str(args.lora_multiplier)] if args.lora_multiplier != 1.0 else []),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=DEF_REPO, help="DockerAnimaRefLora repo root (host)")
    ap.add_argument("--storage", type=Path, default=DEF_STORAGE, help="storage root mounted to /workspace/storage")
    ap.add_argument("--refs", type=Path, default=DEF_REFS, help="ref-image dir mounted to val/test")
    ap.add_argument("--image", default=DEF_IMAGE, help="docker image tag")
    ap.add_argument("--run-subdir", default="", help="run folder under RunpodTraining/ holding checkpoints/ + generated/ (e.g. my-run-20260101-000000)")
    ap.add_argument("--limit", type=int, default=5, help="number of ref characters (each -> correct/wrong/blank)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance-scale", type=float, default=4.5)
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--prompt", default=None, help="override the eval prompt passed to local_ref_ab_infer")
    ap.add_argument("--conditions", default=None, help="subset of correct/wrong/blank passed to local_ref_ab_infer")
    ap.add_argument("--lora-multiplier", type=float, default=1.0, help="runtime strength of the trained adapter network")
    ap.add_argument(
        "--extra-lora",
        action="append",
        default=[],
        metavar="HOSTPATH[:MULT]",
        help="external Anima LoRA merged into the base DiT before sampling; repeatable",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix for the output dir name; defaults to _lora-<stems> when --extra-lora is used",
    )
    ap.add_argument("--only", type=int, nargs="*", help="restrict to these step numbers")
    ap.add_argument("--force", action="store_true", help="regenerate even if output already complete")
    ap.add_argument("--dry-run", action="store_true", help="print plan, do not run docker")
    args = ap.parse_args(argv)

    args.extra_lora_specs = [parse_extra_lora(v) for v in args.extra_lora]
    for host_path, _ in args.extra_lora_specs:
        if not host_path.is_file():
            print(f"[gen] extra lora not found: {host_path}", file=sys.stderr)
            return 2
    out_suffix = args.out_suffix
    if not out_suffix and args.extra_lora_specs:
        stems = "-".join(re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem)[:24] for p, _ in args.extra_lora_specs)
        out_suffix = f"_lora-{stems}"

    run_root = args.repo / "RunpodTraining"
    if args.run_subdir:
        run_root = run_root / args.run_subdir
    ckpt_dir = run_root / "checkpoints"
    gen_dir = run_root / "generated"
    if not ckpt_dir.is_dir():
        print(f"[gen] checkpoints dir not found: {ckpt_dir}", file=sys.stderr)
        return 2

    steps = discover_steps(ckpt_dir)
    if args.only:
        steps = [s for s in steps if s in set(args.only)]
    if not steps:
        print("[gen] no matching lora_step_*.safetensors found")
        return 0

    n_conditions = len([c for c in (args.conditions or "correct,wrong,blank").split(",") if c.strip()])
    expected_images = args.limit * n_conditions
    todo: list[tuple[int, str]] = []
    for step in steps:
        out_name = f"ref_ab_step{step}_{args.bucket}{out_suffix}"
        out_dir = gen_dir / out_name
        if not args.force and is_complete(out_dir, expected_images, require_grid=args.conditions is None):
            print(f"[gen] step {step:>6}: already complete -> skip ({out_name})")
            continue
        todo.append((step, out_name))

    if not todo:
        print("[gen] nothing to do; all checkpoints already generated.")
        return 0

    print(f"[gen] will generate {len(todo)} checkpoint(s): {[s for s, _ in todo]}")
    failures: list[int] = []
    for step, out_name in todo:
        cmd = docker_cmd(args, step, out_name)
        print(f"\n[gen] === step {step} -> {out_name} ===")
        if args.dry_run:
            print("       " + " ".join(cmd))
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[gen] step {step}: FAILED (exit {result.returncode})", file=sys.stderr)
            failures.append(step)
        else:
            print(f"[gen] step {step}: done")

    if failures:
        print(f"\n[gen] completed with failures: {failures}", file=sys.stderr)
        return 1
    print("\n[gen] all requested checkpoints generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
