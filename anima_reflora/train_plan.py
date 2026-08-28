from __future__ import annotations

import subprocess
import sys
import shlex

from .config import STAGE_DEFAULT_STEPS
from .preflight import run_preflight
from .train import main as train_main


STAGE_ARGS = {
    "headroi-short": ["--frames", "3", "--head-loss-weight", "4.0"],
    "head-sigma-short": ["--frames", "3", "--head-sigma-cutoff", "0.6"],
    "dropout-short": ["--frames", "3", "--ref-dropout-prob", "0.1"],
    "rope-smoke": ["--frames", "3", "--rope-refpos", "--no-ref-eval", "--no-viz", "--ckpt-every", "5", "--steps", "20"],
    "rope-short": ["--frames", "3", "--rope-refpos"],
    "cpm-smoke": ["--frames", "3", "--cpm", "--no-ref-eval", "--no-viz", "--ckpt-every", "5", "--steps", "20"],
    "cpm-short": ["--frames", "3", "--cpm"],
    "rope-cpm-short": ["--frames", "3", "--rope-refpos", "--cpm"],
    "combo-long": ["--frames", "3", "--rope-refpos", "--cpm", "--crepa"],
    "from0-headroi-rope-cpm": ["--frames", "3", "--from-scratch", "--rope-refpos", "--cpm", "--head-loss-weight", "4.0"],
}


def print_plan() -> None:
    print("Available stages:")
    for name in STAGE_DEFAULT_STEPS:
        defaults = STAGE_ARGS.get(name, [])
        print(f"  {name}: steps={STAGE_DEFAULT_STEPS[name]} args={' '.join(defaults)}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    stage = args[0] if args and not args[0].startswith("-") else "plan"
    rest = args[1:] if args and not args[0].startswith("-") else args
    if stage == "plan":
        print_plan()
        return 0
    if stage == "tests":
        cmd = [sys.executable, "-m", "pytest", "-q"]
        print("+ " + shlex.join(cmd), flush=True)
        return subprocess.call(cmd)
    if stage == "preflight":
        cmd = [sys.executable, "-m", "anima_reflora.preflight", "--stage", stage, *rest]
        print("+ " + shlex.join(cmd), flush=True)
        report = run_preflight(["--stage", stage, *rest])
        print(report)
        return 0
    if stage == "cpm-preflight":
        cmd = [sys.executable, "-m", "anima_reflora.preflight", "--stage", stage, "--cpm", *rest]
        print("+ " + shlex.join(cmd), flush=True)
        report = run_preflight(["--stage", stage, "--cpm", *rest])
        print(report)
        return 0
    if stage not in STAGE_DEFAULT_STEPS:
        raise SystemExit(f"Unknown stage: {stage}")
    defaults = ["--stage", stage, "--steps", str(STAGE_DEFAULT_STEPS[stage]), *STAGE_ARGS.get(stage, [])]
    print("+ " + shlex.join([sys.executable, "-m", "anima_reflora.train", *defaults, *rest]), flush=True)
    return train_main([*defaults, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
