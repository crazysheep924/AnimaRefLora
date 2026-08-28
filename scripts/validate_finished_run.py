#!/usr/bin/env python3
"""Plan or run post-training validation without creating a Cartesian explosion."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


STEP_RE = re.compile(r"lora_step_(\d+)\.safetensors$")
DEFAULT_PROMPTS = [
    "baseline=standing, cowboy shot, white dress, simple background, looking at viewer",
    "pose=running, full body, outdoors, dynamic pose, looking away",
    "edit=sitting, black jacket, city at night, upper body, looking at viewer",
]


def csv(values: list[str]) -> list[str]:
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def prompts(values: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for item in values or DEFAULT_PROMPTS:
        name, sep, text = item.partition("=")
        if not sep or not name.strip() or not text.strip():
            raise ValueError(f"--prompt must be NAME=TEXT, got {item!r}")
        out.append((re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()), text.strip()))
    return out


def discover_steps(run_dir: Path) -> list[int]:
    found = []
    for path in (run_dir / "checkpoints").glob("lora_step_*.safetensors"):
        match = STEP_RE.match(path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def add_job(jobs: list[dict], suite: str, name: str, command: list[str], output: Path) -> None:
    jobs.append({"suite": suite, "name": name, "output_dir": str(output), "command": command})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory containing checkpoints/")
    ap.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/generated/final_validation")
    ap.add_argument(
        "--suites", nargs="+", default=["all"],
        choices=["all", "trend", "factorial", "components", "rope", "sampler", "generalization"],
    )
    ap.add_argument("--checkpoint-steps", type=int, nargs="*", help="Explicit trend checkpoints")
    ap.add_argument("--trend-count", type=int, default=5, help="Use the last N checkpoints when steps are omitted")
    ap.add_argument("--final-step", type=int, help="Checkpoint used outside the trend suite; default latest")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prompt", action="append", help="Repeat NAME=TEXT; defaults cover baseline, pose, and edit")
    ap.add_argument("--negative-prompt", help="Forwarded verbatim; omit to use the inference default")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--characters", nargs="+", required=True)
    ap.add_argument("--ref-root", default="/workspace/storage/val/test")
    ap.add_argument("--wrong-ref", default="/work/RunpodTraining/wrong.jpg")
    ap.add_argument("--storage", default="/workspace/storage")
    ap.add_argument("--ccip-cache", default="/workspace/storage/runs/ccip_ref_head_emb_cache.pt")
    ap.add_argument("--head-roi-cache", default="/workspace/storage/runs/head_roi_cache.pt")
    ap.add_argument("--conditions", nargs="+", default=["correct,wrong,blank"])
    ap.add_argument("--ref-frame-modes", nargs="+", default=["both", "head_only", "full_only", "blank"])
    ap.add_argument("--guidance-scales", type=float, nargs="+", default=[1.0, 3.0, 4.5, 6.0])
    ap.add_argument("--flow-shifts", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--sample-steps", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument(
        "--rope-layouts", nargs="+", default=["checkpoint", "identity", "disjoint"],
        choices=["checkpoint", "identity", "disjoint", "shifted", "packed"],
    )
    ap.add_argument("--rope-shifts", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--decode-device", default="cuda")
    ap.add_argument("--vae-chunk-size", type=int, default=16)
    ap.add_argument("--ref-letterbox", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ccip-on-the-fly", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--execute", action="store_true", help="Run jobs; default only writes and prints the plan")
    ap.add_argument("--force", action="store_true", help="Run jobs whose output already has a manifest")
    args = ap.parse_args(argv)
    if args.trend_count < 1:
        ap.error("--trend-count must be >= 1")
    if any(value < 0 for value in args.rope_shifts):
        ap.error("--rope-shifts must be >= 0")
    unknown_conditions = set(csv(args.conditions)) - {"correct", "wrong", "blank"}
    if unknown_conditions:
        ap.error(f"unknown --conditions: {sorted(unknown_conditions)}")
    unknown_frames = set(csv(args.ref_frame_modes)) - {"both", "head_only", "full_only", "blank"}
    if unknown_frames:
        ap.error(f"unknown --ref-frame-modes: {sorted(unknown_frames)}")

    args.run_dir = args.run_dir.resolve()
    args.output_dir = (args.output_dir or args.run_dir / "generated" / "final_validation").resolve()
    all_steps = discover_steps(args.run_dir)
    if not all_steps:
        ap.error(f"no lora_step_*.safetensors under {args.run_dir / 'checkpoints'}")
    trend_steps = args.checkpoint_steps or all_steps[-args.trend_count :]
    missing = sorted(set(trend_steps + ([args.final_step] if args.final_step else [])) - set(all_steps))
    if missing:
        ap.error(f"checkpoint steps do not exist: {missing}")
    final_step = args.final_step or all_steps[-1]
    selected_suites = {"trend", "factorial", "components", "rope", "sampler", "generalization"} if "all" in args.suites else set(args.suites)
    prompt_cases = prompts(args.prompt)
    conditions = ",".join(csv(args.conditions))
    jobs: list[dict] = []

    def checkpoint(step: int) -> str:
        return str(args.run_dir / "checkpoints" / f"lora_step_{step}.safetensors")

    def local(step: int, out: Path, prompt: str, seed: int, extra: list[str] | None = None) -> list[str]:
        command = [
            args.python, "-m", "anima_reflora.local_ref_ab_infer",
            "--checkpoint", checkpoint(step), "--output-dir", str(out),
            "--ref-root", args.ref_root, "--wrong-ref", args.wrong_ref,
            "--storage", args.storage, "--ccip-cache", args.ccip_cache,
            "--head-roi-cache", args.head_roi_cache, "--prompt", prompt,
            "--year", str(args.year),
            "--seed", str(seed), "--conditions", conditions, "--limit", str(args.limit),
            "--steps", "24", "--flow-shift", "3.0", "--guidance-scale", "4.5",
            "--bucket-short", str(args.bucket), "--bucket-long-max", str(args.bucket),
            "--dtype", args.dtype, "--device", args.device, "--text-device", args.text_device,
            "--decode-device", args.decode_device, "--vae-chunk-size", str(args.vae_chunk_size),
        ]
        command.append("--ref-letterbox" if args.ref_letterbox else "--no-ref-letterbox")
        command.append("--ccip-on-the-fly" if args.ccip_on_the_fly else "--no-ccip-on-the-fly")
        if args.negative_prompt:
            command += ["--negative-prompt", args.negative_prompt]
        return command + (extra or [])

    base_name, base_prompt = prompt_cases[0]
    if "trend" in selected_suites:
        for step in trend_steps:
            for seed in args.seeds:
                out = args.output_dir / "trend" / f"step_{step}" / f"seed_{seed}"
                add_job(jobs, "trend", f"step={step},seed={seed}", local(step, out, base_prompt, seed), out)

    if "factorial" in selected_suites:
        for seed in args.seeds:
            out = args.output_dir / "factorial" / f"seed_{seed}"
            cmd = [
                args.python, "-m", "anima_reflora.factorial_ref_infer",
                "--checkpoint", checkpoint(final_step), "--output-dir", str(out),
                "--ref-root", args.ref_root, "--storage", args.storage,
                "--ccip-cache", args.ccip_cache, "--head-roi-cache", args.head_roi_cache,
                "--characters", *args.characters, "--prompt", base_prompt, "--seed", str(seed),
                "--year", str(args.year),
                "--steps", "24", "--flow-shift", "3.0", "--guidance-scale", "4.5",
                "--bucket-short", str(args.bucket), "--bucket-long-max", str(args.bucket),
                "--dtype", args.dtype, "--device", args.device, "--text-device", args.text_device,
                "--decode-device", args.decode_device, "--vae-chunk-size", str(args.vae_chunk_size),
            ]
            cmd.append("--ref-letterbox" if args.ref_letterbox else "--no-ref-letterbox")
            cmd.append("--ccip-on-the-fly" if args.ccip_on_the_fly else "--no-ccip-on-the-fly")
            if args.negative_prompt:
                cmd += ["--negative-prompt", args.negative_prompt]
            add_job(jobs, "factorial", f"step={final_step},seed={seed}", cmd, out)

    if "components" in selected_suites:
        modes = {"both": [], "frame": ["--no-cpm-component"], "cpm": ["--no-frame-adapter"], "off": ["--no-frame-adapter", "--no-cpm-component"]}
        for mode, flags in modes.items():
            for seed in args.seeds:
                out = args.output_dir / "components" / mode / f"seed_{seed}"
                add_job(jobs, "components", f"mode={mode},seed={seed}", local(final_step, out, base_prompt, seed, flags), out)

    if "rope" in selected_suites:
        for layout in args.rope_layouts:
            shifts = args.rope_shifts if layout not in {"checkpoint", "identity"} else [None]
            for shift in shifts:
                tag = layout if shift is None else f"{layout}_shift_{shift:g}"
                extra = [] if layout == "checkpoint" else ["--rope-layout-override", layout]
                if shift is not None:
                    extra += ["--rope-shift-override", str(shift)]
                for seed in args.seeds:
                    out = args.output_dir / "rope" / tag / f"seed_{seed}"
                    add_job(jobs, "rope", f"layout={tag},seed={seed}", local(final_step, out, base_prompt, seed, extra), out)

    if "sampler" in selected_suites:
        cases = [("cfg", value, ["--guidance-scale", str(value)]) for value in args.guidance_scales]
        cases += [("flow", value, ["--flow-shift", str(value)]) for value in args.flow_shifts]
        cases += [("steps", value, ["--steps", str(value)]) for value in args.sample_steps]
        for kind, value, flags in cases:
            out = args.output_dir / "sampler" / f"{kind}_{value:g}"
            add_job(jobs, "sampler", f"{kind}={value:g}", local(final_step, out, base_prompt, args.seeds[0], flags), out)

    if "generalization" in selected_suites:
        for name, prompt in prompt_cases:
            for seed in args.seeds:
                for frame_mode in csv(args.ref_frame_modes):
                    out = args.output_dir / "generalization" / name / frame_mode / f"seed_{seed}"
                    add_job(jobs, "generalization", f"prompt={name},frame={frame_mode},seed={seed}", local(final_step, out, prompt, seed, ["--ref-frame-mode", frame_mode]), out)

    plan = {
        "run_dir": str(args.run_dir), "final_step": final_step, "trend_steps": trend_steps,
        "variables": {
            "checkpoints": trend_steps, "seeds": args.seeds, "prompts": dict(prompt_cases),
            "reference_conditions": csv(args.conditions), "reference_frames": csv(args.ref_frame_modes),
            "components": ["LoKR", "frame_adapter", "CPM", "F0_head", "F1_full"],
            "rope_layouts": args.rope_layouts, "rope_shifts": args.rope_shifts,
            "guidance_scales": args.guidance_scales, "flow_shifts": args.flow_shifts,
            "sample_steps": args.sample_steps, "characters": args.characters,
            "negative_prompt": args.negative_prompt or "inference_default", "year": args.year,
            "bucket": args.bucket, "dtype": args.dtype, "ref_letterbox": args.ref_letterbox,
            "ccip_on_the_fly": args.ccip_on_the_fly, "vae_chunk_size": args.vae_chunk_size,
        },
        "measurements": {
            "identity": "CCIP head-crop similarity; report correct minus wrong and correct minus blank",
            "copy": "multi-scale regional NCC to reference; report max NCC and fraction >= 0.70",
            "prompt": "fixed-rubric prompt adherence for pose, clothing, framing, and background",
            "quality": "blind failure rate for anatomy, artifacts, crop, and unusable output",
            "sensitivity": "correct/wrong/blank and F0/F1 paired deltas with the same seed",
            "stability": "mean, standard deviation, and worst case across seeds and characters",
        },
        "training_only_requires_separate_runs": [
            "CREPA enabled/lambda/block/pool/sigma cutoff",
            "diff-loss lambda and minimum weight",
            "head-loss weight and sigma cutoff",
            "reference, caption, and tag dropout",
            "pair dHash threshold and cache coverage",
            "high-sigma sampling probability and timestep weighting",
            "optimizer, learning rate, LoKR dimension/alpha/factor, and training duration",
        ],
        "jobs": jobs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "validation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(f"plan: {plan_path}\njobs: {len(jobs)}")
    for index, job in enumerate(jobs, 1):
        print(f"[{index:03}/{len(jobs):03}] {job['suite']}: {job['name']}\n  {shlex.join(job['command'])}")
        if not args.execute:
            continue
        out = Path(job["output_dir"])
        if not args.force and out.exists() and any(out.glob("manifest*.json")):
            job["status"] = "skipped"
            continue
        result = subprocess.run(job["command"])
        job["status"] = "complete" if result.returncode == 0 else "failed"
        job["returncode"] = result.returncode
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
