"""Reference-use evaluation: correct vs wrong vs blank reference conditions.

Loads a trained checkpoint, selects validation pairs from the latent cache,
and runs single-step model predictions under three reference conditions to
measure whether the model actually uses the reference.

Supports both real (external sd-scripts) and tiny backends.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch

from .cache import (
    LatentCacheIndex,
    SyntheticLatentDataset,
    build_pairs,
    collate_latent_batch,
    latent_to_chw,
)
from .checkpoints import load_checkpoint_into, load_sidecar_into
from .config import ANIMA_DEFAULT_EVAL_PROMPT, TrainConfig, parse_config
from .features import CcipEmbeddingCache, feature_snapshot
from .models import build_model, sidecar_modules
from .noise import make_noised_batch
from .version_meta import collect_runtime_versions


EVAL_SIGMAS = [0.3, 0.5, 0.7]


def _select_eval_pairs(
    index: LatentCacheIndex,
    frames: int,
    num_refs: int,
    seed: int,
) -> list[dict]:
    """Select evaluation pairs: correct ref, wrong ref (different character), blank."""
    pairs = build_pairs(index, frames=frames, require_head_ref=(frames == 3))
    if not pairs:
        return []

    by_char_bucket: dict[tuple[str, tuple[int, int]], list] = defaultdict(list)
    for pair in pairs:
        by_char_bucket[(pair.character, pair.bucket)].append(pair)

    rng = random.Random(seed)
    selected = rng.sample(pairs, min(num_refs, len(pairs)))

    eval_items = []
    for pair in selected:
        wrong_candidates = [
            p for p in pairs
            if p.character != pair.character and p.bucket == pair.bucket
        ]
        wrong_pair = rng.choice(wrong_candidates) if wrong_candidates else None

        eval_items.append({
            "target": pair.target_full,
            "correct_ref_full": pair.ref_full,
            "correct_ref_head": pair.ref_head,
            "wrong_ref_full": wrong_pair.ref_full if wrong_pair else None,
            "wrong_ref_head": wrong_pair.ref_head if wrong_pair else None,
            "character": pair.character,
            "bucket": pair.bucket,
            "caption": pair.target_full.caption,
        })
    return eval_items


def _build_condition_batch(
    index: LatentCacheIndex,
    eval_item: dict,
    condition: str,
    frames: int,
) -> dict:
    """Build a single-item batch dict for one condition."""
    target_rec = eval_item["target"]
    target_latent = latent_to_chw(index.load_latent(target_rec))

    if condition == "correct":
        ref_full_rec = eval_item["correct_ref_full"]
        ref_head_rec = eval_item["correct_ref_head"]
    elif condition == "wrong":
        ref_full_rec = eval_item["wrong_ref_full"]
        ref_head_rec = eval_item["wrong_ref_head"]
        if ref_full_rec is None:
            return {}
    elif condition == "blank":
        ref_full_rec = None
        ref_head_rec = None
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if ref_full_rec is not None:
        ref_full = latent_to_chw(index.load_latent(ref_full_rec))
    else:
        ref_full = torch.zeros_like(target_latent)

    if frames == 3:
        if ref_head_rec is not None:
            ref_head = latent_to_chw(index.load_latent(ref_head_rec))
        else:
            ref_head = torch.zeros_like(target_latent)
        frame_list = [ref_head, ref_full, target_latent]
    else:
        frame_list = [ref_full, target_latent]

    clean = torch.stack(frame_list, dim=1).unsqueeze(0)

    cap = index.load_caption(eval_item["caption"])
    return {
        "clean": clean,
        "prompt_embeds": cap["prompt_embeds"][:1],
        "attn_mask": cap.get("attn_mask", cap.get("t5_attn_mask"))[:1],
        "t5_input_ids": cap.get("t5_input_ids", torch.zeros_like(cap.get("attn_mask", cap.get("t5_attn_mask"))))[:1],
        "t5_attn_mask": cap.get("t5_attn_mask", cap.get("attn_mask"))[:1],
        "ref_path": ref_full_rec.path if ref_full_rec else "blank",
        "target_path": target_rec.path,
    }


def _build_synthetic_condition_batches(
    frames: int,
    seed: int,
) -> list[dict]:
    """Build synthetic eval batches for testing without real cache."""
    dataset = SyntheticLatentDataset(length=8, frames=frames, seed=seed)
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collate_latent_batch(items)
    clean = batch["clean"]

    results = []
    for condition in ["correct", "wrong", "blank"]:
        c = clean.clone()
        if condition == "wrong":
            perm = torch.roll(torch.arange(c.shape[0]), 1)
            c[:, :, :-1] = c[perm, :, :-1]
        elif condition == "blank":
            c[:, :, :-1] = 0.0
        results.append({
            "condition": condition,
            "clean": c,
            "prompt_embeds": batch["prompt_embeds"],
            "attn_mask": batch["attn_mask"],
            "t5_input_ids": batch["t5_input_ids"],
            "t5_attn_mask": batch["t5_attn_mask"],
        })
    return results


@torch.no_grad()
def _eval_single_condition(
    model,
    clean: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    batch_extra: dict,
    sigma: float,
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Run model forward at a fixed sigma, return MSE and prediction stats."""
    clean = clean.to(device=device, dtype=dtype)
    prompt = prompt_embeds.to(device=device, dtype=dtype)
    mask = attn_mask.to(device=device)

    B, C, T, H, W = clean.shape
    noise = torch.randn(B, C, H, W, device=device, dtype=dtype)
    x = clean.clone()
    x[:, :, -1] = (1 - sigma) * clean[:, :, -1] + sigma * noise
    target_velocity = noise - clean[:, :, -1]

    timesteps = torch.zeros(B, T, device=device, dtype=torch.float32)
    timesteps[:, -1] = sigma

    forward_batch = {
        "t5_input_ids": batch_extra["t5_input_ids"].to(device=device),
        "t5_attn_mask": batch_extra["t5_attn_mask"].to(device=device),
    }

    prediction = model(
        x, timesteps,
        caption_embeds=prompt,
        attention_mask=mask,
        batch=forward_batch,
        config=config,
    )

    pred_v = prediction[:, :, -1].float()
    target_v = target_velocity.float()
    mse = torch.nn.functional.mse_loss(pred_v, target_v).item()

    x0_pred = (clean[:, :, -1].float() + sigma * (clean[:, :, -1].float() - x[:, :, -1].float() + pred_v))
    x0_true = clean[:, :, -1].float()
    reconstruction_mse = torch.nn.functional.mse_loss(x0_pred, x0_true).item()

    return {
        "mse": mse,
        "reconstruction_mse": reconstruction_mse,
        "pred_norm": float(pred_v.norm().item()),
        "target_norm": float(target_v.norm().item()),
    }


def run_ref_use_eval(config: TrainConfig, checkpoint: str | None = None) -> dict:
    """Run reference-use evaluation and return structured results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and config.dtype == "bf16" else torch.float32

    ccip_dim = None
    if config.cpm:
        try:
            ccip_cache = CcipEmbeddingCache(config.ccip_cache)
            ccip_dim = ccip_cache.dim
        except Exception:
            pass

    model = build_model(config, ccip_dim=ccip_dim).to(device=device, dtype=dtype)

    ckpt_path = checkpoint or config.resume or config.base_ckpt
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint_into(model, ckpt_path, strict=False)
        for name, module in sidecar_modules(model).items():
            try:
                load_sidecar_into(module, ckpt_path, name, strict=False)
            except FileNotFoundError:
                pass
        # RoPE sidecar fail-safe: a checkpoint trained with ref-position relocation
        # carries a sidecar; apply it (so eval positions match training) and fail loud
        # if it exists but no scheme is installed — silent identity-position eval would
        # produce wrong outputs (B4).
        dit = getattr(model, "dit", None)
        if dit is not None and getattr(dit, "pos_embedder", None) is not None:
            from .rope_refpos import assert_sidecar_applied, maybe_apply_sidecar

            maybe_apply_sidecar(dit, ckpt_path, expected_frames=config.frames)
            assert_sidecar_applied(dit, ckpt_path, expected_frames=config.frames)
    model.eval()

    seeds = [int(s) for s in str(config.ref_eval_seeds).split(",") if s.strip()]
    sigmas = EVAL_SIGMAS
    conditions = ["correct", "wrong", "blank"]

    use_real_cache = not config.synthetic_data
    index = None
    eval_items = []

    if use_real_cache:
        paths = config.paths()
        index = LatentCacheIndex.load(paths.latcache, prompt_mode=config.prompt_mode)
        eval_items = _select_eval_pairs(
            index, frames=config.frames,
            num_refs=config.ref_eval_refs, seed=config.seed,
        )

    all_results = []

    if use_real_cache and eval_items and index is not None:
        for item_idx, eval_item in enumerate(eval_items):
            item_results = {
                "item": item_idx,
                "character": eval_item["character"],
                "target_path": eval_item["target"].path,
                "correct_ref_path": eval_item["correct_ref_full"].path,
                "wrong_ref_path": eval_item["wrong_ref_full"].path if eval_item["wrong_ref_full"] else None,
                "conditions": {},
            }
            for sigma in sigmas:
                sigma_key = f"sigma_{sigma:.2f}"
                item_results["conditions"][sigma_key] = {}
                for condition in conditions:
                    batch_data = _build_condition_batch(index, eval_item, condition, config.frames)
                    if not batch_data:
                        continue
                    metrics = _eval_single_condition(
                        model,
                        batch_data["clean"],
                        batch_data["prompt_embeds"],
                        batch_data["attn_mask"],
                        {"t5_input_ids": batch_data["t5_input_ids"], "t5_attn_mask": batch_data["t5_attn_mask"]},
                        sigma=sigma,
                        config=config,
                        device=device,
                        dtype=dtype,
                    )
                    item_results["conditions"][sigma_key][condition] = metrics
            all_results.append(item_results)
    else:
        synthetic_batches = _build_synthetic_condition_batches(config.frames, config.seed)
        for sigma in sigmas:
            sigma_key = f"sigma_{sigma:.2f}"
            for sb in synthetic_batches:
                condition = sb["condition"]
                metrics = _eval_single_condition(
                    model,
                    sb["clean"],
                    sb["prompt_embeds"],
                    sb["attn_mask"],
                    {"t5_input_ids": sb["t5_input_ids"], "t5_attn_mask": sb["t5_attn_mask"]},
                    sigma=sigma,
                    config=config,
                    device=device,
                    dtype=dtype,
                )
                all_results.append({
                    "condition": condition,
                    "sigma": sigma,
                    **metrics,
                })

    summary = _compute_summary(all_results, use_real_cache)

    return {
        "checkpoint": ckpt_path,
        "seeds": seeds,
        "sigmas": sigmas,
        "conditions": conditions,
        "frames": config.frames,
        "backend": config.backend,
        "num_eval_items": len(eval_items) if use_real_cache else len(synthetic_batches) if not use_real_cache else 0,
        "summary": summary,
        "results": all_results,
        "runtime_versions": collect_runtime_versions(config.sd_scripts),
    }


def _compute_summary(results: list[dict], use_real_cache: bool) -> dict:
    """Compute aggregate metrics across all eval items."""
    if not results:
        return {}

    if use_real_cache:
        correct_mses = []
        wrong_mses = []
        blank_mses = []
        for item in results:
            for sigma_key, conds in item.get("conditions", {}).items():
                if "correct" in conds:
                    correct_mses.append(conds["correct"]["mse"])
                if "wrong" in conds:
                    wrong_mses.append(conds["wrong"]["mse"])
                if "blank" in conds:
                    blank_mses.append(conds["blank"]["mse"])

        correct_mean = sum(correct_mses) / len(correct_mses) if correct_mses else 0.0
        wrong_mean = sum(wrong_mses) / len(wrong_mses) if wrong_mses else 0.0
        blank_mean = sum(blank_mses) / len(blank_mses) if blank_mses else 0.0

        return {
            "correct_mse_mean": correct_mean,
            "wrong_mse_mean": wrong_mean,
            "blank_mse_mean": blank_mean,
            "gap_correct_vs_wrong": wrong_mean - correct_mean,
            "gap_correct_vs_blank": blank_mean - correct_mean,
            "ref_used": correct_mean < wrong_mean and correct_mean < blank_mean,
            "num_correct": len(correct_mses),
            "num_wrong": len(wrong_mses),
            "num_blank": len(blank_mses),
        }
    else:
        by_condition: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_condition[r["condition"]].append(r["mse"])
        means = {k: sum(v) / len(v) for k, v in by_condition.items()}
        return {
            "correct_mse_mean": means.get("correct", 0.0),
            "wrong_mse_mean": means.get("wrong", 0.0),
            "blank_mse_mean": means.get("blank", 0.0),
            "gap_correct_vs_wrong": means.get("wrong", 0.0) - means.get("correct", 0.0),
            "gap_correct_vs_blank": means.get("blank", 0.0) - means.get("correct", 0.0),
        }


def write_eval_output(report: dict, output_dir: Path) -> Path:
    """Write evaluation results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reference-use evaluation: compare correct/wrong/blank reference conditions.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to evaluate")
    parser.add_argument("--output-dir", default=None, help="Output directory for metrics")
    parser.add_argument("--json", action="store_true", help="Print JSON output only")
    known, rest = parser.parse_known_args(argv)

    config = parse_config(rest)
    report = run_ref_use_eval(config, checkpoint=known.checkpoint)

    out_dir = Path(known.output_dir) if known.output_dir else Path(config.paths().out) / "ref_use_eval"
    metrics_path = write_eval_output(report, out_dir)

    if known.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        summary = report.get("summary", {})
        print(f"Reference-use evaluation: {metrics_path}")
        print(f"  Checkpoint: {report.get('checkpoint')}")
        print(f"  Frames: {report['frames']}, Backend: {report['backend']}")
        print(f"  Eval items: {report['num_eval_items']}")
        if summary:
            print(f"  Correct MSE:  {summary.get('correct_mse_mean', 0):.6f}")
            print(f"  Wrong MSE:    {summary.get('wrong_mse_mean', 0):.6f}")
            print(f"  Blank MSE:    {summary.get('blank_mse_mean', 0):.6f}")
            print(f"  Gap (wrong-correct): {summary.get('gap_correct_vs_wrong', 0):.6f}")
            print(f"  Gap (blank-correct): {summary.get('gap_correct_vs_blank', 0):.6f}")
            if "ref_used" in summary:
                print(f"  Reference used: {summary['ref_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
