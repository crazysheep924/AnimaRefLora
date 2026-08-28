from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

from .cache import LatentCacheIndex, build_pairs
from .checkpoints import optimizer_state_path
from .config import parse_config
from .features import CcipEmbeddingCache, HeadRoiCache, feature_sidecar_path
from .paths import run_dir
from .validation import validate_supported_training_features
from .version_meta import collect_runtime_versions


IDENTITY_CAPTION_HINTS = ("hair", "eyes", "skin", "watermark", "signature", "artist", "username", "character")


def check_file(path: Path, label: str, required: bool = True) -> dict:
    exists = path.exists()
    if required and not exists:
        raise FileNotFoundError(f"{label} not found: {path}")
    return {"label": label, "path": str(path), "exists": exists}


def identity_caption_examples(captions: list[str], limit: int = 10) -> list[str]:
    found = []
    for caption in captions:
        lowered = caption.lower().replace("_", " ")
        if any(hint in lowered for hint in IDENTITY_CAPTION_HINTS):
            found.append(caption)
            if len(found) >= limit:
                break
    return found


def run_preflight(argv: list[str] | None = None) -> dict:
    config = parse_config(argv)
    if config.stage not in {"plan", "tests"}:
        validate_supported_training_features(config)
    paths = config.paths()
    paths.ensure_text_encoder_symlink()
    sd_scripts = Path(config.sd_scripts or paths.sd_scripts)
    runtime_versions = collect_runtime_versions(str(sd_scripts))
    expected_commit = os.environ.get("SD_SCRIPTS_COMMIT")
    detected_commit = runtime_versions.get("sd_scripts_commit")
    commit_mismatch = bool(
        expected_commit and detected_commit
        and expected_commit != detected_commit
    )
    report = {
        "cuda": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "runtime_versions": runtime_versions,
        "sd_scripts_commit_mismatch": commit_mismatch,
        "checks": [],
    }
    report["checks"].append(check_file(sd_scripts / "anima_train_network.py", "sd-scripts Anima trainer", required=config.backend == "external"))
    if config.resume:
        resume_path = Path(config.resume)
        report["checks"].append(check_file(resume_path, "resume checkpoint"))
        report["checks"].append(check_file(optimizer_state_path(resume_path), "resume optimizer state"))
        report["checks"].append(check_file(feature_sidecar_path(resume_path), "resume feature sidecar"))
    if not config.synthetic_data:
        if paths.latcache == paths.latcache_512:
            raise ValueError(f"Training latent cache points at the 512 reference cache: {paths.latcache}")
        report["checks"].append(check_file(paths.model_dit, "base DiT"))
        report["checks"].append(check_file(paths.model_vae, "VAE"))
        report["checks"].append(check_file(paths.model_te / "qwen_3_06b_base.safetensors", "text encoder"))
        report["checks"].append(check_file(paths.latcache / "_cache_index.pkl", "latent cache index"))
        index = LatentCacheIndex.load(paths.latcache, prompt_mode=config.prompt_mode)
        from .head_cache import missing_head_records

        full_records = [record for record in index.records if record.kind == "full"]
        caption_sources = Counter(record.caption_source for record in full_records)
        identity_examples = identity_caption_examples([record.caption for record in full_records])
        report["caption_policy"] = {
            "prompt_mode": config.prompt_mode,
            "strict_change_captions": config.strict_change_captions,
            "source_distribution": dict(caption_sources),
            "identity_hint_examples": identity_examples,
        }
        if config.strict_change_captions and identity_examples:
            raise ValueError(f"Change-only caption policy found identity-like caption tags: {identity_examples[:3]}")
        report["latent_cache"] = {
            "records": len(index.records),
            "shards": len(index.shards),
            "version": index.index.get("version"),
            "caption_entries": len(index.index.get("cap_idx", {})),
            "missing_ref_head_latents": len(missing_head_records(index)),
            "bucket_distribution": {str(k): v for k, v in Counter(record.bucket for record in index.records if record.kind == "full").items()},
            "ref_eligible_full": sum(1 for record in index.records if record.kind == "full" and record.ref_eligible),
            "head_records": sum(1 for record in index.records if record.kind == "head"),
        }
        pairs = build_pairs(index, frames=config.frames, require_head_ref=config.frames == 3)
        if config.max_train_items is not None:
            pairs = pairs[: config.max_train_items]
        report["latent_cache"]["train_pairs"] = len(pairs)
        if config.cpm or config.crepa:
            report["checks"].append(check_file(Path(config.ccip_cache), "CCIP cache"))
            ccip = CcipEmbeddingCache(config.ccip_cache)
            ref_paths = sorted(
                {
                    candidate.path
                    for pair in pairs
                    for candidate in (pair.ref_candidates or (pair.ref_full,))
                }
            )
            coverage = ccip.coverage(ref_paths)
            report["ccip_cache"] = coverage
            if coverage["missing"]:
                raise ValueError(f"CCIP cache missing {coverage['missing']} reference embeddings; examples={coverage['missing_examples']}")
        if config.head_loss_weight != 1.0 or (config.crepa and config.crepa_pool == "head_roi"):
            head_roi_path = Path(config.head_roi_cache)
            report["checks"].append(check_file(head_roi_path, "head ROI cache"))
            head_roi = HeadRoiCache(head_roi_path)
            coverage = head_roi.coverage([pair.target_full.path for pair in pairs])
            report["head_roi_cache"] = coverage
            if coverage["missing"]:
                missing_frac = coverage["missing"] / max(1, len(pairs))
                msg = (
                    f"Head ROI cache missing {coverage['missing']}/{len(pairs)} target masks "
                    f"({missing_frac:.1%}); examples={coverage['missing_examples']}"
                )
                # Targets with no detectable head get no mask; training falls back to uniform
                # head weighting for them (HeadRoiCache.gather -> valid=False), so a small
                # natural miss rate is fine. Only fail if so many are missing the cache is wrong.
                if missing_frac > 0.10:
                    raise ValueError(msg + " -- missing fraction too high; rebuild the head ROI cache")
                print(f"[preflight][warn] {msg} (tolerated; uniform head weight for those targets)")
    elif config.cpm or config.crepa:
        report["checks"].append(check_file(Path(config.ccip_cache), "CCIP cache"))
        ccip = CcipEmbeddingCache(config.ccip_cache)
        report["ccip_cache"] = {"cache": str(ccip.path), "entries": len(ccip.embeddings), "embedding_dim": ccip.dim}
    target_run = run_dir(paths.out, config.run_name)
    if target_run.exists() and not config.allow_existing_run:
        raise FileExistsError(f"Run folder already exists: {target_run}")
    report["run_dir"] = str(target_run)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--json", action="store_true")
    known, rest = parser.parse_known_args(argv)
    report = run_preflight(rest)
    if known.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("preflight ok")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
