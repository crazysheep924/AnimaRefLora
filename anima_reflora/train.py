from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .anima_caption import apply_tag_dropout, inject_identity_tags
from .cache import LatentCacheDataset, SameBucketBatchSampler, SyntheticLatentDataset, collate_latent_batch
from .checkpoints import load_checkpoint_into, load_optimizer_state, load_sidecar_into, save_checkpoint, save_optimizer_state
from .config import TrainConfig, parse_config
from .crepa import CrepaProjector, crepa_hidden_loss
from .features import (
    CcipEmbeddingCache,
    HeadRoiCache,
    check_resume_feature_compatibility,
    crepa_loss,
    feature_snapshot,
    write_json,
)
from .models import build_model
from .noise import (
    f1_anti_copy_loss,
    focal_frequency_loss,
    latent_reconstruction_l1,
    make_noised_batch,
    ref_diff_weight_map,
    target_frame_mse,
)
from .paths import run_dir
from .validation import validate_supported_training_features
from .version_meta import collect_runtime_versions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "fp32" or device.type == "cpu":
        return torch.float32
    if value == "fp16":
        return torch.float16
    return torch.bfloat16


def keep_scalar_trainables_fp32(model: torch.nn.Module) -> list[str]:
    kept = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim == 0 and param.dtype != torch.float32:
            param.data = param.data.float()
            kept.append(name)
    return kept


def make_run_dir(config: TrainConfig) -> Path:
    paths = config.paths()
    path = run_dir(paths.out, config.run_name)
    if path.exists() and not config.allow_existing_run:
        raise FileExistsError(f"Run folder already exists: {path}")
    for child in ["checkpoints", "tb", "viz", "ref_use", "logs"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    data["_runtime"] = collect_runtime_versions(config.sd_scripts)
    with (path / "logs" / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


def build_dataloader(config: TrainConfig) -> DataLoader:
    if config.synthetic_data:
        dataset = SyntheticLatentDataset(length=config.max_train_items or max(config.batch * 8, 8), frames=config.frames, seed=config.seed)
        return DataLoader(dataset, batch_size=config.batch, shuffle=True, num_workers=config.num_workers, collate_fn=collate_latent_batch)
    paths = config.paths()
    if paths.latcache == paths.latcache_512:
        raise ValueError(f"Training latent cache points at the 512 reference cache: {paths.latcache}")
    if config.frames == 3 and config.build_missing_head_cache:
        from .head_cache import build_missing_head_cache

        build_missing_head_cache(config, dry_run=False)
    dataset = LatentCacheDataset(
        paths.latcache,
        frames=config.frames,
        max_items=config.max_train_items,
        prompt_mode=config.prompt_mode,
        seed=config.seed,
        pair_dhash_cache=config.pair_dhash_cache,
        pair_min_dhash=config.pair_min_dhash,
        singleton_ref_mode=config.singleton_ref_mode,
    )
    sampler = SameBucketBatchSampler(dataset.pairs, batch_size=config.batch, shuffle=True, seed=config.seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config.num_workers, collate_fn=collate_latent_batch)


def safe_log_line(log_file: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line to train.log without ever raising.

    Diagnostics must never kill an unattended run: a full disk while writing
    viz once burned a 105k-step run, and these appends run every log_every
    steps on the same volume — an unguarded ENOSPC here would kill the run
    within steps of the disk filling. Falls back to stdout, and tolerates
    stdout itself being a redirect onto the same full disk.
    """
    line = json.dumps(payload, sort_keys=True)
    try:
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as err:  # noqa: BLE001
        _safe_print(f"WARNING: train.log write failed ({err}): {line}")


def _safe_print(message: str) -> None:
    try:
        print(message)
    except Exception:  # noqa: BLE001
        pass


def _inject_key(path: str) -> str:
    """Map a latcache target path to the identity-inject-map key: the danbooru id
    (basename without extension). Mirrors the keying in build_identity_inject_map.py."""
    base = os.path.basename(str(path).replace("\\", "/"))
    return os.path.splitext(base)[0]


def cycle_loader(loader: DataLoader, skip: int = 0):
    """Yield batches forever, re-iterating the DataLoader each epoch so the
    SameBucketBatchSampler reshuffles per epoch (seed+epoch).

    Replaces ``itertools.cycle(loader)`` which (1) caches every batch of the
    first pass in RAM until one full epoch completes — for a dataset larger than
    the steps run that accumulates ~one epoch of latents and can OOM-kill the
    process with no traceback — and (2) freezes the epoch-0 order forever (no
    reshuffle). On resume, fast-forward past ``skip`` already-consumed steps so
    data coverage continues from the checkpoint instead of restarting at order 0.
    """
    produced = 0
    while True:
        for batch in loader:
            produced += 1
            if produced <= skip:
                continue
            yield batch


def build_optimizer(config: TrainConfig, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    params = [p for p in params if p.requires_grad]
    if config.optimizer.lower() == "came":
        try:
            from pytorch_optimizer import CAME

            return CAME(params, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999, 0.9999))
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def group_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm over a parameter group's grads (matches clip_grad_norm_ aggregation)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).cpu()) ** 2
    return total**0.5


def trainable_param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def changed_trainable_params(before: dict[str, torch.Tensor], model: torch.nn.Module) -> int:
    changed = 0
    current_params = dict(model.named_parameters())
    for name, old in before.items():
        current = current_params.get(name)
        if current is not None and current.shape == old.shape and not torch.allclose(current.detach().cpu(), old):
            changed += 1
    return changed


def maybe_load_resume(config: TrainConfig, model: torch.nn.Module, log_file: Path) -> Path | None:
    candidates = []
    if config.resume:
        candidates.append(config.resume)
    elif not config.from_scratch and config.base_ckpt:
        candidates.append(config.base_ckpt)
    for path in candidates:
        if not path or not Path(path).exists():
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(f"checkpoint skipped missing={path}\n")
            continue
        before = trainable_param_snapshot(model) if config.resume else {}
        missing, unexpected = load_checkpoint_into(model, path, strict=False)
        changed = changed_trainable_params(before, model) if before else None
        if config.resume and before and changed == 0:
            raise RuntimeError(f"Explicit resume did not change any trainable parameters: {path}")
        with log_file.open("a", encoding="utf-8") as fh:
            changed_text = "" if changed is None else f" changed_trainable={changed}"
            fh.write(f"checkpoint loaded={path} missing={len(missing)} unexpected={len(unexpected)}{changed_text}\n")
        return Path(path)
    return None


def maybe_load_optimizer_resume(config: TrainConfig, optimizer: torch.optim.Optimizer, loaded_path: Path | None, device: torch.device, log_file: Path) -> None:
    if not config.resume or loaded_path is None:
        return
    opt_path = load_optimizer_state(optimizer, loaded_path, device=device)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"optimizer loaded={opt_path}\n")


def feature_config_dict(config: TrainConfig) -> dict:
    return asdict(feature_snapshot(config))


def config_for_checkpoint(config: TrainConfig) -> dict:
    data = config.to_dict()
    data["_feature_snapshot"] = feature_config_dict(config)
    return data


def load_feature_sidecars(config: TrainConfig, model: torch.nn.Module, crepa_projector: CrepaProjector | None, loaded_path: Path | None, log_file: Path) -> None:
    if loaded_path is None:
        return
    explicit_resume = bool(config.resume)
    sidecars: dict[str, torch.nn.Module] = {}
    ref_conditioner = getattr(model, "ref_conditioner", None)
    if isinstance(ref_conditioner, torch.nn.Module):
        sidecars["ref_conditioner"] = ref_conditioner
    cpm_adapter = getattr(model, "cpm_adapter", None)
    if isinstance(cpm_adapter, torch.nn.Module):
        sidecars["cpm_adapter"] = cpm_adapter
    rope_refpos = getattr(model, "rope_refpos", None)
    if isinstance(rope_refpos, torch.nn.Module):
        sidecars["rope_refpos"] = rope_refpos
    if crepa_projector is not None:
        sidecars["crepa_projector"] = crepa_projector
    for name, module in sidecars.items():
        try:
            missing, unexpected = load_sidecar_into(module, loaded_path, name, strict=False)
            line = f"sidecar loaded={name} missing={len(missing)} unexpected={len(unexpected)}\n"
        except FileNotFoundError:
            if explicit_resume:
                raise
            line = f"sidecar initialized={name} source_missing_for_warm_start={loaded_path}\n"
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line)


def update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, local_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    lr = base_lr * min(1.0, local_step / warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def dropout_stats(dropped_ref_frames: torch.Tensor, frames: int) -> dict[str, float]:
    dropped = dropped_ref_frames.detach().bool()
    if dropped.numel() == 0 or frames < 2:
        return {"dropout_total": 0.0, "dropout_head_only": 0.0, "dropout_full_only": 0.0, "dropout_both": 0.0}
    ref = dropped[:, : frames - 1]
    total = ref.any(dim=1)
    if frames == 3:
        head = ref[:, 0]
        full = ref[:, 1]
    else:
        head = torch.zeros_like(total)
        full = ref[:, 0]
    return {
        "dropout_total": float(total.float().mean().cpu()),
        "dropout_head_only": float((head & ~full).float().mean().cpu()),
        "dropout_full_only": float((full & ~head).float().mean().cpu()),
        "dropout_both": float((head & full).float().mean().cpu()),
        "dropout_head_stream": float(head.float().mean().cpu()),
        "dropout_full_stream": float(full.float().mean().cpu()),
    }


def artifact_metadata(config: TrainConfig, step: int, checkpoint_path: Path | None = None) -> dict:
    return {
        "run_name": config.run_name,
        "step": step,
        "checkpoint_step": step,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "seed": config.seed,
        "prompt": config.eval_prompt,
        "negative_prompt": config.negative_prompt,
        "sample_steps": config.ref_eval_steps,
        "guidance_scale": config.guidance_scale,
        "flow_shift": config.flow_shift,
        "ref_guidance_scale": config.ref_guidance_scale,
        "prompt_year": config.prompt_year,
        "prompt_mode": config.prompt_mode,
        "features": feature_config_dict(config),
        "image_tag": os.environ.get("ANIMA_REFLORA_IMAGE_TAG") or os.environ.get("RUNPOD_POD_ID"),
    }


def write_viz_artifacts(
    run_path: Path,
    step: int,
    batch: dict,
    noised: dict,
    prediction: torch.Tensor,
    config: TrainConfig,
    writer,
    checkpoint_path: Path | None = None,
) -> None:
    if config.no_viz:
        return
    out = run_path / "viz" / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        **artifact_metadata(config, step, checkpoint_path),
        "frame_roles": list(batch["frame_roles"]),
        "target_paths": batch["target_paths"][: min(4, len(batch["target_paths"]))],
        "ref_paths": batch["ref_paths"][: min(4, len(batch["ref_paths"]))],
        "caption_keys": batch.get("caption_keys", [])[: min(4, len(batch.get("caption_keys", [])))],
        "caption_sources": batch.get("caption_sources", [])[: min(4, len(batch.get("caption_sources", [])))],
        "condition_label": "train_batch_latent_panel",
    }
    write_json(out / "metadata.json", metadata)
    sample = {
        "clean": noised["clean"][:1].detach().cpu(),
        "x": noised["x"][:1].detach().cpu(),
        "prediction": prediction[:1].detach().cpu(),
        "target": noised["target"][:1].detach().cpu(),
    }
    torch.save(sample, out / "latent_panel.pt")
    if writer is not None:
        writer.add_text("viz/latest", json.dumps(metadata, sort_keys=True), step)


def write_ref_eval_artifacts(
    run_path: Path,
    step: int,
    batch: dict,
    metrics: dict[str, float],
    config: TrainConfig,
    writer,
    checkpoint_path: Path | None = None,
) -> None:
    if config.no_ref_eval:
        return
    out = run_path / "ref_use" / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in str(config.ref_eval_seeds).split(",") if value.strip()]
    payload = {
        **artifact_metadata(config, step, checkpoint_path),
        "eval_kind": "training_batch_proxy",
        "seeds": seeds,
        "ref_paths": batch["ref_paths"][: config.ref_eval_refs],
        "target_paths": batch["target_paths"][: config.ref_eval_refs],
        "caption_keys": batch.get("caption_keys", [])[: config.ref_eval_refs],
        "caption_sources": batch.get("caption_sources", [])[: config.ref_eval_refs],
        "condition_labels": ["train_batch_reference"],
        "frame_layout": list(batch["frame_roles"]),
        "metrics": metrics,
    }
    write_json(out / "metrics.json", payload)
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f"ref_use/{key}", value, step)


def train(config: TrainConfig) -> Path:
    validate_supported_training_features(config)
    check_resume_feature_compatibility(config)
    seed_everything(config.seed)
    if config.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_path = make_run_dir(config)
    log_file = run_path / "logs" / "train.log"
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    ccip_cache = CcipEmbeddingCache(config.ccip_cache) if config.cpm or config.crepa else None
    head_roi_path = Path(config.head_roi_cache)
    head_roi_cache = None
    if config.head_loss_weight != 1.0 or (config.crepa and config.crepa_pool == "head_roi") or config.f1_anti_copy_weight > 0:
        if not head_roi_path.exists():
            raise FileNotFoundError(f"Head ROI cache required by enabled feature: {head_roi_path}")
        head_roi_cache = HeadRoiCache(head_roi_path)
    loader = build_dataloader(config)
    # Re-iterating generator (not itertools.cycle): avoids the RAM leak of caching
    # a full epoch of batches, reshuffles per epoch, and fast-forwards on resume.
    # The fast-forward loads and discards start_step batches from disk (GPU idle,
    # can take tens of minutes on a network volume) — disable via
    # --no-resume-data-skip to start from epoch-0 order immediately.
    batches = cycle_loader(loader, skip=config.start_step if config.resume_data_skip else 0)
    model = build_model(config, ccip_dim=ccip_cache.dim if ccip_cache is not None and config.cpm else None).to(device=device, dtype=dtype)
    fp32_scalar_trainables = keep_scalar_trainables_fp32(model) if dtype != torch.float32 else []
    crepa_in_dim = int(getattr(model, "crepa_hidden_dim", 16))
    crepa_projector = (
        CrepaProjector(in_dim=crepa_in_dim, embedding_dim=ccip_cache.dim, block_index=config.crepa_block).to(device=device)
        if config.crepa and ccip_cache is not None
        else None
    )
    loaded_path = maybe_load_resume(config, model, log_file)
    load_feature_sidecars(config, model, crepa_projector, loaded_path, log_file)
    if fp32_scalar_trainables:
        keep_scalar_trainables_fp32(model)
    trainable_params: list[torch.nn.Parameter] = list(model.parameters())
    if crepa_projector is not None:
        trainable_params.extend(crepa_projector.parameters())
    optimizer = build_optimizer(config, trainable_params)
    maybe_load_optimizer_resume(config, optimizer, loaded_path, device, log_file)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "resume": str(loaded_path) if loaded_path else None,
                    "start_step": config.start_step,
                    "new_steps": max(config.steps - config.start_step, 0),
                    "target_step": config.steps,
                    "warmup_steps": config.warmup_steps,
                    "fp32_scalar_trainables": fp32_scalar_trainables,
                },
                sort_keys=True,
            )
            + "\n"
        )
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_path / "tb"))
    except Exception:
        writer = None

    gen = torch.Generator(device=device).manual_seed(config.seed)
    # Whole-caption CFG dropout: preload the unconditional (empty) caption embedding.
    empty_cap = None
    if config.caption_dropout_prob > 0:
        if config.empty_cap_path:
            empty_path = Path(config.empty_cap_path)
        elif config.ccip_cache:
            empty_path = Path(config.ccip_cache).parent / "empty_cap.pt"
        else:
            raise RuntimeError("caption_dropout_prob > 0 requires --empty-cap-path or --ccip-cache to locate empty_cap.pt")
        if not empty_path.exists():
            raise RuntimeError(f"caption dropout enabled but empty caption embedding not found: {empty_path}")
        _ec = torch.load(empty_path, map_location="cpu", weights_only=False)
        empty_cap = {
            "prompt_embeds": _ec["prompt_embeds"].to(device=device, dtype=dtype),
            "attn_mask": _ec["attn_mask"].to(device=device),
            "t5_input_ids": _ec["t5_input_ids"].to(device=device),
            "t5_attn_mask": _ec["t5_attn_mask"].to(device=device),
        }
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"caption_dropout enabled prob={config.caption_dropout_prob} empty_cap={empty_path}\n")
    # Identity-accessory injection map: {image-id-key: [accessory words (space form)]}.
    # Built offline by scripts/build_identity_inject_map.py; missing/unset is a no-op.
    inject_map = None
    if config.identity_inject_map and config.identity_inject_prob > 0:
        with open(config.identity_inject_map, encoding="utf-8") as fh:
            inject_map = json.load(fh)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(
                f"identity_inject enabled prob={config.identity_inject_prob} "
                f"map={config.identity_inject_map} entries={len(inject_map)}\n"
            )
    # Tag-level caption dropout: re-encode captions on the fly after dropping a random
    # subset of their delta tags. Reuses the same Qwen3 (0.6B) + T5 encoder as the cache
    # builder; the caption strings ride along in batch["caption_keys"]. The same encoder
    # serves identity injection, so build it if either feature is on.
    tag_encoder = None
    if config.tag_dropout_prob > 0 or inject_map is not None:
        from .build_training_cache import SdScriptsPromptEncoder

        tag_encoder = SdScriptsPromptEncoder(config, device=device, dtype=dtype)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(
                f"tag_dropout enabled prob={config.tag_dropout_prob} keep_prob={config.tag_keep_prob} "
                f"keep_min={config.tag_keep_min}\n"
            )
    # --steps is the ABSOLUTE target step; we run start_step+1 .. steps (inclusive).
    progress = tqdm(range(config.start_step + 1, config.steps + 1), desc=f"train {config.run_name}", dynamic_ncols=True)
    final_ckpt: Path | None = None
    skipped_total = 0
    loss_ema: float | None = None
    extra_sidecars = {"crepa_projector": crepa_projector} if crepa_projector is not None else {}
    for global_step in progress:
        step_start = time.perf_counter()
        local_step = global_step - config.start_step
        lr = update_warmup_lr(optimizer, config.lr, local_step, config.warmup_steps)
        batch = next(batches)
        clean = batch["clean"].to(device=device, dtype=dtype, non_blocking=True)
        prompt = batch["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        attn_mask = batch["attn_mask"].to(device=device, non_blocking=True)
        batch["t5_input_ids"] = batch["t5_input_ids"].to(device=device, non_blocking=True)
        batch["t5_attn_mask"] = batch["t5_attn_mask"].to(device=device, non_blocking=True)
        # Tag-level caption dropout + identity-accessory injection. Both mutate the
        # caption string and require an on-the-fly re-encode. Dropout drops a random
        # subset of delta tags (short-prompt training); injection re-inserts a
        # per-image, GT-faithful subset of signature accessory words that sig_subtract
        # had stripped (controllable prompt handle). Injection is decoupled from the
        # dropout gate — an inject-only step still re-encodes.
        tag_dropped = False
        reencoded = False
        if tag_encoder is not None:
            step_rng = random.Random(config.seed + global_step)
            do_dropout = step_rng.random() < config.tag_dropout_prob
            inject_on = inject_map is not None and config.identity_inject_prob > 0
            if do_dropout or inject_on:
                new_caps = []
                for cap, tpath in zip(batch["caption_keys"], batch["target_paths"]):
                    c = cap
                    if do_dropout:
                        c = apply_tag_dropout(c, config.tag_keep_prob, config.tag_keep_min, step_rng)
                    if inject_on:
                        words = inject_map.get(_inject_key(tpath))
                        if words:
                            c = inject_identity_tags(c, words, config.identity_inject_prob, step_rng)
                    new_caps.append(c)
                if any(c != cap for c, cap in zip(new_caps, batch["caption_keys"])):
                    enc = tag_encoder.encode(new_caps)
                    prompt = torch.stack([enc[c]["prompt_embeds"][0] for c in new_caps]).to(device=device, dtype=dtype)
                    attn_mask = torch.stack([enc[c]["attn_mask"][0] for c in new_caps]).to(device=device)
                    batch["t5_input_ids"] = torch.stack([enc[c]["t5_input_ids"][0] for c in new_caps]).to(device=device)
                    batch["t5_attn_mask"] = torch.stack([enc[c]["t5_attn_mask"][0] for c in new_caps]).to(device=device)
                    reencoded = True
                    tag_dropped = do_dropout
        # Whole-caption CFG dropout: swap the entire caption for the unconditional
        # embedding (text-only). Leaves CPM identity (ccip_embeddings/cpm_valid) intact.
        # Mutually exclusive with any re-encode (a re-encoded step is never also blanked).
        caption_dropped = False
        if empty_cap is not None and not reencoded and float(torch.rand((), generator=gen, device=device)) < config.caption_dropout_prob:
            bsz = clean.shape[0]
            prompt = empty_cap["prompt_embeds"].expand(bsz, -1, -1).contiguous()
            attn_mask = empty_cap["attn_mask"].expand(bsz, -1).contiguous()
            batch["t5_input_ids"] = empty_cap["t5_input_ids"].expand(bsz, -1).contiguous()
            batch["t5_attn_mask"] = empty_cap["t5_attn_mask"].expand(bsz, -1).contiguous()
            caption_dropped = True
        noised = make_noised_batch(
            clean,
            high_sigma_mix_prob=config.high_sigma_mix_prob,
            high_sigma_min=config.high_sigma_min,
            high_sigma_max=config.high_sigma_max,
            ref_dropout_prob=config.ref_dropout_prob,
            ref_dropout_mode=config.ref_dropout_mode,
            ref_dropout_t3_mode=config.ref_dropout_t3_mode,
            generator=gen,
            timestep_sampling=config.timestep_sampling,
            sigmoid_scale=config.sigmoid_scale,
            force_ref_blank=batch.get("force_ref_blank"),
        )
        ccip_valid = None
        if ccip_cache is not None:
            ccip_embeddings, ccip_valid_cpu = ccip_cache.gather(batch["ref_paths"])
            ccip_valid = ccip_valid_cpu.to(device=device)
            full_ref_idx = max(config.frames - 2, 0)
            dropped = noised["dropped_ref_frames"].to(device=device)
            cpm_valid = ccip_valid & ~dropped[:, full_ref_idx]
            batch["ccip_embeddings"] = ccip_embeddings.to(device=device)
            batch["ccip_valid"] = ccip_valid
            batch["cpm_valid"] = cpm_valid
        head_mask = None
        head_roi_valid = None
        if head_roi_cache is not None:
            head_mask, head_roi_valid = head_roi_cache.gather(
                batch["target_paths"],
                height=clean.shape[-2],
                width=clean.shape[-1],
                device=device,
                dtype=dtype,
            )
            batch["head_mask"] = head_mask
            batch["head_roi_valid"] = head_roi_valid
        clear_crepa = getattr(model, "clear_crepa_hidden", None)
        if callable(clear_crepa):
            clear_crepa()
        prediction = model(
            noised["x"],
            noised["timesteps"],
            caption_embeds=prompt,
            attention_mask=attn_mask,
            batch=batch,
            config=config,
        )
        diff_weight = ref_diff_weight_map(clean, config.diff_loss_lambda, config.diff_weight_min, head_mask=head_mask)
        if diff_weight is not None:
            # The map is derived from the full ref; when that frame was blanked
            # by ref dropout the model can't see it, and keeping the weighting
            # would bias the reference-unconditional branch (matched regions
            # correlate with character-common content in same-character pairs).
            full_ref_dropped = noised["dropped_ref_frames"][:, max(config.frames - 2, 0)].to(device=diff_weight.device).bool()
            if bool(full_ref_dropped.any()):
                diff_weight[full_ref_dropped] = 1.0
        loss = target_frame_mse(
            prediction,
            noised["target"],
            sigmas=noised["sigmas"],
            weighting_scheme=config.weighting_scheme,
            min_snr_gamma=config.min_snr_gamma,
            head_mask=head_mask,
            head_loss_weight=config.head_loss_weight,
            head_sigma_cutoff=config.head_sigma_cutoff,
            extra_weight_map=diff_weight,
        )
        loss_components: dict[str, float] = {"mse": float(loss.detach().cpu())}
        if diff_weight is not None:
            loss_components["diff_weight_std"] = float(diff_weight.std().detach().cpu())
        loss_components["caption_dropped"] = 1.0 if caption_dropped else 0.0
        loss_components["tag_dropped"] = 1.0 if tag_dropped else 0.0
        fb = batch.get("force_ref_blank")
        if fb is not None:
            loss_components["singleton_blanked"] = float(fb.float().mean())
        if config.latent_recon_loss_weight > 0:
            latent_recon = latent_reconstruction_l1(prediction, noised["x"], clean, noised["sigmas"], extra_weight_map=diff_weight)
            loss = loss + config.latent_recon_loss_weight * latent_recon
            loss_components["latent_recon_l1"] = float(latent_recon.detach().cpu())
        if config.f1_anti_copy_weight > 0 and head_mask is not None:
            anti_copy = f1_anti_copy_loss(
                prediction,
                noised["x"],
                clean,
                noised["sigmas"],
                noised["dropped_ref_frames"],
                head_mask,
                margin=config.f1_anti_copy_margin,
                sigma_cutoff=config.f1_anti_copy_sigma_cutoff,
                head_roi_valid=head_roi_valid,
            )
            loss = loss + config.f1_anti_copy_weight * anti_copy
            loss_components["f1_anti_copy"] = float(anti_copy.detach().cpu())
        if crepa_projector is not None and ccip_valid is not None:
            crepa_sigmas = noised["sigmas"].to(device=device)
            if config.crepa_sigma_cutoff > 0:
                crepa_valid = ccip_valid & (crepa_sigmas <= config.crepa_sigma_cutoff)
            else:
                crepa_valid = ccip_valid
            hidden = getattr(model, "crepa_hidden", None)
            if hidden is not None:
                aux_loss, crepa_metrics = crepa_hidden_loss(
                    crepa_projector,
                    hidden,
                    batch["ccip_embeddings"],
                    crepa_valid,
                    frames=config.frames,
                    sigmas=crepa_sigmas,
                    sigma_cutoff=config.crepa_sigma_cutoff,
                    head_mask=head_mask,
                    pool=config.crepa_pool,
                )
            else:
                aux_loss, crepa_metrics = crepa_loss(
                    crepa_projector,
                    prediction,
                    batch["ccip_embeddings"],
                    crepa_valid,
                    head_mask=head_mask,
                    pool=config.crepa_pool,
                )
            loss = loss + config.crepa_lambda * aux_loss
            loss_components["crepa"] = float(aux_loss.detach().cpu())
            loss_components.update(crepa_metrics)
        if config.ffl_weight > 0:
            ffl = focal_frequency_loss(prediction, noised["target"], alpha=config.ffl_alpha)
            loss = loss + config.ffl_weight * ffl
            loss_components["ffl"] = float(ffl.detach().cpu())
        if not torch.isfinite(loss):
            if config.strict_nonfinite:
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")
            # Never kill an unattended run on one bad batch: skip, count, keep going.
            optimizer.zero_grad(set_to_none=True)
            skipped_total += 1
            safe_log_line(log_file, {"step": global_step, "skipped": 1, "reason": "nonfinite_loss", "skipped_total": skipped_total})
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        cpm_gate = None
        cpm_adapter = getattr(model, "cpm_adapter", None)
        if cpm_adapter is not None:
            gate = getattr(cpm_adapter, "gate", None)
            if gate is not None:
                cpm_gate = float(gate.detach().cpu())
                if gate.grad is not None:
                    loss_components["cpm_gate_grad"] = float(gate.grad.detach().abs().cpu())
        ref_conditioner = getattr(model, "ref_conditioner", None)
        if ref_conditioner is not None:
            scalar_stats = getattr(ref_conditioner, "scalar_stats", None)
            if callable(scalar_stats):
                loss_components.update({f"adapter/{key}": value for key, value in scalar_stats().items()})
            gate_grads = [
                float(param.grad.detach().abs().mean().cpu())
                for name, param in ref_conditioner.named_parameters()
                if name.endswith("gate") and param.grad is not None
            ]
            if gate_grads:
                loss_components["adapter/gate_grad_mean"] = sum(gate_grads) / len(gate_grads)
        if not torch.isfinite(grad_norm):
            if config.strict_nonfinite:
                raise FloatingPointError(f"Non-finite grad-norm at step {global_step}: {grad_norm}")
            optimizer.zero_grad(set_to_none=True)
            skipped_total += 1
            safe_log_line(log_file, {"step": global_step, "skipped": 1, "reason": "nonfinite_grad", "skipped_total": skipped_total})
            continue
        optimizer.step()

        loss_value_ema = float(loss.detach().cpu())
        loss_ema = loss_value_ema if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_value_ema

        if global_step % config.log_every == 0:
            # Per-group grad norms: separate the LoKr/LoRA adapter from the
            # ref_conditioner and crepa projector so we can see which one is
            # actually receiving gradient. Grads are still live here (post-step,
            # pre-zero_grad). clip_grad_norm_ (max_norm=1.0) only rescales when
            # the total exceeds 1.0, so these match the unclipped grads in
            # practice for this run.
            network = getattr(model, "network", None)
            if network is not None:
                loss_components["grad_norm/lora"] = group_grad_norm(network.parameters())
            if ref_conditioner is not None:
                loss_components["grad_norm/ref_conditioner"] = group_grad_norm(ref_conditioner.parameters())
            if crepa_projector is not None:
                loss_components["grad_norm/crepa_projector"] = group_grad_norm(crepa_projector.parameters())
            loss_value = float(loss.detach().cpu())
            progress.set_postfix(loss=f"{loss_value:.5f}", sigma=f"{float(noised['sigmas'].mean().cpu()):.3f}")
            stats = {
                "step": global_step,
                "loss": loss_value,
                "loss_ema": loss_ema,
                "skipped_total": skipped_total,
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu()),
                "sigma_mean": float(noised["sigmas"].mean().detach().cpu()),
                "sigma_high_fraction": float((noised["sigmas"] >= config.high_sigma_min).float().mean().detach().cpu()),
                "step_time_sec": time.perf_counter() - step_start,
                "rope_refpos": float(config.rope_refpos),
                **dropout_stats(noised["dropped_ref_frames"], config.frames),
                **loss_components,
            }
            if ccip_valid is not None:
                stats["ccip_valid_fraction"] = float(ccip_valid.float().mean().detach().cpu())
                stats["cpm_valid_fraction"] = float(batch["cpm_valid"].float().mean().detach().cpu())
            if cpm_gate is not None:
                stats["cpm_gate"] = cpm_gate
            if head_roi_valid is not None and head_mask is not None:
                stats["head_roi_valid_fraction"] = float(head_roi_valid.float().mean().detach().cpu())
                stats["head_roi_mask_fraction"] = float((head_mask > 0).float().mean().detach().cpu())
            safe_log_line(log_file, stats)
            if writer is not None:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/sigma", float(noised["sigmas"].mean().detach().cpu()), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu()), global_step)
                for key, value in stats.items():
                    if isinstance(value, (int, float)) and key not in {"step", "loss", "sigma_mean"}:
                        writer.add_scalar(f"train/{key}", float(value), global_step)
        if config.ckpt_every > 0 and global_step % config.ckpt_every == 0:
            final_ckpt = save_checkpoint(model, run_path, global_step, config_for_checkpoint(config), extra_sidecars=extra_sidecars)
            save_optimizer_state(optimizer, final_ckpt)
            # Diagnostics must never kill an unattended run: a full disk while
            # writing viz once burned a 105k-step run whose checkpoint had
            # already saved fine. Log and keep training.
            try:
                write_viz_artifacts(run_path, global_step, batch, noised, prediction, config, writer, checkpoint_path=final_ckpt)
                if not config.no_ref_eval and config.ref_eval_every > 0 and global_step % config.ref_eval_every == 0:
                    write_ref_eval_artifacts(run_path, global_step, batch, loss_components, config, writer, checkpoint_path=final_ckpt)
            except Exception as artifact_err:  # noqa: BLE001
                safe_log_line(log_file, {"step": global_step, "artifact_write_failed": str(artifact_err)})
                _safe_print(f"WARNING step {global_step}: artifact write failed, training continues: {artifact_err}")
        elif not config.no_ref_eval and config.ref_eval_every > 0 and global_step % config.ref_eval_every == 0:
            try:
                write_ref_eval_artifacts(run_path, global_step, batch, loss_components, config, writer)
            except Exception as artifact_err:  # noqa: BLE001
                safe_log_line(log_file, {"step": global_step, "artifact_write_failed": str(artifact_err)})
                _safe_print(f"WARNING step {global_step}: artifact write failed, training continues: {artifact_err}")
    steps_run = config.steps - config.start_step
    if steps_run <= 0 or final_ckpt is None or config.steps % max(config.ckpt_every, 1) != 0:
        final_step = config.steps
        final_ckpt = save_checkpoint(model, run_path, final_step, config_for_checkpoint(config), extra_sidecars=extra_sidecars)
        save_optimizer_state(optimizer, final_ckpt)
        if steps_run > 0:
            write_viz_artifacts(run_path, final_step, batch, noised, prediction, config, writer, checkpoint_path=final_ckpt)
    if writer is not None:
        writer.flush()
        writer.close()
    with (run_path / "logs" / "done.json").open("w", encoding="utf-8") as fh:
        json.dump({"checkpoint": str(final_ckpt), "steps": config.steps}, fh, indent=2)
    return run_path


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    if config.stage == "plan":
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    run_path = train(config)
    print(f"run_dir={run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
