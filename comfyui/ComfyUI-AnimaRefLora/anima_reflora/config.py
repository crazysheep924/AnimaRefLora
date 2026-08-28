from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import AnimaPaths


WARM_START_CKPT = "/opt/AnimaRefLora/runs/experiments/v2a_highsigma_adaptergate/checkpoints/lora_step_30000.safetensors"
ANIMA_POSITIVE_PREFIX = "masterpiece, best quality, score_9, newest"
ANIMA_DEFAULT_EVAL_PROMPT = "standing, cowboy shot, white dress, simple background, looking at viewer"
ANIMA_NEGATIVE_PROMPT = (
    "worst quality,low quality,score_1,score_2,score_3,artist name,simple background,lowres,(bad),text,error,"
    "fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,"
    "signature,artistic error"
)

STAGE_DEFAULT_STEPS = {
    "plan": 0,
    "tests": 0,
    "preflight": 0,
    "headroi-short": 35000,
    "head-sigma-short": 35000,
    "dropout-short": 35000,
    "rope-smoke": 20,
    "rope-short": 35000,
    "cpm-preflight": 0,
    "cpm-smoke": 20,
    "cpm-short": 35000,
    "rope-cpm-short": 35000,
    "combo-long": 100000,
    "from0-headroi-rope-cpm": 100000,
}


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_str(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None or value == "" else str_to_bool(value)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_step_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"(?:step_|_)(\d+)(?:\.|$)", path)
    return int(match.group(1)) if match else None


@dataclass
class TrainConfig:
    stage: str
    run_name: str
    run_tag: str
    steps: int
    batch: int
    lr: float
    seed: int
    network: str
    network_dim: int
    network_alpha: float
    network_module: str | None
    network_args: list[str]
    optimizer: str
    weight_decay: float
    storage: str
    out_dir: str
    allow_existing_run: bool
    base_ckpt: str
    from_scratch: bool
    resume: str | None
    start_step: int
    warmup_steps: int
    resume_ref_conditioner: str | None
    resume_crepa: str | None
    ckpt_every: int
    smoke_ckpt_every: int
    log_every: int
    no_viz: bool
    ref_eval_every: int
    no_ref_eval: bool
    ref_eval_refs: int
    ref_eval_steps: int
    ref_eval_seeds: str
    prompt_mode: str
    strict_change_captions: bool
    eval_prompt: str
    negative_prompt: str
    guidance_scale: float
    flow_shift: float
    ref_guidance_scale: float
    prompt_year: int
    grad_checkpoint: bool
    tf32: bool
    python_bin: str
    frames: int
    timestep_sampling: str
    sigmoid_scale: float
    high_sigma_mix_prob: float
    high_sigma_min: float
    high_sigma_max: float
    ref_dropout_prob: float
    ref_dropout_mode: str
    ref_dropout_t3_mode: str
    caption_dropout_prob: float
    empty_cap_path: str | None
    tag_dropout_prob: float
    tag_keep_prob: float
    tag_keep_min: int
    identity_inject_map: str | None
    identity_inject_prob: float
    head_loss_weight: float
    head_sigma_cutoff: float
    head_conf: float
    head_margin: float
    head_loss_min_lat: int
    head_loss_max_lat: int
    weighting_scheme: str
    min_snr_gamma: float
    latent_recon_loss_weight: float
    f1_anti_copy_weight: float
    f1_anti_copy_margin: float
    f1_anti_copy_sigma_cutoff: float
    diff_loss_lambda: float
    diff_weight_min: float
    resume_data_skip: bool
    pair_dhash_cache: str
    pair_min_dhash: int
    singleton_ref_mode: str
    strict_nonfinite: bool
    no_ref_conditioner: bool
    no_adapter_gate: bool
    cpm: bool
    no_cpm_train_emb: bool
    ccip_cache: str
    head_roi_cache: str
    cpm_identity_dim: int
    cpm_tokens: int
    adapter_blocks: str
    adapter_dim: int
    adapter_heads: int
    crepa: bool
    crepa_lambda: float
    crepa_block: int
    crepa_sigma_cutoff: float
    crepa_pool: str
    rope_refpos: bool
    rope_layout: str
    rope_shift: float
    ffl_weight: float
    ffl_alpha: float
    train_args: list[str] = field(default_factory=list)
    raw_train_args: list[str] = field(default_factory=list)
    extra_train_args: list[str] = field(default_factory=list)
    backend: str = "external"
    model_factory: str | None = None
    sd_scripts: str | None = None
    attn_mode: str = "torch"
    split_attn: bool = False
    synthetic_data: bool = False
    num_workers: int = 0
    device: str = "auto"
    dtype: str = "bf16"
    max_train_items: int | None = None
    build_missing_head_cache: bool = True
    head_crop_conf: float = 0.4
    head_crop_padding: float = 1.8
    head_cache_shard: str = "shard_auto_head.pt"
    image_root: str | None = None
    image_source_prefix: str | None = None

    def paths(self) -> AnimaPaths:
        return AnimaPaths.from_env(storage=self.storage, out_dir=self.out_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_raw_args(argv: list[str] | None) -> tuple[list[str], list[str]]:
    args = list(argv) if argv is not None else None
    if args is None:
        import sys

        args = sys.argv[1:]
    if "--" not in args:
        return args, []
    idx = args.index("--")
    return args[:idx], args[idx + 1 :]


def normalize_train_arg_values(args: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--train-arg":
            if i + 1 >= len(args):
                raise SystemExit("--train-arg requires a value")
            normalized.append(f"--train-arg={args[i + 1]}")
            i += 2
            continue
        normalized.append(args[i])
        i += 1
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Anima reference LoRA experiments.")
    parser.add_argument("stage_pos", nargs="?", help="Stage name. Equivalent to --stage.")
    parser.add_argument("--stage", default=env_str("STAGE"))
    parser.add_argument("--run-name", default=env_str("RUN_NAME"))
    parser.add_argument("--run-tag", default=env_str("RUN_TAG"))
    parser.add_argument("--steps", type=int, default=env_int("STEPS"))
    parser.add_argument("--batch", type=int, default=env_int("BATCH", 1))
    parser.add_argument("--lr", type=float, default=env_float("LR", 1e-5))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 1234))
    parser.add_argument("--network", default=env_str("NETWORK", "lokr"))
    parser.add_argument("--network-dim", type=int, default=env_int("NETWORK_DIM", 512))
    parser.add_argument("--network-alpha", type=float, default=env_float("NETWORK_ALPHA", 512.0))
    parser.add_argument("--network-module", default=env_str("NETWORK_MODULE"))
    parser.add_argument("--network-arg", action="append", default=[])
    parser.add_argument("--optimizer", default=env_str("OPTIMIZER", "came"))
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", 0.0))
    parser.add_argument("--storage", default=env_str("ANIMA_REFLORA_STORAGE", "/workspace/storage"))
    parser.add_argument("--out-dir", default=env_str("ANIMA_REFLORA_OUT", "/opt/AnimaRefLora/runs"))
    parser.add_argument("--allow-existing-run", action="store_true", default=env_bool("ALLOW_EXISTING_RUN", False))

    parser.add_argument("--base-ckpt", default=env_str("BASE_CKPT", WARM_START_CKPT))
    parser.add_argument("--from-scratch", action="store_true", default=env_bool("FROM_SCRATCH", False))
    parser.add_argument("--resume", default=env_str("RESUME"))
    parser.add_argument("--start-step", type=int, default=env_int("START_STEP"))
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", 400))
    parser.add_argument("--resume-ref-conditioner", default=env_str("RESUME_REF_CONDITIONER"))
    parser.add_argument("--resume-crepa", default=env_str("RESUME_CREPA"))
    parser.add_argument("--ckpt-every", type=int, default=env_int("CKPT_EVERY", 1000))
    parser.add_argument("--smoke-ckpt-every", type=int, default=env_int("SMOKE_CKPT_EVERY", 500))

    parser.add_argument("--log-every", type=int, default=env_int("LOG_EVERY", 1))
    parser.add_argument("--no-viz", action="store_true", default=env_bool("NO_VIZ", False))
    parser.add_argument("--ref-eval-every", type=int, default=env_int("REF_EVAL_EVERY", 2500))
    parser.add_argument("--no-ref-eval", action="store_true", default=env_bool("NO_REF_EVAL", False))
    parser.add_argument("--ref-eval-refs", type=int, default=env_int("REF_EVAL_REFS", 5))
    parser.add_argument("--ref-eval-steps", type=int, default=env_int("REF_EVAL_STEPS", 24))
    parser.add_argument("--ref-eval-seeds", default=env_str("REF_EVAL_SEEDS", "0,1,2"))
    parser.add_argument("--prompt-mode", choices=["change_only", "target_caption"], default=env_str("PROMPT_MODE", "change_only"))
    parser.add_argument("--strict-change-captions", action="store_true", default=env_bool("STRICT_CHANGE_CAPTIONS", False))
    parser.add_argument("--eval-prompt", default=env_str("EVAL_PROMPT", ANIMA_DEFAULT_EVAL_PROMPT))
    parser.add_argument("--negative-prompt", default=env_str("NEGATIVE_PROMPT", ANIMA_NEGATIVE_PROMPT))
    parser.add_argument("--guidance-scale", type=float, default=env_float("GUIDANCE_SCALE", 4.5))
    parser.add_argument("--flow-shift", type=float, default=env_float("FLOW_SHIFT", 3.0))
    parser.add_argument("--ref-guidance-scale", type=float, default=env_float("REF_GUIDANCE_SCALE", 1.0))
    parser.add_argument("--prompt-year", type=int, default=env_int("PROMPT_YEAR", 2024))

    parser.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=env_bool("GRAD_CHECKPOINT", True))
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=env_bool("ALLOW_TF32", False))
    parser.add_argument("--python-bin", default=env_str("PYTHON_BIN", "python"))

    parser.add_argument("--frames", type=int, choices=[2, 3], default=env_int("FRAMES"))
    parser.add_argument("--timestep-sampling", choices=["sigmoid", "uniform"], default=env_str("TIMESTEP_SAMPLING", "sigmoid"))
    parser.add_argument("--sigmoid-scale", type=float, default=env_float("SIGMOID_SCALE", 1.0))
    parser.add_argument("--high-sigma-mix-prob", type=float, default=env_float("HIGH_SIGMA_MIX_PROB", 0.20))
    parser.add_argument("--high-sigma-min", type=float, default=env_float("HIGH_SIGMA_MIN", 0.8))
    parser.add_argument("--high-sigma-max", type=float, default=env_float("HIGH_SIGMA_MAX", 1.0))
    parser.add_argument("--ref-dropout-prob", type=float, default=env_float("REF_DROPOUT_PROB", 0.1))
    parser.add_argument("--ref-dropout-mode", default=env_str("REF_DROPOUT_MODE", "blank"))
    parser.add_argument("--ref-dropout-t3-mode", default=env_str("REF_DROPOUT_T3_MODE", "structured"))
    # Whole-caption CFG dropout: with this probability a training step swaps the
    # caption for the unconditional (empty) embedding, so the model learns text-CFG
    # and leans less on the reference. Independent of ref-dropout; does NOT touch CPM.
    parser.add_argument("--caption-dropout-prob", type=float, default=env_float("CAPTION_DROPOUT_PROB", 0.0))
    parser.add_argument("--empty-cap-path", default=env_str("EMPTY_CAP_PATH"))
    # Tag-level caption dropout: with this probability a step re-encodes the caption
    # after randomly dropping a subset of its delta tags (structural prefix + subject
    # anchors always kept, at least tag_keep_min delta tags kept). Trains the model on
    # short-but-non-empty captions so short user prompts stay in-distribution and the
    # model fills unstated attributes instead of copying the reference.
    parser.add_argument("--tag-dropout-prob", type=float, default=env_float("TAG_DROPOUT_PROB", 0.0))
    parser.add_argument("--tag-keep-prob", type=float, default=env_float("TAG_KEEP_PROB", 0.5))
    parser.add_argument("--tag-keep-min", type=int, default=env_int("TAG_KEEP_MIN", 3))
    # Identity-accessory injection: sig_subtract strips a character's signature tags
    # (>=45% of their images) out of the caption, so recurring identity accessories
    # (glasses, eyepatch, hat, halo...) become invisible to the text encoder and can
    # only ride the copy/identity pathway — which anti-copy then suppresses. This
    # re-injects a per-image, GT-faithful subset (signature ∩ accessory-type ∩ this
    # image's tags) into the caption so they become a controllable prompt handle.
    # inject-prob is a KEEP rate applied per word when the GT image has it; set it
    # higher than tag-keep-prob so accessories survive more often than generic tags.
    # Decoupled from tag-dropout: an inject-only step still re-encodes.
    parser.add_argument("--identity-inject-map", default=env_str("IDENTITY_INJECT_MAP"))
    parser.add_argument("--identity-inject-prob", type=float, default=env_float("IDENTITY_INJECT_PROB", 0.0))
    parser.add_argument("--head-loss-weight", type=float, default=env_float("HEAD_LOSS_WEIGHT", 1.0))
    parser.add_argument("--head-loss-sigma-cutoff", "--head-sigma-cutoff", dest="head_sigma_cutoff", type=float, default=env_float("HEAD_SIGMA_CUTOFF", 0.6))
    parser.add_argument("--head-conf", type=float, default=env_float("HEAD_CONF", 0.3))
    parser.add_argument("--head-margin", type=float, default=env_float("HEAD_MARGIN", 0.6))
    parser.add_argument("--head-loss-min-lat", type=int, default=env_int("HEAD_LOSS_MIN_LAT", 24))
    parser.add_argument("--head-loss-max-lat", type=int, default=env_int("HEAD_LOSS_MAX_LAT", 64))
    parser.add_argument("--weighting-scheme", default=env_str("WEIGHTING_SCHEME", "none"))
    parser.add_argument("--min-snr-gamma", type=float, default=env_float("MIN_SNR_GAMMA", 5.0))
    parser.add_argument(
        "--latent-recon-loss-weight",
        "--image-loss-weight",
        dest="latent_recon_loss_weight",
        type=float,
        default=env_float("LATENT_RECON_LOSS_WEIGHT", env_float("IMAGE_LOSS_WEIGHT", 0.0)),
    )
    parser.add_argument("--f1-anti-copy-weight", type=float, default=env_float("F1_ANTI_COPY_WEIGHT", 0.0))
    parser.add_argument("--f1-anti-copy-margin", type=float, default=env_float("F1_ANTI_COPY_MARGIN", 0.35))
    parser.add_argument("--f1-anti-copy-sigma-cutoff", type=float, default=env_float("F1_ANTI_COPY_SIGMA_CUTOFF", 0.6))
    parser.add_argument("--diff-loss-lambda", type=float, default=env_float("DIFF_LOSS_LAMBDA", 0.0))
    parser.add_argument("--diff-weight-min", type=float, default=env_float("DIFF_WEIGHT_MIN", 0.2))
    parser.add_argument(
        "--resume-data-skip",
        action=argparse.BooleanOptionalAction,
        default=env_bool("RESUME_DATA_SKIP", True),
        help="On resume, fast-forward the dataloader past start_step batches (loads+discards them from disk).",
    )
    parser.add_argument(
        "--pair-dhash-cache",
        default=env_str("PAIR_DHASH_CACHE", ""),
        help="Path to pair dhash cache (.pt) for 差分-aware ref pairing; empty -> <ccip cache dir>/pair_dhash_cache.pt, missing file -> random pairing.",
    )
    parser.add_argument("--pair-min-dhash", type=int, default=env_int("PAIR_MIN_DHASH", 25))
    parser.add_argument(
        "--singleton-ref-mode",
        choices=["self", "blank"],
        default=env_str("SINGLETON_REF_MODE", "self"),
        help="Single-image (character,bucket) cells: 'self' pairs the target with itself "
             "(legacy; pure copy signal), 'blank' forces the ref frames blanked so the "
             "sample trains the unconditional branch instead.",
    )
    parser.add_argument("--strict-nonfinite", action="store_true", default=env_bool("STRICT_NONFINITE", False))

    parser.add_argument("--no-ref-conditioner", action="store_true", default=env_bool("NO_REF_CONDITIONER", False))
    parser.add_argument("--no-adapter-gate", action="store_true", default=env_bool("NO_ADAPTER_GATE", False))
    parser.add_argument("--cpm", action="store_true", default=env_bool("CPM", False))
    parser.add_argument("--no-cpm-train-emb", action="store_true", default=env_bool("NO_CPM_TRAIN_EMB", False))
    parser.add_argument("--ccip-cache", default=env_str("ANIMA_REFLORA_CCIP_EMB_CACHE"))
    parser.add_argument("--head-roi-cache", default=env_str("ANIMA_REFLORA_HEAD_ROI_CACHE"))
    parser.add_argument("--cpm-identity-dim", type=int, default=env_int("CPM_IDENTITY_DIM", 768))
    parser.add_argument("--cpm-tokens", type=int, default=env_int("CPM_TOKENS", 4))
    parser.add_argument("--adapter-blocks", default=env_str("REF_ADAPTER_BLOCKS", "4,12,20"))
    parser.add_argument("--adapter-dim", type=int, default=env_int("REF_ADAPTER_DIM", 512))
    parser.add_argument("--adapter-heads", type=int, default=env_int("REF_ADAPTER_HEADS", 8))

    parser.add_argument("--crepa", action="store_true", default=env_bool("CREPA", False))
    parser.add_argument("--crepa-lambda", type=float, default=env_float("CREPA_LAMBDA", 0.1))
    parser.add_argument("--crepa-block", type=int, default=env_int("CREPA_BLOCK", 8))
    parser.add_argument("--crepa-sigma-cutoff", type=float, default=env_float("CREPA_SIGMA_CUTOFF", 0.0))
    parser.add_argument("--crepa-pool", choices=["global", "head_roi"], default=env_str("CREPA_POOL", "global"))

    parser.add_argument("--rope-refpos", action="store_true", default=env_bool("ROPE_REFPOS", False))
    parser.add_argument("--rope-layout", choices=["disjoint", "shifted", "packed"], default=env_str("ROPE_LAYOUT", "disjoint"))
    parser.add_argument("--rope-refpos-shift", "--rope-shift", dest="rope_shift", type=float, default=env_float("ROPE_SHIFT", 1.0))

    parser.add_argument("--ffl-weight", type=float, default=env_float("FFL_WEIGHT", 0.0))
    parser.add_argument("--ffl-alpha", type=float, default=env_float("FFL_ALPHA", 1.0))
    parser.add_argument("--train-arg", action="append", default=[])

    parser.add_argument("--backend", choices=["external", "tiny"], default=env_str("MODEL_BACKEND", "external"))
    parser.add_argument("--model-factory", default=env_str("ANIMA_REFLORA_MODEL_FACTORY"))
    parser.add_argument("--sd-scripts", default=env_str("ANIMA_REFLORA_SD_SCRIPTS"))
    parser.add_argument("--attn-mode", choices=["torch", "xformers", "flash", "sageattn", "sdpa"], default=env_str("ATTN_MODE", "torch"))
    parser.add_argument("--split-attn", action="store_true", default=env_bool("SPLIT_ATTN", False))
    parser.add_argument("--synthetic-data", action="store_true", default=env_bool("SYNTHETIC_DATA", False))
    parser.add_argument("--num-workers", type=int, default=env_int("NUM_WORKERS", 0))
    parser.add_argument("--device", default=env_str("DEVICE", "auto"))
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=env_str("DTYPE", "bf16"))
    parser.add_argument("--max-train-items", type=int, default=env_int("MAX_TRAIN_ITEMS"))
    parser.add_argument("--build-missing-head-cache", dest="build_missing_head_cache", action="store_true", default=env_bool("BUILD_MISSING_HEAD_CACHE", True))
    parser.add_argument("--no-build-missing-head-cache", dest="build_missing_head_cache", action="store_false")
    parser.add_argument("--head-crop-conf", type=float, default=env_float("HEAD_CROP_CONF", 0.4))
    parser.add_argument("--head-crop-padding", type=float, default=env_float("HEAD_CROP_PADDING", 1.8))
    parser.add_argument("--head-cache-shard", default=env_str("HEAD_CACHE_SHARD", "shard_auto_head.pt"))
    parser.add_argument("--image-root", default=env_str("ANIMA_REFLORA_IMAGES"))
    parser.add_argument("--image-source-prefix", default=env_str("IMAGE_SOURCE_PREFIX"))
    return parser


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    main_args, raw_args = split_raw_args(argv)
    main_args = normalize_train_arg_values(main_args)
    extra_train_args = shlex.split(os.environ.get("EXTRA_TRAIN_ARGS", ""))
    ns = build_parser().parse_args(main_args + raw_args + extra_train_args)
    stage = ns.stage or ns.stage_pos or "plan"
    run_tag = ns.run_tag or timestamp_tag()
    steps = ns.steps if ns.steps is not None else STAGE_DEFAULT_STEPS.get(stage, 35000)
    frames = ns.frames if ns.frames is not None else (2 if stage in {"plan", "tests"} else 3)
    run_name = ns.run_name or f"{stage}-{run_tag}"
    resume = ns.resume
    start_step = ns.start_step
    if start_step is None:
        start_step = parse_step_from_path(resume) or (0 if ns.from_scratch else parse_step_from_path(ns.base_ckpt) or 0)
    # --steps is the ABSOLUTE target step (train to step N), matching the original repo.
    # On resume the target must exceed the resumed step, else we would run 0 / negative
    # steps or silently over-train. Fail loud.
    if resume and steps <= start_step:
        raise SystemExit(
            f"--steps ({steps}) must exceed resume start-step ({start_step}); "
            "--steps is the absolute target step, not the number of extra steps."
        )
    paths = AnimaPaths.from_env(storage=ns.storage, out_dir=ns.out_dir)
    return TrainConfig(
        stage=stage,
        run_name=run_name,
        run_tag=run_tag,
        steps=steps,
        batch=ns.batch,
        lr=ns.lr,
        seed=ns.seed,
        network=ns.network,
        network_dim=ns.network_dim,
        network_alpha=ns.network_alpha,
        network_module=ns.network_module,
        network_args=ns.network_arg,
        optimizer=ns.optimizer,
        weight_decay=ns.weight_decay,
        storage=ns.storage,
        out_dir=ns.out_dir,
        allow_existing_run=ns.allow_existing_run,
        base_ckpt=ns.base_ckpt,
        from_scratch=ns.from_scratch,
        resume=resume,
        start_step=start_step,
        warmup_steps=ns.warmup_steps,
        resume_ref_conditioner=ns.resume_ref_conditioner,
        resume_crepa=ns.resume_crepa,
        ckpt_every=ns.ckpt_every,
        smoke_ckpt_every=ns.smoke_ckpt_every,
        log_every=ns.log_every,
        no_viz=ns.no_viz,
        ref_eval_every=ns.ref_eval_every,
        no_ref_eval=ns.no_ref_eval,
        ref_eval_refs=ns.ref_eval_refs,
        ref_eval_steps=ns.ref_eval_steps,
        ref_eval_seeds=ns.ref_eval_seeds,
        prompt_mode=ns.prompt_mode,
        strict_change_captions=ns.strict_change_captions,
        eval_prompt=ns.eval_prompt,
        negative_prompt=ns.negative_prompt,
        guidance_scale=ns.guidance_scale,
        flow_shift=ns.flow_shift,
        ref_guidance_scale=ns.ref_guidance_scale,
        prompt_year=ns.prompt_year,
        grad_checkpoint=ns.grad_checkpoint,
        tf32=ns.tf32,
        python_bin=ns.python_bin,
        frames=frames,
        timestep_sampling=ns.timestep_sampling,
        sigmoid_scale=ns.sigmoid_scale,
        high_sigma_mix_prob=ns.high_sigma_mix_prob,
        high_sigma_min=ns.high_sigma_min,
        high_sigma_max=ns.high_sigma_max,
        ref_dropout_prob=ns.ref_dropout_prob,
        ref_dropout_mode=ns.ref_dropout_mode,
        ref_dropout_t3_mode=ns.ref_dropout_t3_mode,
        caption_dropout_prob=ns.caption_dropout_prob,
        empty_cap_path=ns.empty_cap_path,
        tag_dropout_prob=ns.tag_dropout_prob,
        tag_keep_prob=ns.tag_keep_prob,
        tag_keep_min=ns.tag_keep_min,
        identity_inject_map=ns.identity_inject_map,
        identity_inject_prob=ns.identity_inject_prob,
        head_loss_weight=ns.head_loss_weight,
        head_sigma_cutoff=ns.head_sigma_cutoff,
        head_conf=ns.head_conf,
        head_margin=ns.head_margin,
        head_loss_min_lat=ns.head_loss_min_lat,
        head_loss_max_lat=ns.head_loss_max_lat,
        weighting_scheme=ns.weighting_scheme,
        min_snr_gamma=ns.min_snr_gamma,
        latent_recon_loss_weight=ns.latent_recon_loss_weight,
        f1_anti_copy_weight=ns.f1_anti_copy_weight,
        f1_anti_copy_margin=ns.f1_anti_copy_margin,
        f1_anti_copy_sigma_cutoff=ns.f1_anti_copy_sigma_cutoff,
        diff_loss_lambda=ns.diff_loss_lambda,
        diff_weight_min=ns.diff_weight_min,
        resume_data_skip=ns.resume_data_skip,
        pair_dhash_cache=ns.pair_dhash_cache or str(Path(ns.ccip_cache or str(paths.ccip_cache)).parent / "pair_dhash_cache.pt"),
        pair_min_dhash=ns.pair_min_dhash,
        singleton_ref_mode=ns.singleton_ref_mode,
        strict_nonfinite=ns.strict_nonfinite,
        no_ref_conditioner=ns.no_ref_conditioner,
        no_adapter_gate=ns.no_adapter_gate,
        cpm=ns.cpm,
        no_cpm_train_emb=ns.no_cpm_train_emb,
        ccip_cache=ns.ccip_cache or str(paths.ccip_cache),
        head_roi_cache=ns.head_roi_cache or str(paths.head_roi_cache),
        cpm_identity_dim=ns.cpm_identity_dim,
        cpm_tokens=ns.cpm_tokens,
        adapter_blocks=ns.adapter_blocks,
        adapter_dim=ns.adapter_dim,
        adapter_heads=ns.adapter_heads,
        crepa=ns.crepa,
        crepa_lambda=ns.crepa_lambda,
        crepa_block=ns.crepa_block,
        crepa_sigma_cutoff=ns.crepa_sigma_cutoff,
        crepa_pool=ns.crepa_pool,
        rope_refpos=ns.rope_refpos,
        rope_layout=ns.rope_layout,
        rope_shift=ns.rope_shift,
        ffl_weight=ns.ffl_weight,
        ffl_alpha=ns.ffl_alpha,
        train_args=ns.train_arg,
        raw_train_args=raw_args,
        extra_train_args=extra_train_args,
        backend=ns.backend,
        model_factory=ns.model_factory,
        sd_scripts=ns.sd_scripts,
        attn_mode="torch" if ns.attn_mode == "sdpa" else ns.attn_mode,
        split_attn=ns.split_attn,
        synthetic_data=ns.synthetic_data,
        num_workers=ns.num_workers,
        device=ns.device,
        dtype=ns.dtype,
        max_train_items=ns.max_train_items,
        build_missing_head_cache=ns.build_missing_head_cache,
        head_crop_conf=ns.head_crop_conf,
        head_crop_padding=ns.head_crop_padding,
        head_cache_shard=ns.head_cache_shard,
        image_root=ns.image_root,
        image_source_prefix=ns.image_source_prefix,
    )
