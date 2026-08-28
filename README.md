# AnimaRefLoRA

Reference-based character identity transfer for the Anima anime diffusion
transformer — and the training recipe that prevents the reference from
becoming a copy shortcut.

AnimaRefLoRA gives the text-only Anima DiT a one-shot character reference: a
head crop and the full reference image are placed beside the noised target as
in-context frames, while the text prompt controls pose, outfit, and scene. The
central failure mode is the *reference-copy shortcut* — when the reference is
easier to follow than the prompt, the model reproduces its composition instead
of performing the requested edit. The released recipe counters it with
difference-weighted flow matching, near-duplicate-aware pairing, tag-level
caption dropout, structured reference dropout, and anime-specific identity
conditioning (CCIP prototype tokens + head-region representation alignment).

📖 **Full technical report:** [blog/index.html](blog/index.html)
([中文版](blog/index_zh.html)) — method, ablations, and a controlled
qualitative study of the 500K checkpoint on two AI-generated original
characters.

## Repository Layout

| Path | Contents |
| --- | --- |
| `anima_reflora/` | Training package: three-frame data pipeline, LoKr training loop, cache builders, inference |
| `sd-scripts/` | Vendored fork of kohya-ss/sd-scripts with the Anima loader (Apache-2.0, see its `LICENSE.md`) |
| `scripts/` | Run recipes, cache/merge utilities, plugin dist builder, report tooling |
| `comfyui/ComfyUI-AnimaRefLora/` | ComfyUI plugin (self-contained dist build with vendored dependencies) |
| `blog/` | Technical report pages (English default, `index_zh.html` for Traditional Chinese) |
| `tests/` | Unit tests for the training package |

## Model Weights

Weights are published on HuggingFace, not in this repo. The
`idinject_500k.animaref.safetensors` bundle packs the LoKr adapter plus the
identity conditioning modules; both the ComfyUI plugin and the local
inference scripts consume it directly.

The adapter is trained and verified on Anima Base v1.0 (it also loads onto
same-architecture community finetunes, see the base-swap section of the
report). Support for the larger community-expanded Anima variants
(2.9B/3B-class layer expansions of the 2B base) has not yet been verified.

## ComfyUI Plugin

Copy `comfyui/ComfyUI-AnimaRefLora` into your ComfyUI `custom_nodes/`
directory and restart. The dist build vendors its `anima_reflora` and
`sd-scripts` dependencies, so no extra checkout is needed. See the plugin's
[README](comfyui/ComfyUI-AnimaRefLora/README.md) for node usage and model
placement.

## Training

The training harness targets Docker/RunPod. Model weights, datasets, latent
caches, checkpoints, and credentials are mounted under `/workspace/storage`
or synchronized at runtime — never baked into the image.

### Storage layout

Default paths can all be overridden with environment variables.

- App: `/opt/AnimaRefLora`
- Runs / caches (local): `/opt/AnimaRefLora/runs`
- Storage (mounted/S3): `/workspace/storage`
- Native latent cache: `/workspace/storage/_latcache`
- Dataset (extracted): `/workspace/storage/dataset`
- Base DiT: `/workspace/storage/anima_models/diffusion_models/anima-base-v1.0.safetensors`
- Text encoder dir: `/workspace/storage/anima_models/text_encoders` (expects `qwen_3_06b_base.safetensors`)
- VAE: `/workspace/storage/anima_models/vae/qwen_image_vae.safetensors`

### Docker

```bash
docker build \
  --build-arg BASE_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  -t anima-reflora-runpod:latest \
  .
```

Run a training stage:

```bash
docker run --gpus all --rm -it \
  -v /path/to/storage:/workspace/storage \
  anima-reflora-runpod:latest \
  train rope-smoke --batch 1 --tf32 --log-every 10 --no-viz --network lokr
```

Preflight before a cloud run:

```bash
docker run --gpus all --rm -it \
  -v /path/to/storage:/workspace/storage \
  anima-reflora-runpod:latest \
  preflight --stage cpm-short --cpm --ccip-cache /opt/AnimaRefLora/runs/ccip_ref_head_emb_cache.pt
```

Stage wrapper (inside the container or a configured environment):

```bash
./scripts/train_plan.sh plan
./scripts/train_plan.sh tests
./scripts/train_plan.sh rope-smoke --batch 2 --no-grad-checkpoint --tf32 --log-every 10 --no-viz
./scripts/train_plan.sh from0-headroi-rope-cpm --batch 2 --tf32 --steps 100000
```

### Local smoke test (no GPU weights needed)

```bash
python -m pytest -q
python -m anima_reflora.train \
  --stage rope-smoke \
  --run-name local-tiny-smoke \
  --out-dir /tmp/anima-reflora-runs \
  --storage /tmp/anima-reflora-storage \
  --steps 2 --ckpt-every 1 --batch 2 \
  --backend tiny --synthetic-data --dtype fp32 \
  --from-scratch --allow-existing-run
```

`--backend tiny --synthetic-data` is for mechanical smoke tests only. Real
training uses `--backend external` with the vendored `sd-scripts/` (set
`ANIMA_REFLORA_SD_SCRIPTS` if the checkout is elsewhere).

## Reference Frame Contract

T=3 training is strict reference training — not ControlNet and not
single-frame LoRA:

- `F0`: cropped head reference latent
- `F1`: full reference latent from the same source image as `F0`
- `F2`: target/result latent

Only `F2` is noised. `F0` and `F1` keep timestep `0`, and loss is computed
only from the model prediction on `F2`. The primary loss is flow velocity MSE
against `noise - F2_latent`, with difference-weighted per-pixel scaling so
changed regions receive more gradient than regions identical to the
reference. Optional `--latent-recon-loss-weight` adds an L1 loss on the
denoised target latent.

Identity should come from the reference frames, not text: the default
`--prompt-mode change_only` prefers identity-stripped cached captions, and
tag-level caption dropout keeps short prompts from collapsing into
copy-the-reference behavior.

## Cache Builders

Build the native training cache (latents + prompt tensors) from raw images:

```bash
docker run --gpus all --rm -it \
  -v /path/to/storage:/workspace/storage \
  anima-reflora-runpod:latest \
  training-cache \
    --storage /workspace/storage \
    --image-root /workspace/storage/dataset/images \
    --metadata /workspace/storage/dataset/records.jsonl \
    --output-cache /workspace/storage/_latcache
```

Then build the feature caches used by identity conditioning:

```bash
# head-ROI masks for difference weighting / representation alignment
docker run --gpus all --rm -it -v /path/to/storage:/workspace/storage \
  anima-reflora-runpod:latest \
  head-roi-cache --storage /workspace/storage --image-root /workspace/storage/dataset/images

# per-character CCIP prototypes from detected head crops
docker run --gpus all --rm -it -v /path/to/storage:/workspace/storage \
  anima-reflora-runpod:latest \
  head-ccip-cache \
    --storage /workspace/storage \
    --image-root /workspace/storage/dataset/images \
    --output /opt/AnimaRefLora/runs/ccip_ref_head_emb_cache.pt
```

Metadata can be JSON, JSONL, or CSV (`path`/`image_path`/`file`, `caption`,
`change_caption`, `target_caption`, `character`, `ref_eligible`); without
metadata the builder scans images recursively and uses sidecar `.txt`
captions.

## Run Outputs

Every run writes:

```text
/opt/AnimaRefLora/runs/experiments/<RUN_NAME>/
  checkpoints/
  tb/
  viz/
  ref_use/
  logs/
```

Existing run folders fail fast unless `--allow-existing-run` is set. Explicit
`--resume` expects the matching `optimizer_step_<N>.pt` next to
`lora_step_<N>.safetensors`; warm-start `--base-ckpt` skips missing sidecars.

## License Notes

`sd-scripts/` is a vendored fork of
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts); its original
license is preserved in `sd-scripts/LICENSE.md`.
