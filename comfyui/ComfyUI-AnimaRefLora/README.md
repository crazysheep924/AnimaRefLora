# ComfyUI-AnimaRefLora

Standalone ComfyUI nodes for AnimaRefLora. Generation always goes through
`anima_reflora.local_ref_ab_infer.sample_target()` — the same inference path
as the training repo's REF evaluation. It does not use ComfyUI's KSampler,
so it is not subject to ComfyUI's Anima RoPE `max_h=120` limit.

## Install

This folder is already a self-contained portable build (it vendors the
minimal `anima_reflora/` and `sd-scripts/` subsets needed for inference).
Copy it into ComfyUI and install the Python dependencies:

```text
ComfyUI/custom_nodes/ComfyUI-AnimaRefLora
```

```bat
:: Windows portable example
python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-AnimaRefLora\requirements.txt
```

(To regenerate this build from the source repo, run
`scripts/build_comfyui_plugin_dist.sh` at the repo root; the output lands in
`dist/ComfyUI-AnimaRefLora`.)

## Model placement (recommended: single-file bundle)

```text
models/diffusion_models/anima-base-v1.0.safetensors
models/text_encoders/model.safetensors                    ← must be named model.safetensors
models/vae/qwen_image_vae.safetensors
models/anima_reflora/<release>.animaref.safetensors       ← one file = the whole RefLora model
models/loras/Anima/extra_style_lora.safetensors           ← (optional) regular style LoRA
```

- The text encoder is the Qwen3-0.6B base encoder; the loader resolves the
  *directory* of the file you select and expects `model.safetensors` inside
  it, so name (or symlink) the file exactly that.
- The RefLora bundle (e.g. `idinject_500k.animaref.safetensors`) is
  distributed via HuggingFace — see the root README's Model Weights section.
  Drop it into `models/anima_reflora/`.

> Note: verified on Anima Base v1.0 only. Support for the community
> layer-expanded Anima variants (2.9B/3B-class) has not been confirmed.

The `.animaref.safetensors` bundle packs the LoKr weights, the identity
modules (ref_conditioner / crepa_projector), the feature config, and the
RoPE layout into a single safetensors file. The Loader's `checkpoint`
dropdown lists only the bundles in `models/anima_reflora/` — pick one file
and everything matches; no mismatched-step problems. Bundles are packed on
the training side:

```bash
python scripts/pack_animaref_bundle.py <run_dir> --latest --name my_model \
    -o my_model.animaref.safetensors
```

**Legacy multi-file format is still supported**: if `models/anima_reflora/`
contains no bundle, the dropdown falls back to `lora_step_*.safetensors`
under `models/loras/`, and the same directory must then hold the
matching-step `ref_conditioner_step_*` / `crepa_projector_step_*` /
`feature_config_step_*.json` / `rope_refpos_step_*.json` files.

## Node graph

```text
Anima Extra LoRA (optional) ─► Anima RefLora Loader ─┐
                                                      ├─► Anima Ref Encode ─► Anima RefLora Sampler
Load Image ───────────────────────────────────────────┘
```

- **Anima Extra LoRA (standalone)**: optional extra Anima DiT LoRA. Supports
  the `lora_unet_*.lora_down/up.weight` format; text-encoder LoRAs are not
  applied.
- **Anima RefLora Loader**: loads the base model, RefLora/LoKr, ref
  conditioner, CPM, RoPE layout, and VAE.
- **Anima Ref Encode**: turns the reference image into head and full
  reference latents plus the CCIP identity embedding.
- **Anima RefLora Sampler**: generates with the same RF sampling loop as the
  repo.

Output size is set on `Anima Ref Encode` via `generation_width` /
`generation_height`, both in steps of 64 pixels. Examples:

```text
1024 × 1024  square
1024 × 576   landscape 16:9
576 × 1024   portrait 9:16
1152 × 768   landscape 3:2
768 × 1152   portrait 2:3
```

Larger sizes need more VRAM for DiT attention and the VAE. The reference
image is letterboxed to the target aspect ratio by default, so characters
are not cropped to fit the output ratio.

With an extra LoRA attached, the sampling implementation is unchanged, but
outputs will of course no longer be bit-identical to the LoRA-free REF
evaluation. Regular extra LoRAs are selected with the `Anima Extra LoRA`
node and connected to the Loader's `extra_lora` input.
