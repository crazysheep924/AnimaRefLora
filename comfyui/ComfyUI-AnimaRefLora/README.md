# ComfyUI-AnimaRefLora

AnimaRefLora 的 ComfyUI standalone 節點。生成固定走
`anima_reflora.local_ref_ab_infer.sample_target()`，與 repo 的 REF 推論路徑相同，
不使用 ComfyUI KSampler，因此不受 ComfyUI Anima RoPE `max_h=120` 限制。

## 節點流程

```text
Anima Extra LoRA（可選） ─► Anima RefLora Loader ─┐
                                                   ├─► Anima Ref Encode ─► Anima RefLora Sampler
Load Image ────────────────────────────────────────┘
```

- **Anima Extra LoRA (standalone)**：可選的額外 Anima DiT LoRA。支援
  `lora_unet_*.lora_down/up.weight` 格式；Text Encoder LoRA 不會套用。
- **Anima RefLora Loader**：載入 base、RefLora/LoKr、ref conditioner、CPM、RoPE、VAE。
- **Anima Ref Encode**：參照圖轉成頭部與全圖 reference latent，加上 CCIP identity。
- **Anima RefLora Sampler**：使用 repo 相同的 RF sampling loop 生成圖片。

輸出尺寸在 `Anima Ref Encode` 的 `generation_width`、`generation_height` 設定，兩者以
64 pixels 為步進。例如：

```text
1024 × 1024  方形
1024 × 576   橫向 16:9
576 × 1024   直向 9:16
1152 × 768   橫向 3:2
768 × 1152   直向 2:3
```

尺寸越大，DiT attention 與 VAE 所需的 VRAM 越高。參照圖預設會 letterbox 到指定比例，
避免為了配合輸出比例而裁掉角色邊緣。

加上額外 LoRA 後，sampling 實作仍相同，但輸出當然不再與未加 LoRA 的 REF 評測逐位一致。

## 建立與安裝

從 repo 根目錄建立可攜式發佈包：

```bash
scripts/build_comfyui_plugin_dist.sh
```

將 `dist/ComfyUI-AnimaRefLora` 複製到：

```text
ComfyUI/custom_nodes/ComfyUI-AnimaRefLora
```

發佈包已包含推論所需的 `anima_reflora/` 與 `sd-scripts/` 最小子集。安裝依賴；
Windows portable 版例如：

```bat
python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-AnimaRefLora\requirements.txt
```

模型放置（**建議：單檔 bundle**）：

```text
models/diffusion_models/anima-base-v1.0.safetensors
models/text_encoders/model.safetensors
models/vae/qwen_image_vae.safetensors
models/anima_reflora/idinject_485k.animaref.safetensors   ← 一個檔＝整套 RefLora 模型
models/loras/Anima/extra_style_lora.safetensors           ← （選用）一般風格 LoRA
```

> 注意：目前僅在 Anima Base v1.0 上驗證。社群以層擴充放大的 Anima 變體
> （2.9B／3B 級）是否支援尚未確認。

`.animaref.safetensors` bundle 把 LoKr 權重、身分模組（ref_conditioner /
crepa_projector）、feature 設定與 RoPE 配置打包成一個 safetensors；Loader 的
`checkpoint` 下拉只列 `models/anima_reflora/` 裡的 bundle，選一個檔就完成，
不會再有步數不成套的問題。bundle 由訓練端打包：

```bash
python scripts/pack_animaref_bundle.py <run_dir> --latest --name my_model \
    -o my_model.animaref.safetensors
```

**Legacy 多檔格式仍支援**：若 `models/anima_reflora/` 沒有任何 bundle，下拉會
退回列出 `models/loras/` 的 `lora_step_*.safetensors`，此時同目錄需放齊同步數的
`ref_conditioner_step_*` / `crepa_projector_step_*` / `feature_config_step_*.json` /
`rope_refpos_step_*.json`。

一般額外 LoRA 由 `Anima Extra LoRA` 節點選擇並接到 Loader 的 `extra_lora`。
