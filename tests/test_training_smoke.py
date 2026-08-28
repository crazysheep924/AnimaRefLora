from pathlib import Path
import json

import torch

from anima_reflora.models import TinyAnimaRefModel
from anima_reflora.preflight import run_preflight
from anima_reflora.train import keep_scalar_trainables_fp32, main


def test_scalar_trainables_keep_fp32_optimizer_state_under_bf16():
    model = TinyAnimaRefModel().to(dtype=torch.bfloat16)
    kept = keep_scalar_trainables_fp32(model)

    gate = model.ref_conditioner.gate
    assert "ref_conditioner.gate" in kept
    assert gate.dtype == torch.float32
    assert model.net[0].weight.dtype == torch.bfloat16

    opt = torch.optim.AdamW([gate], lr=1e-5)
    gate.grad = torch.tensor(1e-4)
    opt.step()
    assert opt.state[gate]["exp_avg"].dtype == torch.float32


def test_tiny_training_smoke(tmp_path):
    out = tmp_path / "runs"
    code = main(
        [
            "--stage",
            "rope-smoke",
            "--run-name",
            "tiny-smoke",
            "--out-dir",
            str(out),
            "--storage",
            str(tmp_path / "storage"),
            "--steps",
            "2",
            "--ckpt-every",
            "1",
            "--batch",
            "2",
            "--backend",
            "tiny",
            "--synthetic-data",
            "--dtype",
            "fp32",
            "--from-scratch",
            "--no-viz",
            "--no-ref-eval",
            "--allow-existing-run",
        ]
    )
    assert code == 0
    run_dir = out / "experiments" / "tiny-smoke"
    assert (run_dir / "checkpoints" / "lora_step_1.safetensors").exists()
    assert (run_dir / "checkpoints" / "lora_step_2.safetensors").exists()
    assert (run_dir / "checkpoints" / "optimizer_step_2.pt").exists()
    assert (run_dir / "logs" / "done.json").exists()


def test_tiny_training_resume_loads_optimizer_state(tmp_path):
    out = tmp_path / "runs"
    storage = tmp_path / "storage"
    base_args = [
        "--stage",
        "rope-smoke",
        "--out-dir",
        str(out),
        "--storage",
        str(storage),
        "--ckpt-every",
        "1",
        "--batch",
        "2",
        "--backend",
        "tiny",
        "--synthetic-data",
        "--dtype",
        "fp32",
        "--no-viz",
        "--no-ref-eval",
    ]
    assert main([*base_args, "--run-name", "tiny-first", "--steps", "2", "--from-scratch"]) == 0
    checkpoint = out / "experiments" / "tiny-first" / "checkpoints" / "lora_step_2.safetensors"

    # --steps is the ABSOLUTE target step: resuming a step-2 ckpt to step 3 runs 1 more step.
    assert main([*base_args, "--run-name", "tiny-resume", "--steps", "3", "--resume", str(checkpoint)]) == 0
    run_dir = out / "experiments" / "tiny-resume"
    assert (run_dir / "checkpoints" / "lora_step_3.safetensors").exists()
    assert (run_dir / "checkpoints" / "optimizer_step_3.pt").exists()
    log_text = (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "optimizer loaded=" in log_text
    assert '"start_step": 2' in log_text

    report = run_preflight([*base_args, "--run-name", "tiny-preflight-resume", "--steps", "3", "--resume", str(checkpoint)])
    labels = {item["label"] for item in report["checks"]}
    assert "resume checkpoint" in labels
    assert "resume optimizer state" in labels
    assert "resume feature sidecar" in labels


def test_tiny_feature_flags_create_observable_artifacts(tmp_path):
    storage = tmp_path / "storage"
    out = tmp_path / "runs"
    ccip_path = storage / "runs" / "ccip_ref_emb_cache.pt"
    head_roi_path = storage / "runs" / "head_roi_cache.pt"
    ccip_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = {f"synthetic/ref_{i}.latent": torch.ones(8) * (i + 1) for i in range(16)}
    masks = {f"synthetic/target_{i}.latent": torch.ones(16, 16) for i in range(16)}
    torch.save(embeddings, ccip_path)
    torch.save(masks, head_roi_path)

    code = main(
        [
            "--stage",
            "rope-smoke",
            "--run-name",
            "tiny-feature-smoke",
            "--out-dir",
            str(out),
            "--storage",
            str(storage),
            "--steps",
            "2",
            "--ckpt-every",
            "1",
            "--ref-eval-every",
            "1",
            "--batch",
            "2",
            "--backend",
            "tiny",
            "--synthetic-data",
            "--dtype",
            "fp32",
            "--from-scratch",
            "--allow-existing-run",
            "--rope-refpos",
            "--cpm",
            "--crepa",
            "--crepa-pool",
            "head_roi",
            "--head-loss-weight",
            "2.0",
            "--latent-recon-loss-weight",
            "0.1",
            "--ccip-cache",
            str(ccip_path),
            "--head-roi-cache",
            str(head_roi_path),
        ]
    )
    assert code == 0
    run_dir = out / "experiments" / "tiny-feature-smoke"
    assert (run_dir / "checkpoints" / "feature_config_step_2.json").exists()
    assert (run_dir / "checkpoints" / "rope_refpos_step_2.json").exists()
    assert (run_dir / "checkpoints" / "cpm_adapter_step_2.safetensors").exists()
    assert (run_dir / "checkpoints" / "crepa_projector_step_2.safetensors").exists()
    assert (run_dir / "viz" / "step_2" / "metadata.json").exists()
    assert (run_dir / "ref_use" / "step_2" / "metrics.json").exists()
    viz_meta = json.loads((run_dir / "viz" / "step_2" / "metadata.json").read_text(encoding="utf-8"))
    assert viz_meta["checkpoint"].endswith("lora_step_2.safetensors")
    assert viz_meta["seed"] == 1234
    assert viz_meta["prompt"] == "standing, cowboy shot, white dress, simple background, looking at viewer"
    assert viz_meta["guidance_scale"] == 4.5
    assert viz_meta["flow_shift"] == 3.0
    assert viz_meta["ref_guidance_scale"] == 1.0
    assert viz_meta["prompt_mode"] == "change_only"
    assert viz_meta["condition_label"] == "train_batch_latent_panel"
    eval_meta = json.loads((run_dir / "ref_use" / "step_2" / "metrics.json").read_text(encoding="utf-8"))
    assert eval_meta["checkpoint"].endswith("lora_step_2.safetensors")
    assert eval_meta["eval_kind"] == "training_batch_proxy"
    assert eval_meta["condition_labels"] == ["train_batch_reference"]
    assert eval_meta["prompt"] == "standing, cowboy shot, white dress, simple background, looking at viewer"
    log_text = (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "cpm_valid_fraction" in log_text
    assert "crepa" in log_text
    assert "head_roi_valid_fraction" in log_text


def test_head_roi_missing_batch_falls_back_to_uniform_loss(tmp_path):
    storage = tmp_path / "storage"
    out = tmp_path / "runs"
    ccip_path = storage / "runs" / "ccip_ref_emb_cache.pt"
    head_roi_path = storage / "runs" / "head_roi_cache.pt"
    ccip_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({f"synthetic/ref_{i}.latent": torch.ones(8) for i in range(16)}, ccip_path)
    torch.save({"unrelated-target.latent": torch.ones(16, 16)}, head_roi_path)

    code = main(
        [
            "--stage",
            "from0-headroi-rope-cpm",
            "--run-name",
            "tiny-missing-head-roi",
            "--out-dir",
            str(out),
            "--storage",
            str(storage),
            "--steps",
            "1",
            "--ckpt-every",
            "1",
            "--batch",
            "1",
            "--backend",
            "tiny",
            "--synthetic-data",
            "--dtype",
            "fp32",
            "--from-scratch",
            "--allow-existing-run",
            "--rope-refpos",
            "--cpm",
            "--crepa",
            "--crepa-pool",
            "head_roi",
            "--head-loss-weight",
            "4.0",
            "--ccip-cache",
            str(ccip_path),
            "--head-roi-cache",
            str(head_roi_path),
            "--no-viz",
            "--no-ref-eval",
        ]
    )

    assert code == 0
    log_text = (out / "experiments" / "tiny-missing-head-roi" / "logs" / "train.log").read_text(encoding="utf-8")
    assert '"head_roi_valid_fraction": 0.0' in log_text
