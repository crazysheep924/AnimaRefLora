import pickle

import pytest
import torch

from anima_reflora.config import parse_config
from anima_reflora.preflight import run_preflight
from anima_reflora.validation import UnsupportedFeatureError, validate_supported_training_features


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("STEPS", "10")
    monkeypatch.setenv("BATCH", "1")
    cfg = parse_config(["rope-smoke", "--steps", "3", "--batch", "2", "--backend", "tiny", "--synthetic-data"])
    assert cfg.stage == "rope-smoke"
    assert cfg.steps == 3
    assert cfg.batch == 2
    assert cfg.backend == "tiny"
    assert cfg.synthetic_data is True
    assert cfg.raw_train_args == []
    assert cfg.prompt_mode == "change_only"
    assert cfg.guidance_scale == 4.5
    assert cfg.flow_shift == 3.0
    assert cfg.ref_guidance_scale == 1.0
    assert cfg.prompt_year == 2024
    assert cfg.eval_prompt == "standing, cowboy shot, white dress, simple background, looking at viewer"
    assert "worst quality" in cfg.negative_prompt


def test_raw_args_override_known_args():
    cfg = parse_config(["--stage", "cpm-short", "--train-arg", "--foo", "--", "--ref-eval-refs", "3", "--seed", "123"])
    assert cfg.stage == "cpm-short"
    assert cfg.train_args == ["--foo"]
    assert cfg.raw_train_args == ["--ref-eval-refs", "3", "--seed", "123"]
    assert cfg.ref_eval_refs == 3
    assert cfg.seed == 123


def test_unknown_raw_args_fail_loudly():
    with pytest.raises(SystemExit):
        parse_config(["--stage", "cpm-short", "--", "--bar", "1"])


def test_unimplemented_train_arg_escape_hatch_fails_loudly():
    cfg = parse_config(["--stage", "rope-short", "--backend", "external", "--train-arg", "--foo"])
    try:
        validate_supported_training_features(cfg)
    except UnsupportedFeatureError as exc:
        assert "--train-arg" in str(exc)
    else:
        raise AssertionError("--train-arg is not wired and should fail loudly")


def test_key_value_train_arg_is_supported_for_external_network_kwargs():
    cfg = parse_config(["--stage", "rope-short", "--backend", "external", "--train-arg", "factor=8"])
    validate_supported_training_features(cfg)
    assert cfg.train_args == ["factor=8"]


def test_train_arg_is_rejected_for_tiny_backend():
    cfg = parse_config(["--stage", "rope-smoke", "--backend", "tiny", "--train-arg", "factor=8"])
    with pytest.raises(UnsupportedFeatureError):
        validate_supported_training_features(cfg)


def test_feature_flags_pass_validation_after_wiring():
    cfg = parse_config(
        [
            "--stage",
            "rope-short",
            "--backend",
            "external",
            "--rope-refpos",
            "--cpm",
            "--crepa",
            "--head-loss-weight",
            "4.0",
            "--no-viz",
            "--no-ref-eval",
        ]
    )
    validate_supported_training_features(cfg)


def test_minimal_external_training_config_passes_feature_validation():
    cfg = parse_config(["--stage", "rope-short", "--backend", "external", "--no-viz", "--no-ref-eval"])
    validate_supported_training_features(cfg)


def test_preflight_respects_sd_scripts_arg(tmp_path):
    sd_scripts = tmp_path / "sd-scripts"
    sd_scripts.mkdir()
    (sd_scripts / "anima_train_network.py").write_text("# test\n", encoding="utf-8")
    report = run_preflight(
        [
            "--stage",
            "preflight",
            "--backend",
            "external",
            "--synthetic-data",
            "--sd-scripts",
            str(sd_scripts),
            "--storage",
            str(tmp_path / "storage"),
            "--out-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "uses-cli-sd-scripts",
        ]
    )
    check = next(item for item in report["checks"] if item["label"] == "sd-scripts Anima trainer")
    assert check["path"] == str(sd_scripts / "anima_train_network.py")


def test_preflight_checks_all_resampled_reference_candidates_for_ccip(tmp_path):
    storage = tmp_path / "storage"
    latcache = storage / "_latcache"
    latcache.mkdir(parents=True)
    for path in [
        storage / "anima_models" / "diffusion_models" / "anima-base-v1.0.safetensors",
        storage / "anima_models" / "vae" / "qwen_image_vae.safetensors",
        storage / "anima_models" / "text_encoders" / "qwen_3_06b_base.safetensors",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    bucket = (16, 16)
    image_paths = [f"/cache/images/{idx}.webp" for idx in range(1, 4)]
    lat_idx = {}
    meta = {}
    for idx, image_path in enumerate(image_paths, start=1):
        lat_idx[(image_path, bucket, "full")] = 0
        lat_idx[(image_path, bucket, "head")] = 0
        meta[idx] = {
            "path": image_path,
            "character": "char-a",
            "ref_eligible": True,
            "caption": f"change {idx}",
            "change_caption": f"change {idx}",
            "has_head": True,
        }
    with (latcache / "_cache_index.pkl").open("wb") as fh:
        pickle.dump({"version": 3, "sig": [("shard_000.pt", 0, 0)], "lat_idx": lat_idx, "cap_idx": {}, "meta": meta}, fh)

    ccip_path = storage / "runs" / "ccip_ref_emb_cache.pt"
    ccip_path.parent.mkdir(parents=True)
    # With --max-train-items=1, the deterministic first pair uses image_paths[1]
    # as pair.ref_full. The other candidates can still be sampled on later epochs.
    torch.save({image_paths[1]: torch.ones(8)}, ccip_path)

    with pytest.raises(ValueError, match="CCIP cache missing"):
        run_preflight(
            [
                "--stage",
                "rope-short",
                "--backend",
                "tiny",
                "--storage",
                str(storage),
                "--out-dir",
                str(tmp_path / "runs"),
                "--run-name",
                "ccip-candidate-coverage",
                "--max-train-items",
                "1",
                "--cpm",
                "--ccip-cache",
                str(ccip_path),
            ]
        )
